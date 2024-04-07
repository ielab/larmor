import gc
import os
import random
from utils.distances import get_binary_entropy
import json
import rbo

import torch
import numpy as np
import scipy.stats as stats

from transformers import AutoTokenizer

from utils.get_args import get_args
from utils.distances import calculate_frechet_distance, stats_for_frechet, entropy
from utils.read_and_write import (read_doc_enc_from_pickle, get_embedding_subset, load_search_results,
                                  transform_search_results, save_search_results, load_eval_results)
from utils.scores import get_score

from utils.qpp_post import SIGMA_MAX, NQC, SMV
from encoding_and_eval import search
from data.dataset_collection import Datasets
from beir.retrieval.evaluation import EvaluateRetrieval
from model.model_zoo import CustomModel, BeirModels
from utils.fusion import fuse_runs
from ranx import Qrels, Run
from ranx import fuse



def query_similarity(models, datasets, model_name, queries_target):
    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False

    model = models.load_model(model_name, model_name_or_path=None, cuda=cuda)
    if models.source_datasets[model_name] == "nq":
        split = "test"
    else:
        split = "train"

    # Load source queries
    _, queries_source, qrels_source = datasets.load_dataset(models.source_datasets[model_name],
                                                            load_corpus=False, split=split)
    # Get query embeddings
    query_list_t = [queries_target[qid] for qid in queries_target]
    query_embeds_t = model.encode_queries(query_list_t, batch_size=32, show_progress_bar=True,
                                          convert_to_tensor=False)
    query_list_s = [queries_source[qid] for qid in queries_source]
    query_embeds_s = model.encode_queries(query_list_s, batch_size=32, show_progress_bar=True,
                                          convert_to_tensor=False)
    # For each target query, search for the closest source query
    scores, indices = search(query_embeddings=query_embeds_t, doc_embeddings=query_embeds_s,
                             top_k=1, score_function=models.score_function[model_name])

    return scores, indices


def _batched_frechet(doc_embeds_s, doc_embeds_t, subsample_size):
    if len(doc_embeds_s) > subsample_size or len(doc_embeds_t) > subsample_size:
        frechet_array = []
        # cuda_cka = CudaCKA()
        # device = torch.device('cuda')
        for i in range(10):
            # print("Prior to embedding subsets")
            doc_embeds_s_small = get_embedding_subset(doc_embeds_s, subsample_size=subsample_size)
            doc_embeds_t_small = get_embedding_subset(doc_embeds_t, subsample_size=subsample_size)
            # print("After target embedding stats are loaded")
            # frechet = cuda_cka.linear_CKA(torch.tensor(doc_embeds_s_small.T, device=device),
            #                              torch.tensor(doc_embeds_t_small.T, device=device))
            mu_s, sigma_s = stats_for_frechet(doc_embeds_s_small)
            mu_t, sigma_t = stats_for_frechet(doc_embeds_t_small)
            # print("Cov shape", sigma_s.shape, sigma_t.shape)
            frechet = calculate_frechet_distance(mu_s, sigma_s, mu_t, sigma_t)
            frechet_array.append(frechet.cpu().item())
        # print("After CKA loss")
        frechet = np.mean(frechet_array)
        del doc_embeds_s, doc_embeds_t, doc_embeds_s_small, doc_embeds_t_small
        gc.collect()

    else:
        # raise "Should not be called"
        # print("Prior to MMD loss")
        # mmd_loss = MMDLoss()
        # frechet = mmd_loss(torch.tensor(doc_embeds_s), torch.tensor(doc_embeds_t))
        mu_s, sigma_s = stats_for_frechet(doc_embeds_s)
        mu_t, sigma_t = stats_for_frechet(doc_embeds_t)
        frechet = calculate_frechet_distance(mu_s, sigma_s, mu_t, sigma_t)
    return frechet


def corpus_similarity(args, models, model_name):
    doc_embeds_s, _ = read_doc_enc_from_pickle(models.source_datasets[model_name], model_name,
                                               args.embedding_dir)
    doc_embeds_t, _ = read_doc_enc_from_pickle(args.dataset_name, model_name, args.embedding_dir)
    frechet = _batched_frechet(doc_embeds_s, doc_embeds_t, subsample_size=args.subsample_size)
    return frechet


def extracted_corpus_similarity(args, models, model_name, queries):
    query_list = [queries[qid] for qid in queries]
    model = models.load_model(model_name, model_name_or_path=None)

    search_result = load_search_results(args.search_results_dir, args.dataset_name, model_name)

    query_embeds = model.encode_queries(query_list, batch_size=32, show_progress_bar=True,
                                        convert_to_tensor=False)
    embeddings_target, docids_target = read_doc_enc_from_pickle(args.dataset_name, model_name,
                                                                args.embedding_dir)
    # speed up the search of the embedding index by doc name/id
    # docids_target_dict[ doc_name ] == embedding_indx
    docids_target_dict = {}
    for i, el in enumerate(docids_target):
        docids_target_dict[el] = i

    # Source embeddings
    embeddings_source, docids_source = read_doc_enc_from_pickle(models.source_datasets[model_name], model_name,
                                                                args.embedding_dir)

    # Search for the target query within the SOURCE corpus
    scores_source, indices_source = search(query_embeddings=query_embeds,
                                           doc_embeddings=embeddings_source,
                                           score_function=models.score_function[model_name],
                                           top_k=100)
    search_result_source = embeddings_source[indices_source]
    frechet_result = {}
    import time
    t1 = time.time()
    for i, qid in enumerate(queries):
        # Documents extracted from the source dataset (Usually, it is msmarco)
        docs_per_query_source = search_result_source[i]
        # assert len(docs_per_query_source) == 1000
        docs_per_query_source = docs_per_query_source[:100]
        assert len(docs_per_query_source) == 100

        # For the target data the search has been done already
        docs2q_id_target, doc2q_scores_target = search_result[qid].keys(), search_result[qid].values()

        # id <- get the id of docs2q_id_target in docids_target
        target_indx = [docids_target_dict[el] for el in docs2q_id_target]

        # Documents extracted from the target dataset
        docs_per_query_target = embeddings_target[target_indx]
        docs_per_query_target = docs_per_query_target[:100]
        assert len(docs_per_query_target) == 100

        frechet_result[qid] = _batched_frechet(docs_per_query_source, docs_per_query_target,
                                               subsample_size=args.subsample_size)
        if i % 100:
            print(i, frechet_result[qid], "out of", len(queries))
    return frechet_result


def binary_entropy(args, models, model_name, queries, corpus):
    """
    Evaluate the model based on the entropy of its predictions
    Read the search id of the document for the query
    Perform MinMax normalization of the scores to bring them in the range [0, 1]
    Calculate the score between the query and the document
    Get the negative entropy from the score.
    The smaller is the value - the more confident is the model.
    """
    model = models.load_model(model_name, model_name_or_path=None)
    score_function = models.score_function[model_name]
    entropy_per_query_10, entropy_per_query_1000 = {}, {}

    # log_dir, dataset_name, user_id="", model_name=None, special_id=None
    search_results = load_search_results(log_dir=os.path.join(args.log_dir, "search_results"),
                                         dataset_name=args.dataset_name,
                                         model_name=model_name)
    # Set of irrelevant doc ids
    irrelevant_doc_ids = _mine_for_irrelevant_docs(search_results, corpus)
    del corpus
    gc.collect()

    embeddings, docids = read_doc_enc_from_pickle(args.dataset_name, model_name, args.embedding_dir,
                                                  user_id=args.user_id)

    # speed up the search of the embedding index by doc name/id
    # docids_target_dict[ doc_name ] == embedding_indx
    docids_target_dict = {}
    for i, el in enumerate(docids):
        docids_target_dict[el] = i

    assert len(docids_target_dict) == len(embeddings)

    stats = {}

    mask_irrelevant = [docids_target_dict[i] for i in irrelevant_doc_ids]
    emb_irrelevant = embeddings[mask_irrelevant]

    query_list = [queries[qid] for qid in queries]

    if "instructor" in model_name:
        query_embeds = model.encode_queries(query_list, batch_size=32,
                                            convert_to_tensor=False,
                                            dataset_name=args.dataset_name)
    else:
        query_embeds = model.encode_queries(query_list, batch_size=32,
                                            convert_to_tensor=False)

    irrelevant_scores = get_score(query_embeds, emb_irrelevant,
                                  score_function=score_function)
    approximated_min = min(irrelevant_scores.flatten())
    all_max_scores = [list(val.values())[0] for val in search_results.values()]
    approximated_max = max(all_max_scores)

    # Iterate through the target queries
    for query_ids in search_results.keys():

        ranked_dict = search_results[query_ids]

        scores = torch.tensor(list(ranked_dict.values()))

        # Min Max normalization for scores
        scores = (scores - approximated_min) / (approximated_max - approximated_min)

        stats[query_ids] = {}
        for k in [3, 5, 10, 50, 100, 1000]:
            stats[query_ids]["mean@" + str(k)] = torch.mean(scores[:k]).cpu().item()
            stats[query_ids]["std@" + str(k)] = torch.std(scores[:k]).cpu().item()

        entropy_per_doc = get_binary_entropy(scores)
        # The entropy of the first value might be nan due to the log,
        #       as after minmax the first doc has the score=1
        entropy_per_doc = torch.nan_to_num(entropy_per_doc)

        assert len(entropy_per_doc) == 1000

        # entropy_value = sum(entropy_per_doc).item()

        entropy_per_query_10[query_ids] = sum(entropy_per_doc[:10])
        entropy_per_query_1000[query_ids] = sum(entropy_per_doc[:1000])

    # with open(f"{args.log_dir}/model_selection/binary_entropy/{args.dataset_name}/{args.dataset_name}_{model_name}.txt", 'w') as f:
    #     for qid in entropy_per_query_10.keys():
    #         f.write(f"{qid}\t{entropy_per_query_10[qid]}\n")

    return entropy_per_query_10, entropy_per_query_1000


def _mine_for_irrelevant_docs(search_results, corpus):
    """
    Mines for up to 1000 irrelevant documents

    :param search_results:
    :param corpus: Target Corpus
    :return: id of 1000 or less irrelevant documents
    """
    # Slice define top-k extracted documents
    for slice in [1000, 800, 500, 300, 200, 100, 50, 30, 20, 10]:
        relevant_docs = set()
        for query_ids in search_results.keys():
            ranked_dict = list(search_results[query_ids].keys())[:slice]
            relevant_docs.update(ranked_dict)
        irrelevant_docs = set(corpus.keys()) - relevant_docs
        if len(irrelevant_docs) != 0:
            break
    irrelevant_docs = list(irrelevant_docs)
    random.shuffle(irrelevant_docs)
    # Only return 1000 relevant documents for approximation
    return irrelevant_docs[:1000]


def query_alteration(args, models, model_name, queries, cuda=True):
    entropy_result, std_result = {}, {}
    alteration_number = 100

    softmax = torch.nn.Softmax(dim=1)
    model = models.load_model(model_name, model_name_or_path=None)
    embeddings, docids = read_doc_enc_from_pickle(args.dataset_name, model_name, args.embedding_dir,
                                                  user_id=args.user_id)
    docids_target_dict = {}
    for i, el in enumerate(docids):
        docids_target_dict[el] = i

    search_results = load_search_results(log_dir=os.path.join(args.log_dir, "search_results"),
                                         dataset_name=args.dataset_name,
                                         model_name=model_name)
    print(f"Model {model_name} is loaded")
    query_ids = search_results.keys()
    for query_id in query_ids:
        # Load query, modify it and get its encoding
        query_text = queries[query_id]
        altered_queries = []
        altered_queries.append(query_text)
        for _ in range(alteration_number):
            modified_query = _adding_mask_tokens(model.tokenizer, query_text, mask_ratio=args.mask_ratio)
            altered_queries.append(modified_query)

        if "instructor" in model_name:
            altered_query_encodings = model.encode_queries(altered_queries, batch_size=32,
                                                           dataset_name=args.dataset_name)
        else:
            altered_query_encodings = model.encode_queries(altered_queries, batch_size=32)
        if cuda:
            altered_query_encodings = torch.tensor(altered_query_encodings).cuda()

        # Load document encodings for this unaltered query
        extracted_docs_dict = search_results[query_id]
        doc_ids, doc_scores = list(extracted_docs_dict.keys())[:args.topk], \
            list(extracted_docs_dict.values())[:args.topk]

        assert extracted_docs_dict[doc_ids[1]] == doc_scores[1]
        mask = [docids_target_dict[i] for i in doc_ids]
        doc_per_query_embedding = embeddings[mask]
        if cuda:
            doc_per_query_embedding = torch.tensor(doc_per_query_embedding).cuda()

        # Get the score for altered documents
        # For each document - 100 different scores
        scores = get_score(doc_per_query_embedding, altered_query_encodings,
                           score_function=models.score_function[model_name])

        # Check that the pre-computed score is the same as our score
        if not cuda:
            # For debugging purpose only
            assert torch.all(torch.isclose(scores.T[0], torch.tensor(doc_scores[:args.topk])))

        scores = softmax(scores)

        per_doc_entropy = entropy(scores)
        # For each query, the entropy is the avrg negative entropy across its documents
        per_query_entropy = torch.mean(per_doc_entropy)
        entropy_result[query_id] = per_query_entropy.cpu().item()

        # Also, record the standard deviation
        per_doc_std = torch.std(scores, dim=1)
        per_query_std = torch.mean(per_doc_std)
        std_result[query_id] = per_query_std.cpu().item()

    folder = f"{args.log_dir}/model_selection/query_alteration/"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/{args.dataset_name}_{model_name}.txt", 'w') as f:
        for qid in std_result.keys():
            f.write(f"{qid}\t{std_result[qid]}\n")
    return entropy_result, std_result


def _adding_mask_tokens(tokenizer: AutoTokenizer, text: str, mask_ratio: float):
    mask_token = tokenizer.mask_token
    tokens = tokenizer.tokenize(text)
    mask_num = int(len(tokens) * mask_ratio)
    mask_index = random.sample(range(len(tokens)), mask_num)
    for i in mask_index:
        if mask_token is None:
            tokens[i] = f"<extra_id_{i}>"
        else:
            tokens[i] = mask_token

    return tokenizer.convert_tokens_to_string(tokens)


def gold_fusion(args, model_name):
    if args.task == "fusion":
        predicted_rank = load_search_results(os.path.join(args.log_dir, "search_results"),
                                             args.dataset_name, model_name=model_name)
        gold_rank = load_search_results(os.path.join(args.log_dir, f"search_results_fusion"),
                                        args.dataset_name, special_id="fusion_run")
    elif "fake_fusion" in args.task:
        predicted_rank = load_search_results(os.path.join(args.log_dir, "search_results_fake"),
                                             args.dataset_name,
                                             special_id=f"sch_{args.dataset_name}_{model_name}_{args.fake_id_qrels}")
        gold_rank = load_search_results(os.path.join(args.log_dir, f"search_results_{args.task}"),
                                        args.dataset_name,
                                        special_id=f"{args.dataset_name}-{args.fake_id_queries}")
    else:
        raise ValueError("Task not supported")

    rbos = []
    for q in gold_rank.keys():
        list_gold = list(gold_rank[q])
        list_predicted = list(predicted_rank[q])
        rbo_per_query = rbo.RankingSimilarity(list_gold, list_predicted).rbo()
        rbos.append(rbo_per_query)
    return rbos


def pseudo_eval(args, models, model_name,
                pseudo_qid, pseudo_query, pseudo_qrels, ):
    model = models.load_model(model_name, model_name_or_path=None)
    doc_embeds, doc_ids = read_doc_enc_from_pickle(args.dataset_name, model_name,
                                                   args.embedding_dir, user_id=args.user_id)

    query_embeds = model.encode_queries(pseudo_query,
                                        batch_size=args.batch_size,
                                        show_progress_bar=True,
                                        convert_to_tensor=False)
    scores, indices = search(query_embeds, doc_embeds, top_k=1000,
                             score_function=models.score_function[model_name])

    # Save search resuls fake
    folder = os.path.join(args.log_dir, "search_results_fake")
    os.makedirs(folder, exist_ok=True)
    file_name = "sch_{}_{}_{}.txt".format(args.dataset_name, model_name, args.fake_id_qrels)
    save_search_results(pseudo_qid, doc_ids, scores, indices,
                        where_to_save=os.path.join(folder, file_name),
                        run_id=f"sch_{args.dataset_name}_{model_name}_{args.fake_id_qrels}")
    # Pseudo Evaluation
    res = transform_search_results(pseudo_qid, doc_ids, scores, indices)
    eval_retrieval = EvaluateRetrieval()
    # Can add 1000 here if needed
    eval_results = eval_retrieval.evaluate(qrels=pseudo_qrels,
                                           results=res,
                                           k_values=[1, 3, 5, 10, 100, 1000])

    # Final score is NDCG@10 on pseudo-labels
    ndcg10 = eval_results[0]['NDCG@10']
    return ndcg10


def qpps(args, model_name, queries, name="sigma_max"):
    search_results = load_search_results(log_dir=os.path.join(args.log_dir, "search_results"),
                                         dataset_name=args.dataset_name,
                                         model_name=model_name)

    scores = []
    for q_indx, query_id in enumerate(queries.keys()):
        score_list = [score for (pid, score) in sorted(search_results[query_id].items(),
                                                       key=lambda x: x[1], reverse=True)]
        if name == "sigma_max":
            score, _ = SIGMA_MAX(score_list)
        elif name == "nqc":
            # only take the normalized version
            _, score = NQC(score_list)
        elif name == "smv":
            _, score = SMV(score_list)
        else:
            raise ValueError("QPP method not supported")
        scores.append(score)
    score = np.mean(scores)
    return score


def fake_qrels(args, model_name):
    eval_log_dir = os.path.join(args.log_dir, "eval_results_fake")
    eval_name = "eval_{}_{}_{}.txt".format(args.dataset_name, model_name, args.fake_id_qrels)
    with open(os.path.join(eval_log_dir, eval_name), "r") as f:
        eval_results = json.load(f)
        ndsg10 = eval_results[0]['NDCG@10']

    return ndsg10


def model_ranking_fusion(args):
    rankings = []

    qrel_ranking_file = os.path.join(args.log_dir, 'model_selection', 'fake_qrels', f'score_fake_qrels_{args.dataset_name}_{args.fake_id_qrels}.txt')
    rankings.append(Run.from_file(qrel_ranking_file, kind="trec"))

    ref_ranking_file = os.path.join(args.log_dir, 'model_selection', 'fake_fusion', f'score_fake_fusion_{args.dataset_name}_{args.fake_id_queries}.txt')
    rankings.append(Run.from_file(ref_ranking_file, kind="trec"))

    fusion_ranking = fuse(
        runs=rankings,
        method="rrf",
    )
    return fusion_ranking['0']


def model_selection():
    args = get_args()
    print(args.log_dir)
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, args.user_id, args.dataset_name)
    print("Log dir:", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    # Define models
    # if args.model_type == "beir":
    #     models = BeirModels(args.model_dir)
    # else:
    #     models = CustomModel(model_dir=args.model_dir)
    # model_names = models.names
    # Load dataset
    datasets = Datasets(data_dir=args.dataset_dir)
    corpus, queries, qrels = datasets.load_dataset(args.dataset_name, load_corpus=True, user_id=args.user_id)
    print("Dataset is loaded")

    if args.specific_model:
        MODELS = [CustomModel(args.model_dir, specific_model=args.specific_model)]
        if args.specific_model not in MODELS[0].score_function.keys():
            MODELS = [BeirModels(model_dir=args.model_dir, specific_model=args.specific_model)]
    else:
        # run on all models
        MODELS = [BeirModels(args.model_dir),
                  CustomModel(model_dir=args.model_dir)]
    # MODELS = [BeirModels(args.model_dir, specific_model="gte-tiny")]

    if args.task == "fusion":
        fused_run = fuse_runs(os.path.join(args.log_dir, f"search_results"))
        folder = os.path.join(args.log_dir, "search_results_fusion")
        # check if the fusion run exists
        fusion_file = os.path.join(folder, "fusion_run.txt")
        if not os.path.exists(fusion_file):
            os.makedirs(folder, exist_ok=True)
            fused_run.save(os.path.join(folder, "fusion_run.txt"), kind='trec')

    score_dict_fused = None
    if args.task == "model_ranking_fusion":
        score_dict_fused = model_ranking_fusion(args)

    score_dict = {}
    ndsg10 = {}
    reverse_score = False
    for models in MODELS:
        model_names = models.names
        for model_name in model_names:
            if args.task == "binary_entropy":
                scores, entropy_1000 = binary_entropy(args, models, model_name, queries, corpus)
                score = np.mean(list(scores.values()))

            elif args.task == "query_alteration":
                if "SGPT" in model_name:
                    score = 1
                else:
                    scores, std_res = query_alteration(args, models, model_name, queries,
                                                       cuda=torch.cuda.is_available())
                    score = np.mean(list(std_res.values()))

            elif args.task == "model_ranking_fusion":
                score = score_dict_fused[model_name]
                reverse_score = True

            elif "fusion" in args.task:
                scores = gold_fusion(args, model_name)
                score = np.mean(scores)
                reverse_score = True

            elif args.task == "qpp_sigma_max":
                score = qpps(args, model_name, queries, name="sigma_max")
            elif args.task == "qpp_nqc":
                score = qpps(args, model_name, queries, name="nqc")
            elif args.task == "qpp_smv":
                score = qpps(args, model_name, queries, name="smv")
            elif args.task == "fake_qrels":
                score = fake_qrels(args, model_name)
                reverse_score = True
            else:
                raise ValueError("Task {} not supported".format(args.task))

            score_dict[model_name] = score
            if len(args.user_id) == 0:
                # Load ground truth NDCG@10
                eval_log_dir = os.path.join(args.log_dir, "eval_results")
                eval_name = "eval_{}_{}.txt".format(args.dataset_name, model_name)
                with open(os.path.join(eval_log_dir, eval_name), "r") as f:
                    eval_results = json.load(f)
                    ndsg10[model_name] = eval_results[0]['NDCG@10']
                print("Model:", model_name, "Task:", args.task,
                      "Score:", score_dict[model_name], "NDSG10:", ndsg10[model_name])

    if args.task == "model_ranking_fusion":
        id_temp = 'larmor'
    elif args.task == 'fake_qrels':
        id_temp = args.fake_id_qrels
    else:
        id_temp = args.fake_id_queries

    log_dir = os.path.join(args.log_dir, "model_selection", args.job_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, args.task), exist_ok=True)

    score_file = "score_{}_{}_{}.txt".format(args.task, args.dataset_name, id_temp)
    score_sorted = sorted(score_dict.items(), key=lambda x: x[1], reverse=reverse_score)

    with open(os.path.join(log_dir, args.task, score_file), "w") as f:
        # f.write("Rank\tModel\tScore\n")
        for rank_i, (model_name, score) in enumerate(score_sorted):
            f.write(f"{0} Q0 {model_name} {rank_i} {score} {args.task}\n")

    # If this is not a demo - calculate evaluation, namely tau correlation and
    if len(args.user_id) == 0:
        # with open(os.path.join(log_dir, args.task, score_file), "w") as f:
        #     json.dump(score_dict, f, default=_convert_to_serializable)

        # Compute Tau correlation
        tau, p_value = stats.kendalltau(list(score_dict.values()),
                                        list(ndsg10.values()))
        print("TAU on NDCG@10:", tau)
        # Save Tau values in file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        tau_file = "tau_{}_{}_{}.txt".format(args.task, args.dataset_name, id_temp)
        with open(os.path.join(log_dir, args.task, tau_file), "w") as f:
            json.dump(tau, f)

        max_ndcg10 = np.max(list(ndsg10.values()))
        predicted_best_model = max(score_dict, key=score_dict.get)
        delta_ndcg10 = (max_ndcg10 - ndsg10[predicted_best_model]) * 100
        print('Delta on NDCG@10:', delta_ndcg10)
        ndcg10_file = "ndcg10_{}_{}_{}.txt".format(args.task, args.dataset_name, id_temp)
        with open(os.path.join(log_dir, args.task, ndcg10_file), "w") as f:
            json.dump(delta_ndcg10, f)

    print("Finished")


def _convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError("Type not serializable")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_selection()
