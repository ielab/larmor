

from pyserini.index import IndexReader
from pyserini.search.lucene import LuceneSearcher

import numpy as np
from data.dataset_collection import Datasets
import pytrec_eval
from utils.read_and_write import load_search_results, load_eval_results

from model.model_zoo import CustomModel, BeirModels
from scipy.stats import   kendalltau
from utils.get_args import get_args
import os
import time


def RM1(pid_list, score_list, index_reader, k, mu=1000):
    # V = []
    V = set()
    doc_len = np.zeros(k)

    doc_vectors, empty_docs = {}, []
    for idx_p, pid in enumerate(pid_list[:k]):
        try:
            doc_vector = index_reader.get_document_vector(pid)
            if doc_vector is None:
                empty_docs.append(pid)
                continue
            else:
                doc_vectors[idx_p] = doc_vector
                V.update(doc_vector.keys())
                doc_len[idx_p] = sum(doc_vector.values())
        except:
            empty_docs.append(pid)
            continue

    V = list(V)
    mat = np.zeros([k, len(V)])
    # add very small number to avoid division by zero
    doc_len += 1e-10
    for idx_p, pid in enumerate(pid_list[:k]):
        if pid in empty_docs:
            continue
        # doc_vector = index_reader.get_document_vector(pid)
        doc_vector = doc_vectors[idx_p]
        for token in doc_vector.keys():
            mat[idx_p, V.index(token)] = doc_vector[token]

    _p_w_q = np.dot(np.array([score_list[:k] / doc_len , ]), mat) # [1, V] become a probability distribution
    p_w_q = np.asarray(_p_w_q/ sum(score_list[:k])).squeeze() # normalisation [V]
    rm1 = np.sort(np.array(list(zip(V, p_w_q)), dtype=[('tokens', object), ('token_scores', np.float32)]),
                  order='token_scores')[::-1] # [V]
    return rm1


def CLARITY(rm1, index_reader, term_num=100):

    rm1_cut = rm1[:term_num] # [term num]
    # make sure it is a probability distribution after sampling
    p_w_q = rm1_cut['token_scores'] / rm1_cut['token_scores'].sum()
    p_t_D = (np.array([[index_reader.get_term_counts(token, analyzer=None)[1] for token in rm1_cut['tokens']], ])
             / index_reader.stats()['total_terms']) # [1, term num]

    return np.log(p_w_q / p_t_D).dot(p_w_q)[0]


def WIG(qtokens, score_list, k):
    corpus_score = np.mean(score_list)
    wig_norm = (np.mean(score_list[:k]) - corpus_score)/ np.sqrt(len(qtokens))
    wig_no_norm = np.mean(score_list[:k]) / np.sqrt(len(qtokens))

    return wig_norm, wig_no_norm


def NQC(score_list, k=100):
    corpus_score = np.mean(score_list)
    nqc_norm = np.std(score_list[:k]) / corpus_score
    nqc_no_norm = np.std(score_list[:k])

    return nqc_norm, nqc_no_norm


def SIGMA_MAX(score_list):
    max_std=0
    scores=[]

    for idx, score in enumerate(score_list):
        scores.append(score)
        if np.std(scores)>max_std:
            max_std = np.std(scores)

    return max_std, len(scores)


def SIGMA_X(qtokens, score_list, x):

    top_score = score_list[0]
    scores = []

    for idx, score in enumerate(score_list):
        if score>=(top_score*x):
            scores.append(score)

    return np.std(scores)/np.sqrt(len(qtokens)), len(scores)


def SMV(score_list, k=100):
    corpus_score = np.mean(score_list)
    mu = np.mean(score_list[:k])
    smv_norm = np.mean(np.array(score_list[:k])*abs(np.log(score_list[:k]/mu)))/corpus_score
    smv_no_norm = np.mean(np.array(score_list[:k])*abs(np.log(score_list[:k]/mu)))

    return smv_norm, smv_no_norm


def get_model_names(model_dir, old_models_only, special_token):
    MODELS = [BeirModels(model_dir, old_models_only=old_models_only),
              CustomModel(model_dir=model_dir,
                          special_token=special_token,
                          old_models_only=old_models_only
                          )]
    model_names = [model.names for model in MODELS]
    model_names = [item for sublist in model_names for item in sublist]
    return model_names


def qpp(args):
    # Hyperparameters for QPP
    term_num = 100
    k_clarity = 100
    k_wig = 5
    k_smv = 100
    k_nqc = 100
    x = 0.5

    datasets = Datasets(data_dir=args.dataset_dir)
    _, queries, qrels_gt = datasets.load_dataset(args.dataset_name, load_corpus=False)

    # given a folder, find a subfolder with the model name
    # Get a list of all subfolders in the specified folder
    pyserini_path = os.path.join(args.dataset_dir, "pyserini/indexes/")
    subfolders = [f for f in os.listdir(pyserini_path) if os.path.isdir(os.path.join(pyserini_path, f))]
    correct_subfolder = [f for f in subfolders if args.dataset_name in f]

    if len(correct_subfolder) == 0:
        # Load the index reader from the prebuilt index and save it to the default folder
        index_reader = IndexReader.from_prebuilt_index(f"beir-v1.0.0-{args.dataset_name}.flat")
    elif len(correct_subfolder) == 1:
        index_reader = IndexReader(os.path.join(pyserini_path, correct_subfolder[0]))
    else:
        raise ValueError("More than one subfolder with the dataset name")

    # searcher = LuceneSearcher.from_prebuilt_index(f"beir-v1.0.0-{args.dataset_name}.flat")
    model_names = get_model_names(args.model_dir, args.old_models_only, args.special_token)

    for m_indx, model_name in enumerate(model_names):
        print(f"Model {m_indx+1}/{len(model_names)}: {model_name}")
        search_results = load_search_results(f"{args.log_dir}/search_results/",
                                             args.dataset_name,
                                             model_name)

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_gt, {"ndcg_cut.10"})
        scores_gt = evaluator.evaluate(search_results)

        score_dict = {}
        for score_names in ["clarity", "wig", "wig_norm",  "nqc", "nqc_norm",
                            "smv", "smv_norm", "sigma_x", "sigma_max"]:
            score_dict[score_names] = {}

        t0 = time.time()
        for q_indx, query_id in enumerate(queries.keys()):
            t1 = time.time()
            qtokens = index_reader.analyze(queries[query_id])
            pid_list = [pid for (pid, score) in sorted(search_results[query_id].items(),
                                                       key=lambda x: x[1], reverse=True)]
            score_list = [score for (pid, score) in sorted(search_results[query_id].items(),
                                                           key=lambda x: x[1], reverse=True)]
            rm1 = RM1(pid_list, score_list, index_reader, k_clarity, mu=1000)
            score_dict["clarity"][query_id] = CLARITY(rm1, index_reader, term_num=term_num)

            score_dict["wig"][query_id], score_dict["wig_norm"][query_id] = WIG(qtokens, score_list, k_wig)
            score_dict["nqc"][query_id], score_dict["nqc_norm"][query_id] = NQC(score_list, k_nqc)
            score_dict["smv"][query_id], score_dict["smv_norm"][query_id] = SMV(score_list, k_smv)
            score_dict["sigma_x"][query_id], _ = SIGMA_X(qtokens, score_list, x)
            score_dict["sigma_max"][query_id], _ = SIGMA_MAX(score_list)

        t2 = time.time()
        # estimated time till the end
        print(f"Estimated time till the end: {(t2-t0)*(len(model_names)-m_indx)/60} minutes")

        # save results to txt file
        for score_name in score_dict.keys():

            # if the folder doesn't exist - create it
            if not os.path.exists(f"{args.log_dir}/qpp_results/{score_name}"):
                os.makedirs(f"{args.log_dir}/qpp_results/{score_name}")

            with open(f"{args.log_dir}/qpp_results/{score_name}/{args.dataset_name}_{model_name}.txt", 'w') as f:
                for qid in score_dict[score_name].keys():
                    f.write(f"{qid}\t{score_dict[score_name][qid]}\n")


def qpp_mselect_eval(args):

    taus, deltas = {}, {}
    score_names = ["clarity", "wig", "wig_norm",  "nqc", "nqc_norm",
                      "smv", "smv_norm", "sigma_x", "sigma_max"]
    for score_name in score_names:
        taus[score_name] = []
        deltas[score_name] = []
    for d_indx, dset_name in enumerate(["nfcorpus", "fiqa", "arguana", "scidocs", "scifact", "trec-covid",
                      "quora", "nq", "dbpedia-entity" , "hotpotqa",
                      "signal1m", "robust04", "trec-news",
                      ]):
        args.dataset_name = dset_name
        model_names = get_model_names(args.model_dir, args.old_models_only, args.special_token)
        eval_results = []
        score_dict = {}
        for score_name in score_names:
            score_dict[score_name] = []

        for model_name in model_names:
            eval_result = load_eval_results(args.log_dir,
                                            args.dataset_name,
                                            model_name)
            eval_results.append(eval_result)

            for score_name in score_names:
                scores = []
                with open(f"{args.log_dir}/qpp_results/{score_name}/{args.dataset_name}_{model_name}.txt", 'r') as f:
                    for line in f:
                        qid, score = line.split()
                        # if score is inf - continue
                        if score == "inf":
                            continue
                        # score_dict[score_names][qid] = float(score)
                        scores.append(float(score))

                score_dict[score_name].append(np.max(scores))

        for score_name in score_names:
            tau, p_value = kendalltau(eval_results, score_dict[score_name])
            taus[score_name].append(tau)

            best_indx = np.argmax(score_dict[score_name])
            delta = max(eval_results) - eval_results[best_indx]
            deltas[score_name].append(delta)

    for score_name in score_names:
        print(score_name)
        print(' & '.join(map(lambda x: str(round(x, 3)), taus[score_name])))
        print(f"Average Tau across dsets {round(np.mean(taus[score_name]), 3)}")

        # print(' & '.join(map(lambda x: str(round(x*100, 2)), deltas[score_name])))
        # print(f"Average Delta across dsets {round(np.mean(deltas[score_name])*100, 2)}")


if __name__ == "__main__":
    args = get_args()
    if args.task == "qpp_eval":
        qpp_mselect_eval(args)
    else:
        qpp(args)



