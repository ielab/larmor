import ir_datasets
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, T5ForConditionalGeneration
from judgers import FlanT5QgJudger, OpenaiQgJudger
import torch
random.seed(929)
set_seed(929)
import os
import json


class Document:
    def __init__(self, doc_id, title, text):
        self.doc_id = doc_id
        self.title = title
        self.text = text


class CustomDataset:
    def __init__(self, path_to_dir):
        with open(os.path.join(path_to_dir, "corpus.jsonl")) as f:
            self.docs = f.readlines()

    def docs_iter(self):
        for line in self.docs:
            data = json.loads(line)
            yield Document(data['_id'], data['title'], data['text'])


def main(args):
    if 'flan-t5' in args.model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float16).eval().to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        judge_model = FlanT5QgJudger(model,
                                     tokenizer,
                                     args.ir_dataset,
                                     num_gen_qry_per_doc=args.num_gen_qry_per_doc,
                                     batch_size=args.batch_size)
    elif 'gpt' in args.model_name and args.openai_key is not None:
        judge_model = OpenaiQgJudger(args.model_name,
                                     args.openai_key,
                                     args.ir_dataset,
                                     num_gen_qry_per_doc=args.num_gen_qry_per_doc)
    else:
        raise NotImplementedError
    try:
        dataset = ir_datasets.load(args.ir_dataset)
    except:
        dataset = CustomDataset(args.ir_dataset)

    num_docs = len(dataset.docs)
    sample_doc_ids = set(random.sample(range(num_docs), args.num_sample_docs))
    sample_docs = []
    count = 0
    for doc in dataset.docs_iter():
        if count in sample_doc_ids:
            docid = doc.doc_id
            try:
                title = doc.title.strip()
                if args.dataset == 'beir/scidocs': # we dont include title for this dataset because the task is tilte generation
                    title = ""
            except:
                title = ""
            text = doc.text.strip()
            sample_docs.append((docid, title, text))
        count += 1
    qrels = judge_model.judge(sample_docs)

    with open(args.save_path + ".queries.tsv", "w") as f:
        for qid, query in zip(judge_model.qids, judge_model.queries):
            f.write(f"{qid}\t{query}\n")

    with open(args.save_path + ".qrels", "w") as f:
        for qid in qrels:
            for doc in qrels[qid]:
                f.write(f"{qid} 0 {doc} {qrels[qid][doc]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_dataset", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_sample_docs", type=int, default=100)
    parser.add_argument("--num_gen_qry_per_doc", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--openai_key", type=str, default=None)
    args = parser.parse_args()

    main(args)

''' bash
for dataset in trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do

export TRANSFORMERS_CACHE=/mnt/storage/arvin/cache/transformers
export IR_DATASETS_HOME=/mnt/storage/arvin/cache/ir_datasets
export PYSERINI_CACHE=/mnt/storage/arvin/cache/pyserini


# for dataset in trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do

NUM_Q=10
MODEL=flan-t5-large
for dataset in trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do
    mkdir -p qrels_test/${dataset}
    CUDA_VISIBLE_DEVICES=0 python3 run_qg.py \
    --ir_dataset beir/${dataset} \
    --model_name google/${MODEL} \
    --save_path qrels_test/${dataset}/${dataset}-${MODEL}-q${NUM_Q} \
    --num_gen_qry_per_doc ${NUM_Q} \
    --batch_size 16
done

for dataset in scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do
    mkdir -p qrels/${dataset}
    CUDA_VISIBLE_DEVICES=1 python3 run_judge_bm25.py \
    --pyserini_index beir-v1.0.0-${dataset}.flat \
    --ir_dataset beir/${dataset} \
    --judge_depth 100 \
    --query_file qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q}.queries.tsv \
    --qrel_file qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q}.qrels \
    --model_name google/${MODEL} \
    --save_path qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q} \
    --batch_size 16
done

for dataset in dbpedia-entity robust04 trec-news signal1m arguana; do
    mkdir -p qrels/${dataset}
    CUDA_VISIBLE_DEVICES=1 python3 run_judge_bm25.py \
    --pyserini_index beir-v1.0.0-${dataset}.flat \
    --ir_dataset beir/${dataset} \
    --judge_depth 100 \
    --query_file qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q}.queries.tsv \
    --qrel_file qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q}.qrels \
    --model_name google/${MODEL} \
    --save_path qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q} \
    --batch_size 8
done


# gpt-4-1106-preview gpt-3.5-turbo-1106
NUM_Q=10
MODEL=gpt-3.5-turbo-1106
for dataset in dbpedia-entity; do
    mkdir -p qrels/${dataset}
    CUDA_VISIBLE_DEVICES=1 python3 run_qg.py \
    --ir_dataset beir/${dataset} \
    --model_name ${MODEL} \
    --save_path qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q} \
    --num_gen_qry_per_doc ${NUM_Q} \
    --openai_key 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
done
'''
