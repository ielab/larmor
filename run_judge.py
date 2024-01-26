import os.path

import ir_datasets
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import json
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from pyserini.search._base import get_topics
from dataclasses import dataclass
from judgers import FlanT5PointwiseJudger, OpenAiPointwiseJudger
from pyserini.search.lucene import LuceneSearcher
import glob
from ranx import Qrels, Run
from ranx import fuse
random.seed(929)
set_seed(929)

VALID_MODELS = ['gte-small', 'msmarco-distilbert-base-tas-b', 'bge-small-en-v1', 'msmarco-distilbert-base-v3', 'jina-embeddings-v2-small-en', 'e5-base', 'SGPT-2', 'gtr-t5-large', 'instructor-base', 'bge-base-en-v1', 'sentence-t5-large', 'gte-large', 'all-MiniLM-L12-v2', 'gtr-t5-base', 'instructor-xl', 'msmarco-MiniLM-L-12-v3', 'e5-base-v2', 'co-condenser-marco-retriever', 'gtr-t5-xl', 'contriever', 'msmarco-MiniLM-L-6-v3', 'e5-large-v2', 'bge-large-en-v1', 'multilingual-e5-base', 'e5', 'UAE-Large-V1', 'SGPT-1', 'ember-v1', 'all-mpnet-base-v2', 'jina-embeddings-v2-base-en', 'e5-small-v2', 'e5-large', 'gte-base', 'all-MiniLM-L6-v2', 'sentence-t5-xl', 'stella-base-en-v2', 'msmarco-distilbert-base-dot-prod-v3', 'SGPT-5', 'msmarco-distilbert-base-v2', 'SGPT-125M-weightedmean-msmarco-specb-bitfit', 'gte-tiny', 'multilingual-e5-small', 'simlm-base-msmarco-finetuned', 'msmarco-roberta-base-ance-firstp', 'e5-small', 'instructor-large', 'multilingual-e5-large']

@dataclass
class SearchDoc:
    raw: str
    score: float


def main(args):
    if 'flan-t5' in args.model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float16).eval().to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        judge_model = FlanT5PointwiseJudger(model,
                                            tokenizer,
                                            args.ir_dataset,
                                            batch_size=args.batch_size)
    elif 'gpt' in args.model_name and args.openai_key is not None:
        judge_model = OpenAiPointwiseJudger(args.model_name,
                                            args.openai_key,
                                            args.ir_dataset)
    else:
        raise NotImplementedError

    queries = []
    qids = []
    if args.query_file is None:
        topics = get_topics(args.pyserini_index[:-5] + '-test')
        for topic_id in list(topics.keys()):
            qids.append(str(topic_id))
            text = topics[topic_id]['title']
            queries.append(text)
    else:
        with open(args.query_file) as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) == 1:
                    qid, query = items[0], ""
                else:
                    qid, query = items
                queries.append(query)
                qids.append(qid)

    if args.qrel_file is not None:
        qrels = {}
        with open(args.qrel_file) as f:
            for line in f:
                qid, _, docid, rel = line.strip().split()
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = int(rel)
    else:
        qrels = None

    searcher = LuceneSearcher.from_prebuilt_index(args.pyserini_index)

    if args.run_files is not None:
        method = 'fusion'
        run_files = glob.glob(args.run_files)
        print(run_files)
        runs = []
        for run_file in tqdm(run_files, desc='loading run files'):
            # runs.append(Run.from_file(run_file, kind="trec"))

            if run_file.split('_')[-1].split('.')[0] in VALID_MODELS:
                print(run_file.split('_')[-1].split('.')[0])
                runs.append(Run.from_file(run_file, kind="trec"))
        print(len(runs))

        assert len(runs) == 47  # temp check

        print('fusing runs')
        fusion_run = fuse(
            runs=runs,
            method="rrf",
        )

        if args.fusion_save_path is not None:
            fusion_run.save(args.fusion_save_path, kind='trec')

        fusion_run_dict = fusion_run.to_dict()
        results = {}
        for qid in tqdm(fusion_run_dict, desc='getting top docs'):
            ranking = list(fusion_run_dict[qid].items())[:args.judge_depth]
            results[qid] = []
            for docid, score in ranking:
                try:
                    results[qid].append(SearchDoc(raw=searcher.doc(docid).raw(), score=score))
                except AttributeError:
                    print(f'cannot find docid {docid}')
                    continue

    else:  # bm25
        method = 'bm25'
        results = searcher.batch_search(
            queries, qids, k=args.judge_depth, threads=12
        )

    qid_docid = []
    inputs = []
    for qid, query in zip(qids, queries):
        result = results[qid]
        for doc in result:
            data = json.loads(doc.raw)
            qid_docid.append((qid, data['_id']))
            inputs.append((qid, data['_id'], query, data['title'], data['text']))

    new_qrels = judge_model.judge(inputs)

    if qrels is None:
        qrels = new_qrels
    else:
        for qid in new_qrels:
            for docid in new_qrels[qid]:
                if docid not in qrels[qid]:
                    qrels[qid][docid] = new_qrels[qid][docid]

    with open(args.save_path + f".{method}-top{args.judge_depth}" + ".qrels", "w") as f:
        for qid in qrels:
            for doc in qrels[qid]:
                f.write(f"{qid} 0 {doc} {qrels[qid][doc]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=str, default=None)
    parser.add_argument("--qrel_file", type=str, default=None)
    parser.add_argument("--ir_dataset", type=str)
    parser.add_argument("--pyserini_index", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--run_files", type=str, default=None)
    parser.add_argument("--fusion_save_path", type=str, default=None)
    parser.add_argument("--judge_depth", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--openai_key", type=str, default=None)
    args = parser.parse_args()

    main(args)

