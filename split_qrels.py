import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file_id", type=str)
parser.add_argument("--dataset_dir", type=str)
args = parser.parse_args()


queries = {}
with open(f'{args.dataset_dir}/{args.file_id}.queries.tsv') as f:
    lines = f.readlines()
    for line in lines:
        items = line.strip().split('\t')
        if len(items) == 1:
            qid, query = items[0], ""
        else:
            qid, query = items
        queries[qid] = query


qrels = {}

with open(f'{args.dataset_dir}/{args.file_id}.qrels') as f:
    lines = f.readlines()
    for line in lines:
        qid, _, docid, rel = line.strip().split()
        if docid not in qrels:
            qrels[docid] = {}
        qrels[docid][qid] = int(rel)

for num_q in [1, 3, 5]:
    new_queries = []
    new_qrels = {}
    for docid in qrels:
        qids = list(qrels[docid].keys())
        if len(qids) < num_q:
            print(f'cannot generate for num_q {num_q}, not enough queries')
            continue
        new_qids = qids[:num_q]
        for qid in new_qids:
            if qid not in new_qrels:
                new_qrels[qid] = {}
            new_qrels[qid][docid] = qrels[docid][qid]
            new_queries.append((qid, queries[qid]))

        with open(f'{args.dataset_dir}/{args.file_id}-q{num_q}.qrels', 'w') as f:
            for qid in new_qrels:
                for docid in new_qrels[qid]:
                    f.write(f'{qid} 0 {docid} {new_qrels[qid][docid]}\n')

        with open(f'{args.dataset_dir}/{args.file_id}-q{num_q}.queries.tsv', 'w') as f:
            for qid, query in new_queries:
                f.write(f'{qid}\t{query}\n')

        # check file exists
        if os.path.exists(f'{args.dataset_dir}/{args.file_id}.bm25-top100.qrels'):
            bm25_qrels = {}
            with open(f'{args.dataset_dir}/{args.file_id}.bm25-top100.qrels', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    qid, _, docid, rel = line.strip().split()
                    if qid not in bm25_qrels:
                        bm25_qrels[qid] = {}
                    bm25_qrels[qid][docid] = int(rel)
            with open(f'{args.dataset_dir}/{args.file_id}-q{num_q}.bm25-top100.qrels', 'w') as f:
                for qid in new_qrels:
                    for docid in bm25_qrels[qid]:
                        f.write(f'{qid} 0 {docid} {bm25_qrels[qid][docid]}\n')


'''
for dataset in robust04 trec-news signal1m trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity arguana; do
for dataset in robust04 trec-news signal1m trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity arguana; do
    for id in flan-t5-large-q10 flan-t5-xxl-q10; do
        echo ${dataset}-${id}
        python3 split_qrels.py \
            --file_id ${dataset}-${id} \
            --dataset_dir qrels/${dataset}
    done
done
'''
