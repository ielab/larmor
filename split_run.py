import argparse
import os
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--query_file", type=str)
parser.add_argument("--run_files", type=str)
parser.add_argument("--num_q", type=str)
args = parser.parse_args()


queries = {}
with open(args.query_file) as f:
    lines = f.readlines()
    for line in lines:
        items = line.strip().split('\t')
        if len(items) == 1:
            qid, query = items[0], ""
        else:
            qid, query = items
        queries[qid] = query

run_files = glob.glob(args.run_files)

# read run file
for file in tqdm(run_files):
    splits = file.split(".")
    splits[-2] = f'{splits[-2]}-q{args.num_q}'
    save_path = ".".join(splits)
    run = {}
    with open(file, 'r') as f:
        with open(save_path, 'w') as wf:
            lines = f.readlines()
            for line in lines:
                qid, _, docid, rank, score, run_name = line.strip().split()
                if qid in queries.keys():
                    wf.write(f'{qid} Q0 {docid} {rank} {score} {run_name}\n')



'''
for dataset in robust04 trec-news signal1m trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity arguana; do
for dataset in robust04 trec-news signal1m trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity arguana; do
    for id in flan-t5-xxl-q10; do
        for q in 1 3 5; do
        echo ${dataset}-${id}
        python3 split_run.py \
            --query_file /scratch/project/neural_ir/arvin/llm-qrel/qrels/${dataset}/${dataset}-${id}-q${q}.queries.tsv \
            --run_file /scratch/project/neural_ir/katya/code/beir_eval/logs/res_t5xxl/search_results_fake/${dataset}/sch_${dataset}_'*'${id}.txt \
            --num_q ${q}
    done
done
done
'''
