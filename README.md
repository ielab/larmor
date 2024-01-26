# larmor

## To generate queries and judgments for generated queries for each BEIR datasets, run:

```bash
NUM_Q=10
MODEL=flan-t5-large  # or flan-t5-xl, flan-t5-xxl, 
for dataset in trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do
    mkdir -p qrels/${dataset}
    CUDA_VISIBLE_DEVICES=0 python3 run_qg.py \
    --ir_dataset beir/${dataset} \
    --model_name google/${MODEL} \
    --save_path qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q} \
    --num_gen_qry_per_doc ${NUM_Q} \
    --batch_size 16
done

```
Then run all DRs for generated queries and save run files in `search_results`


## To fuse all the runs and generate judgments on the fused ranking, run:
```bash
for dataset in trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do
mkdir -p fusion/${dataset}

python3 run_judge.py \
  --run_files search_results/${dataset}/'*'.txt \
  --fusion_save_path fusion/${dataset}/${dataset}-${MODEL}-q${NUM_Q}.fusion.txt \
  --ir_dataset beir/${dataset} \
  --pyserini_index beir-v1.0.0-${dataset}.flat \
  --judge_depth 100 \
  --model_name google/${MODEL} \
  --save_path qrels/${dataset}/${dataset}-${MODEL}-q${NUM_Q} \
  --batch_size 8

done
```
Then you can evaluate and rank all the DRs with generated qrel files.

## To re-rank all the fused ranking to generate reference lists, run:
```bash
for dataset in trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do

python run_rerank.py \
  run --model_name_or_path google/${MODEL} \
      --tokenizer_name_or_path google/${MODEL} \
      --run_path fusion/${dataset}/${dataset}-${MODEL}-q${NUM_Q}.fusion.txt \
      --save_path fusion/${dataset}/${dataset}-${MODEL}-q${NUM_Q}.fusion.setwise.txt \
      --pyserini_index beir-v1.0.0-${dataset} \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --scoring generation \
      --device cuda \
  setwise --num_child 2 \
          --method heapsort \
          --k 10
done
```
Then you can evaluate and rank all the DRs with generated reference lists (Re-ranked fusion run files).
