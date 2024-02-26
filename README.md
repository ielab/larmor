# Larmor

Larmor is a framework for evaluating and ranking dense retrieval systems (DRs) using generated queries and judgments.

## To encode the documents for all BEIR datasets with each Dense Retriever, run:

```bash
for dataset in trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do
    CUDA_VISIBLE_DEVICES=0  python3 encoding_and_eval.py \
        --dataset_name $dset \
        --dataset_dir /PATH_TO_BEIR_DATASET/   \
        --model_dir /PATH_TO_MODELS/ \
        --log_dir  /PATH_TO_STORE_LOGS/ \
        --embedding_dir /PATH_TO_STORE_ENCODINGS/ \
        --task encode \
done
```

If you do not want to run the evaluation on all Retrievers, you can specify the DR you want to run the evaluation on by adding the `--specific_model MODEL_NAME` argument. <br>
Change task to `eval` to run evaluation on the encoded documents. This step is only applicable if you have ground truth judgments for the dataset. 

## To generate queries and judgments for these queries for each BEIR datasets, run:

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


## To perform model selection, run:

```bash
for dset in trec-covid nfcorpus fiqa scidocs scifact quora nq hotpotqa dbpedia-entity robust04 trec-news signal1m arguana; do
		CUDA_VISIBLE_DEVICE=0 python model_selection_methods.py \
		--dataset_name $dset \
		--dataset_dir /PATH_TO_BEIR_DATASET/    \
		--model_dir /PATH_TO_MODELS/  \
		--log_dir  /PATH_TO_STORE_LOGS/ \
		--embedding_dir /PATH_TO_STORE_ENCODINGS/ \
		--task binary_entropy
		#--task fake_fusion \
		#--fake_id_queries flan-t5-large-q10.fusion.setwise \
		#--fake_id_qrels flan-t5-large-q10
done
```

Supported tasks are: 
- `binary_entropy`: Model selection using binary entropy
- `query_alteration`: Model selection using query alteration
- QPP-based methods: `qpp_sigma_max`, `qpp_nqc`, `qpp_smv`.
- `fusion`: Fuse the run files from all the DRs and evaluate the fused ranking
- `fake_fusion`: Fuse the run files from all the DRs and evaluate the fused ranking using fake queries and judgments. This requires the `--fake_id_queries` and `--fake_id_qrels` arguments to be specified. 

