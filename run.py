import ir_datasets
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, T5ForConditionalGeneration
import torch
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import json
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from prompts import PROMPT_LLAMA, PROMPT_T5, PROMPT_T5_JUDGE, ICL_PROMPT_LLAMA, ICL_PROMPT_T5
random.seed(929)
set_seed(929)


class Text2TextGenerationDataset(Dataset):
    def __init__(self, data: List[str], tokenizer: PreTrainedTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        inputs = self.tokenizer(self.data[item], return_tensors='pt', truncation=True, max_length=512)
        return {'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0]}


class T5Judger:
    def __init__(self, model, tokenizer, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def judge(self, inputs: List[str], ir_dataset=None) -> List[str]:
        dataset = Text2TextGenerationDataset(inputs, self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(
                self.tokenizer,
                max_length=512,
                padding='longest',
            ),
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        decoder_input_ids = self.tokenizer.encode("<pad> " + PROMPT_T5_JUDGE[ir_dataset][2],
                                                  return_tensors="pt", truncation=True, max_length=512,
                                                  add_special_tokens=False).to("cuda").repeat(self.batch_size, 1)

        outputs = []
        with torch.no_grad():
            for batch_inputs in tqdm(loader):
                batch_outputs = self.model.generate(batch_inputs['input_ids'].to("cuda"),
                                                    decoder_input_ids=decoder_input_ids
                                                    if decoder_input_ids.shape[0] == len(batch_inputs['input_ids'])
                                                    else decoder_input_ids[:len(batch_inputs['input_ids']), :], # last batch might be smaller
                                                    max_new_tokens=4)
                outputs.extend(batch_outputs[:, decoder_input_ids.shape[1]:].cpu().numpy())

        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs


def main(args):
    dataset = ir_datasets.load(args.ir_dataset)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if 'llama' in args.model_name:
        tokenizer.use_default_system_prompt = False
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        PROMPT = PROMPT_LLAMA
        ICL_PROMPT = ICL_PROMPT_LLAMA

        judge_model = None
        decoder_input_ids = None
    elif 't5' in args.model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        PROMPT = PROMPT_T5
        ICL_PROMPT = ICL_PROMPT_T5
        decoder_input_ids = tokenizer.encode("<pad> " + PROMPT[args.ir_dataset][2],
                                             return_tensors="pt", truncation=True, max_length=512,
                                             add_special_tokens=False).to("cuda")
        judge_model = T5Judger(model, tokenizer, batch_size=args.batch_size)
    else:
        raise NotImplementedError

    model = model.to("cuda")
    model.eval()

    icl_queries = [query.text for query in dataset.queries_iter()]
    random.shuffle(icl_queries)
    icl_queries = icl_queries[:args.icl_num]

    qid = 0
    queries = []
    qids = []
    qrels = {}
    for docid, title, text in tqdm(sample_docs):
        user_prompt = PROMPT[args.ir_dataset][1].format(title=title, text=text)
        if args.icl:
            system_prompt = ICL_PROMPT[args.ir_dataset][0].format(queries="\n".join(icl_queries))
        else:
            system_prompt = PROMPT[args.ir_dataset][0]

        if 'llama' in args.model_name:
            conversation = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
            prompt += " " + PROMPT[args.ir_dataset][2]
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512,
                                         add_special_tokens=False).to("cuda")

            with torch.no_grad():
                output_ids = model.generate(input_ids,
                                            do_sample=True,
                                            max_new_tokens=128,
                                            num_return_sequences=args.num_gen_qry_per_doc,
                                            top_p=0.9)
            output_texts = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

        elif 't5' in args.model_name:
            prompt = system_prompt + user_prompt
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to("cuda")

            output_ids = model.generate(input_ids,
                                        do_sample=True,
                                        max_new_tokens=128,
                                        num_return_sequences=args.num_gen_qry_per_doc,
                                        top_p=0.9,
                                        decoder_input_ids=decoder_input_ids,
                                        )
            output_texts = tokenizer.batch_decode(output_ids[:, decoder_input_ids.shape[1]:], skip_special_tokens=True)

        else:
            raise NotImplementedError

        for output_text in output_texts:
            qids.append(str(qid))
            queries.append(output_text.strip())
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = 1
            qid += 1
    with open(args.save_path + ".queries.tsv", "w") as f:
        for qid, query in zip(qids, queries):
            f.write(f"{qid}\t{query}\n")

    with open(args.save_path + ".qrels", "w") as f:
        for qid in qrels:
            for doc in qrels[qid]:
                f.write(f"{qid} 0 {doc} {qrels[qid][doc]}\n")

    if args.bm25_judgements:
        searcher = LuceneSearcher.from_prebuilt_index(args.pyserini_index)
        results = searcher.batch_search(
            queries, qids, k=args.judge_depth, threads=12
        )
        results = [(id_, results[id_]) for id_ in qids]
        inputs = []
        qid_docid = []
        for qid, result in results:
            query = queries[int(qid)]
            for doc in result:
                data = json.loads(doc.raw)
                qid_docid.append((qid, data['_id']))
                prompt = (PROMPT_T5_JUDGE[args.ir_dataset][0] +
                          PROMPT_T5_JUDGE[args.ir_dataset][1].format(query=query, title=data['title'], text=data['text']))
                inputs.append(prompt)

        outputs = judge_model.judge(inputs, args.ir_dataset)

        for (qid, docid), output in zip(qid_docid, outputs):
            if ('highly relevant' in output.lower()
                    or 'counter argument' in output.lower()
                    or output.lower() == 'supports'
                    or output.lower() == 'same'):
                qrels[int(qid)][docid] = 1

        with open(args.save_path + f".bm25-top{args.judge_depth}" + ".qrels", "w") as f:
            for qid in qrels:
                for doc in qrels[qid]:
                    f.write(f"{qid} 0 {doc} {qrels[qid][doc]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_dataset", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_sample_docs", type=int, default=100)
    parser.add_argument("--num_gen_qry_per_doc", type=int, default=1)
    parser.add_argument("--bm25_judgements", action="store_true")
    parser.add_argument("--pyserini_index", type=str)
    parser.add_argument("--judge_depth", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--icl", action="store_true")
    parser.add_argument("--icl_num", type=int, default=3)
    args = parser.parse_args()

    main(args)

''' bash
for dataset in trec-covid nfcorpus fiqa arguana scidocs scifact quora nq hotpotqa dbpedia-entity; do

for dataset in scifact quora; do
    mkdir -p qrels/${dataset}
    CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --ir_dataset beir/${dataset} \
    --model_name google/flan-t5-xl \
    --save_path qrels/${dataset}/${dataset}-flan-t5-xl-q3 \
    --num_gen_qry_per_doc 3 \
    --bm25_judgements \
    --judge_depth 100 \
    --batch_size 4 \
    --pyserini_index beir-v1.0.0-${dataset}.flat &

    CUDA_VISIBLE_DEVICES=1 python3 run.py \
    --ir_dataset beir/${dataset} \
    --model_name google/flan-t5-xl \
    --save_path qrels/${dataset}/${dataset}-icl-flan-t5-xl-q3 \
    --num_gen_qry_per_doc 3 \
    --icl \
    --bm25_judgements \
    --judge_depth 100 \
    --batch_size 4 \
    --pyserini_index beir-v1.0.0-${dataset}.flat
    wait
done
'''
