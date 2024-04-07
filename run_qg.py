import random
import argparse
from transformers import AutoTokenizer, set_seed, T5ForConditionalGeneration
from judgers import FlanT5QgJudger, OpenaiQgJudger
import torch
import os
import json
random.seed(929)
set_seed(929)


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
                                     args.dataset_name,
                                     num_gen_qry_per_doc=args.num_gen_qry_per_doc,
                                     batch_size=args.batch_size)
    elif 'gpt' in args.model_name and args.openai_key is not None:
        judge_model = OpenaiQgJudger(args.model_name,
                                     args.openai_key,
                                     args.dataset_name,
                                     num_gen_qry_per_doc=args.num_gen_qry_per_doc)
    else:
        raise NotImplementedError

    dataset = CustomDataset(os.path.join(args.dataset_dir, args.dataset_name))

    num_docs = len(dataset.docs)
    sample_doc_ids = set(random.sample(range(num_docs), args.num_sample_docs))
    sample_docs = []
    count = 0
    for doc in dataset.docs_iter():
        if count in sample_doc_ids:
            docid = doc.doc_id
            try:
                title = doc.title.strip()
                if args.dataset_name == 'scidocs':  # we dont include title for this dataset because the task is tilte generation
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
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_sample_docs", type=int, default=100)
    parser.add_argument("--num_gen_qry_per_doc", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--openai_key", type=str, default=None)
    args = parser.parse_args()

    main(args)
