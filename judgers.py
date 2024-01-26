import random
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import json
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from prompts import PROMPT_QG_FLAN_T5, PROMPT_QG_OpenAI, PROMPT_POINTWISE_FLAN_T5, PROMPT_POINTWISE_OPENAI
from openai import OpenAI
import openai
import time
random.seed(929)
set_seed(929)


class Text2TextGenerationDataset(Dataset):
    def __init__(self, data: List[str], tokenizer: PreTrainedTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        inputs = self.tokenizer(self.data[item],
                                return_tensors='pt',
                                truncation=True,
                                padding='longest',
                                max_length=512)
        return {'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0]}


class Judger:
    def judge(self, inputs: List[Tuple]) -> Dict[str, Dict[str, int]]:
        raise NotImplementedError

    def _set_dataloader(self, inputs: List[Tuple]):
        raise NotImplementedError


class FlanT5QgJudger(Judger):
    def __init__(self, model, tokenizer, dataset_name, num_gen_qry_per_doc=1, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.num_gen_qry_per_doc = num_gen_qry_per_doc
        self.batch_size = batch_size
        self.qids = []
        self.queries = []
        self.docids = []

    def _set_dataloader(self, inputs: List[Tuple]):
        prompts = []
        for docid, title, text in inputs:
            self.docids.extend([docid] * self.num_gen_qry_per_doc)
            prompts.append(PROMPT_QG_FLAN_T5[self.dataset_name]['user'].format(title=title, text=text))

        dataset = Text2TextGenerationDataset(prompts, self.tokenizer)
        self.loader = DataLoader(
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
        self.decoder_input_ids = self.tokenizer.encode("<pad> " + PROMPT_QG_FLAN_T5[self.dataset_name]['assistant'],
                                                       return_tensors="pt", truncation=True, max_length=512,
                                                       add_special_tokens=False).to("cuda").repeat(self.batch_size, 1)

    def judge(self, inputs: List[Tuple]) -> Dict[str, Dict[str, int]]:
        self._set_dataloader(inputs)
        output_ids = []
        with torch.no_grad():
            for batch_inputs in tqdm(self.loader, desc="Generating queries"):
                batch_inputs.to("cuda")
                batch_outputs = self.model.generate(**batch_inputs,
                                                    decoder_input_ids=self.decoder_input_ids
                                                    if self.decoder_input_ids.shape[0] == len(batch_inputs['input_ids'])
                                                    else self.decoder_input_ids[:len(batch_inputs['input_ids']), :],
                                                    # last batch might be smaller
                                                    do_sample=True,
                                                    max_new_tokens=128,
                                                    num_return_sequences=self.num_gen_qry_per_doc,
                                                    top_p=0.9)
                output_ids.extend(batch_outputs[:, self.decoder_input_ids.shape[1]:].cpu().numpy())

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        qrels = {}
        for qid, (docid, text) in enumerate(zip(self.docids, outputs)):
            qid = str(qid)
            self.qids.append(qid)
            self.queries.append(text.strip())
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = 1

        return qrels


class OpenaiQgJudger(Judger):
    def __init__(self, model, key, dataset_name, num_gen_qry_per_doc=1):
        self.model = model
        self.dataset_name = dataset_name
        self.num_gen_qry_per_doc = num_gen_qry_per_doc
        self.qids = []
        self.docids = []
        self.queries = []

        self.client = OpenAI(
          api_key=key,
        )

    def judge(self, inputs: List[Tuple]) -> Dict[str, Dict[str, int]]:
        for docid, title, text in tqdm(inputs, desc="Generating queries"):
            while True:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        response_format={"type": "json_object"},
                        n=self.num_gen_qry_per_doc,
                        top_p=0.9,
                        messages=[
                            {"role": "system", "content": PROMPT_QG_OpenAI[self.dataset_name]['system']},
                            {"role": "user", "content": PROMPT_QG_OpenAI[self.dataset_name]['user'].format(title=title, text=text)}
                        ]
                    )
                    for choice in completion.choices:
                        self.queries.append(json.loads(choice.message.content)[PROMPT_QG_OpenAI[self.dataset_name]['key']])
                        self.docids.append(docid)
                    break

                except openai.APIError as e:
                    # Handle API error here, e.g. retry or log
                    print(f"OpenAI API returned an API Error: {e}")
                    time.sleep(5)
                    continue

                except openai.APIConnectionError as e:
                    # Handle connection error here
                    print(f"Failed to connect to OpenAI API: {e}")
                    time.sleep(5)
                    continue

                except openai.RateLimitError as e:
                    # Handle rate limit error (we recommend using exponential backoff)
                    print(f"OpenAI API request exceeded rate limit: {e}")
                    time.sleep(5)
                    continue

                except openai.BadRequestError as e:
                    # Handle invalid request error
                    print(f"OpenAI API request was invalid: {e}")
                    raise e

                except openai.AuthenticationError as e:
                    # Handle authentication error
                    print(f"OpenAI API request failed authentication: {e}")
                    raise e

                except openai.Timeout as e:
                    # Handle timeout error
                    print(f"OpenAI API request timed out: {e}")
                    time.sleep(5)
                    continue

                except openai.InternalServerError as e:
                    # Handle service unavailable error
                    print(f"OpenAI API request failed with a service unavailable error: {e}")
                    time.sleep(5)
                    continue

                except Exception as e:
                    print(f"Unknown error: {e}")
                    raise e

        qrels = {}
        for qid, (docid, text) in enumerate(zip(self.docids, self.queries)):
            qid = str(qid)
            self.qids.append(qid)
            self.queries.append(text.strip())
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = 1

        return qrels


class FlanT5PointwiseJudger(Judger):
    def __init__(self, model, tokenizer, dataset_name, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.qids = []
        self.docids = []

    def _set_dataloader(self, inputs: List[Tuple]):
        prompts = []
        for qid, docid, query, title, text in inputs:
            self.qids.append(qid)
            self.docids.append(docid)
            prompts.append(PROMPT_POINTWISE_FLAN_T5[self.dataset_name]['user'].format(query=query,
                                                                                      title=title,
                                                                                      text=text))
        dataset = Text2TextGenerationDataset(prompts, self.tokenizer)
        self.loader = DataLoader(
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
        self.decoder_input_ids = self.tokenizer.encode("<pad> " + PROMPT_POINTWISE_FLAN_T5[self.dataset_name]['assistant'],
                                                       return_tensors="pt", truncation=True, max_length=512,
                                                       add_special_tokens=False).to("cuda").repeat(self.batch_size, 1)

    def judge(self, inputs: List[Tuple]) -> Dict[str, Dict[str, int]]:
        self._set_dataloader(inputs)
        output_ids = []
        with torch.no_grad():
            for batch_inputs in tqdm(self.loader, desc="Generating relevance labels"):
                batch_inputs.to("cuda")
                batch_outputs = self.model.generate(**batch_inputs,
                                                    decoder_input_ids=self.decoder_input_ids
                                                    if self.decoder_input_ids.shape[0] == len(batch_inputs['input_ids'])
                                                    else self.decoder_input_ids[:len(batch_inputs['input_ids']), :],
                                                    max_new_tokens=4)
                output_ids.extend(batch_outputs[:, self.decoder_input_ids.shape[1]:].cpu().numpy())

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        qrels = {}
        for qid, docid, output in zip(self.qids, self.docids, outputs):
            if qid not in qrels:
                qrels[qid] = {}
            if output.lower() == PROMPT_POINTWISE_FLAN_T5[self.dataset_name]['key'].lower():
                qrels[qid][docid] = 1

        return qrels


class OpenAiPointwiseJudger(Judger):
    def __init__(self, model, key, dataset_name):
        self.model = model
        self.dataset_name = dataset_name
        self.qids = []
        self.docids = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.client = OpenAI(
          api_key=key,
        )

    def judge(self, inputs: List[Tuple]) -> Dict[str, Dict[str, int]]:
        outputs = []
        for qid, docid, query, title, text in tqdm(inputs, desc="Generating relevance labels"):
            while True:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": PROMPT_POINTWISE_OPENAI[self.dataset_name]['system']},
                            {"role": "user", "content": PROMPT_POINTWISE_OPENAI[self.dataset_name]['user'].format(query=query,
                                                                                                                  title=title,
                                                                                                                  text=text)}
                        ]
                    )
                    outputs.append(list(json.loads(completion.choices[0].message.content).values())[0])
                    self.docids.append(docid)
                    self.qids.append(qid)

                    self.total_completion_tokens += int(completion.usage.completion_tokens)
                    self.total_prompt_tokens += int(completion.usage.prompt_tokens)

                    break

                except openai.APIError as e:
                    # Handle API error here, e.g. retry or log
                    print(f"OpenAI API returned an API Error: {e}")
                    time.sleep(5)
                    continue
                except openai.APIConnectionError as e:
                    # Handle connection error here
                    print(f"Failed to connect to OpenAI API: {e}")
                    time.sleep(5)
                    continue
                except openai.RateLimitError as e:
                    # Handle rate limit error (we recommend using exponential backoff)
                    print(f"OpenAI API request exceeded rate limit: {e}")
                    time.sleep(5)
                    continue
                except openai.BadRequestError as e:
                    # Handle invalid request error
                    print(f"OpenAI API request was invalid: {e}")
                    raise e
                except openai.AuthenticationError as e:
                    # Handle authentication error
                    print(f"OpenAI API request failed authentication: {e}")
                    raise e
                except openai.Timeout as e:
                    # Handle timeout error
                    print(f"OpenAI API request timed out: {e}")
                    time.sleep(5)
                    continue
                except openai.InternalServerError as e:
                    # Handle service unavailable error
                    print(f"OpenAI API request failed with a service unavailable error: {e}")
                    time.sleep(5)
                    continue
                except Exception as e:
                    print(f"Unknown error: {e}")
                    raise e

        print("Total prompt tokens:", self.total_prompt_tokens)
        print("Total completion tokens:", self.total_completion_tokens)

        qrels = {}
        for qid, docid, output in zip(self.qids, self.docids, outputs):
            if qid not in qrels:
                qrels[qid] = {}
            if output.lower() == PROMPT_POINTWISE_OPENAI[self.dataset_name]['key'].lower():
                qrels[qid][docid] = 1

        return qrels
