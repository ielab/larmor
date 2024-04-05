
from typing import List, Dict, cast, Mapping
from model.model_collection import CustomDEModel
from model.instructor.instructor_definitions import DEFINITIONS_INSTRUCTOR
from model.instructor.instructor_model import INSTRUCTOR
from functools import partial
from typing import List, Dict, Union, Tuple

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer

from beir.retrieval.models import SentenceBERT

import torch
import numpy as np
from tqdm import tqdm


class SimLMDEModel(CustomDEModel):
    def __init__(self, model_name_or_path, cache_dir, cuda):
        super().__init__()
        # Re-init this in the constructor
        self.query_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                       cache_dir=cache_dir).eval()
        if cuda:
            self.query_encoder = self.query_encoder.cuda()

        self.doc_encoder = self.query_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir)
        self.config = None
        self.score_function = "dot"
        self.cuda = cuda

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries, batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                inputs = self.tokenizer(batch,
                                        max_length=32,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')

                if self.cuda:
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                outputs = self.query_encoder(**inputs, return_dict=True)
                query_embedding = self._l2_normalize(outputs.last_hidden_state[:, 0, :])
                embeddings.append(query_embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def tokenize_query(self, queries, **kwargs):
        inputs = self.tokenizer(queries,
                                max_length=32,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                # return_tensors='pt'
                                )["input_ids"]
        return inputs

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(corpus), batch_size):
                titles = [doc['title'] if 'title' in doc.keys() else '-' for doc in corpus[i:i + batch_size]]
                texts = [doc['text'] for doc in corpus[i:i + batch_size]]
                inputs = self.tokenizer(titles,
                                        text_pair=texts,
                                        max_length=144,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
                if self.cuda:
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                outputs = self.doc_encoder(**inputs, return_dict=True)
                doc_embedding = self._l2_normalize(outputs.last_hidden_state[:, 0, :])
                embeddings.append(doc_embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def tokenize_corpus(self, corpus, **kwargs):
        titles = [doc['title'] if 'title' in doc.keys() else '-' for doc in corpus]
        texts = [doc['text'] for doc in corpus]
        inputs = self.tokenizer(titles,
                                text_pair=texts,
                                max_length=350,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                # return_tensors='pt'
                                )["input_ids"]
        return inputs

    def _l2_normalize(self, x: torch.Tensor):
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class CoCondenserModel(CustomDEModel):
    def __init__(self, name, cache_dir, cuda):
        super().__init__()
        # Re-init this in the constructor

        self.query_encoder = AutoModel.from_pretrained(name,
                                                       cache_dir=cache_dir).eval()
        if cuda:
            self.query_encoder = self.query_encoder.cuda()

        self.doc_encoder = self.query_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(name,
                                                       cache_dir=cache_dir)
        self.config = None
        self.score_function = "dot"
        self.cuda = cuda

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries, batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                inputs = self.tokenizer(batch,
                                        max_length=32,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')

                if self.cuda:
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                outputs = self.query_encoder(**inputs, return_dict=True)
                query_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(query_embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def tokenize_query(self, queries, **kwargs):
        inputs = self.tokenizer(queries,
                                max_length=32,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                #return_tensors='pt',
                                )['input_ids']
        return inputs

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus, batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(corpus), batch_size):
                titles = [doc['title'] if 'title' in doc.keys() else '-' for doc in corpus[i:i + batch_size]]
                texts = [doc['text'] for doc in corpus[i:i + batch_size]]
                input_texts = [f'{title}{self.tokenizer.sep_token}{text}' for title, text in zip(titles, texts)]
                inputs = self.tokenizer(input_texts,
                                        max_length=144,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
                if self.cuda:
                    inputs = {key: value.cuda() for key, value in inputs.items()}
                outputs = self.doc_encoder(**inputs, return_dict=True)
                doc_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(doc_embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def tokenize_corpus(self, corpus):
        titles = [doc['title'] if 'title' in doc.keys() else '-' for doc in corpus]
        texts = [doc['text'] for doc in corpus]
        input_texts = [f'{title}{self.tokenizer.sep_token}{text}' for title, text in zip(titles, texts)]
        inputs = self.tokenizer(input_texts,
                                max_length=350,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                # return_tensors='pt'
                                )['input_ids']
        return inputs


class AnglEModel(CustomDEModel):
    def __init__(self, model_name_or_path, cache_dir, cuda, add_query_instructions=True):
        super().__init__()
        if add_query_instructions:
            self.query_instruction_for_retrieval = 'Represent this sentence for searching relevant passages:'
        else:
            self.query_instruction_for_retrieval = None
        self.normalize_embeddings = True
        self.pooling_method = 'cls'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               cache_dir=cache_dir)

        if cuda:
            self.query_encoder = self.model.cuda()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cuda = cuda
        self.score_function = "cos_sim"

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        if self.query_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus

        return self.encode(input_texts, batch_size)

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int, **kwargs) -> np.ndarray:
        self.model.eval()

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d


class E5Model(CustomDEModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, model_name_or_path, cache_dir, cuda, **kwargs):
        self.encoder = AutoModel.from_pretrained(model_name_or_path,
                                                 cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir)
        # self.gpu_count = torch.cuda.device_count()
        # if self.gpu_count > 1:
        #     self.encoder = torch.nn.DataParallel(self.encoder)
        self.cuda = cuda
        if cuda:
            self.encoder.cuda()

        self.encoder.eval()
        self.pooling_method = "mean"
        self.score_function = "cos_sim"

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = ['query: {}'.format(q) for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        input_texts = ['passage: {}'.format(t) for t in input_texts]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        dataset: Dataset = Dataset.from_dict({'contents': input_texts})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=128,   #  * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        # for batch_dict in tqdm(data_loader, desc='encoding', mininterval=10):
        for batch_dict in data_loader:
            if self.cuda:
                batch_dict = move_to_cuda(batch_dict)

            # with torch.cuda.amp.autocast():
            outputs = self.encoder(**batch_dict)
            embeds = self._pooling(outputs.last_hidden_state, batch_dict['attention_mask'])
            encoded_embeds.append(embeds.cpu().numpy())
        return np.concatenate(encoded_embeds, axis=0)

    def _pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.pooling_method == "mean":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling_method == "cls":
            emb = last_hidden[:, 0]
        else:
            raise ValueError(f"pool_type {self.pooling_method} not supported")
        return emb


def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    return tokenizer(examples['contents'],
                     max_length=512,
                     padding=True,
                     return_token_type_ids=False,
                     truncation=True)


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}
    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor
    return _move_to_cuda(sample)


class SentenceTransformerSpec(CustomDEModel):
    def __init__(self, model_name_or_path, cache_dir, cuda, speca=False, specb=True, **kwargs):
        self.encoder = SentenceTransformer(model_name_or_path, cache_folder=cache_dir)
        word_embedding_model = self.encoder._first_module()
        self.speca = speca
        self.specb = specb
        self.sep = " "
        if self.specb:
            tokens = ["[SOS]", "{SOS}"]
            word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

            # Will be replaced with the rep ones
            word_embedding_model.bos_spec_token_q = \
            word_embedding_model.tokenizer.encode("[SOS]", add_special_tokens=False)[0]
            word_embedding_model.bos_spec_token_d = \
            word_embedding_model.tokenizer.encode("{SOS}", add_special_tokens=False)[0]

            word_embedding_model.bos_spec_token_q_rep = \
            word_embedding_model.tokenizer.encode("[", add_special_tokens=False)[0]
            word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode("]", add_special_tokens=False)[0]

            word_embedding_model.bos_spec_token_d_rep = \
            word_embedding_model.tokenizer.encode("{", add_special_tokens=False)[0]
            word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode("}", add_special_tokens=False)[0]

            word_embedding_model.replace_bos = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir)
        self.cuda = cuda
        if cuda:
            self.encoder.cuda()
        self.encoder.eval()
        self.score_function = "cos_sim"

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        if self.speca or self.specb:
            # Will be replaced with [ in the models tokenization
            # If we would put [ here, there is a risk of it getting chained with a different token when encoding
            queries = ["[SOS]" + q for q in queries]
        return self.encoder.encode(queries, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        sentences = [("{SOS}" + doc["title"] + self.sep + doc["text"]).strip()
                         if "title" in doc else "{SOS}" + doc["text"].strip() for doc in corpus]
        return self.encoder.encode(sentences, **kwargs)


class InstructorModel(CustomDEModel):
    def __init__(self, model_name_or_path, cache_dir, cuda, **kwargs):

        self.model_name_or_path = model_name_or_path
        self.encoder = INSTRUCTOR(model_name_or_path, cache_folder=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir)
        self.cuda = cuda
        if cuda:
            self.encoder.cuda()
        self.encoder.eval()
        self.score_function = "cos_sim"

    def encode_queries(self, queries: List[str], dataset_name: str, **kwargs) -> np.ndarray:
        new_sentences = []
        # if isinstance(DEFINITIONS_INSTRUCTOR[self.model_name_or_path][dataset_name], str):
        #     instruction = DEFINITIONS_INSTRUCTOR[self.model_name_or_path][dataset_name]
        # else:
        instruction = DEFINITIONS_INSTRUCTOR[self.model_name_or_path][dataset_name]['query']
        for s in queries:
            new_sentences.append([instruction, s, 0])
        return self.encoder.encode(new_sentences, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], dataset_name: str, **kwargs) -> np.ndarray:
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + ' ' + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + ' ' + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        new_sentences = []
        instruction = DEFINITIONS_INSTRUCTOR[self.model_name_or_path][dataset_name]['corpus']
        for s in sentences:
            new_sentences.append([instruction, s, 0])
        # kwargs['show_progress_bar'] = False
        kwargs['batch_size'] = 128
        return self.encoder.encode(sentences, **kwargs)


class JinaModel(SentenceBERT):
    def __init__(self, model_path):
        self.score_function = "cos_sim"
        self.sep = " "
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.q_model = model
        self.doc_model = model
