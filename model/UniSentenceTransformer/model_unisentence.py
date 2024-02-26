

from typing import Callable, Iterable, List, Dict, Optional, Type, Union, Tuple
import json
import logging
import math
import os
import queue

import numpy as np
from tqdm.autonotebook import trange
import torch
from torch import nn, distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device, fullname
from sentence_transformers.model_card_templates import ModelCardTemplate
logger = logging.getLogger(__name__)


class UniSentenceTransformer(SentenceTransformer):
    def __init__(
        self,
        *args,
        max_seq_length: int = 300,
        default_query: bool = False,
        setting: str = 'spec',
        sep: str = " ",
        accelerator = None,
        encode_batch_size: int = 0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_seq_length = max_seq_length
        self.default_query = default_query  # SGPT is True ?
        self.sep = sep
        self.encode_batch_size = encode_batch_size
        self.accelerator = accelerator

        tokenizer: PreTrainedTokenizerFast = self._first_module().tokenizer
        if tokenizer.padding_side != 'right':
            # weighted mean pooling need pad right
            logger.warn(f"Change tokenizer.padding_side from {tokenizer.padding_side} to right")
            tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None:
            logger.warn(f"Set tokenizer.pad_token as eos_token {tokenizer.eos_token}")
            tokenizer.pad_token = tokenizer.eos_token

        self.setting = setting
        if not setting:
            return  # vanilla
        elif setting == 'sgpt':
            self.added_tokens = dict(boq='[SOS]', bod='{SOS}')
            # Will be replaced with the rep tokens in the model ones
            # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module,
            # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
            # If we would directly use the brackets here, they may become part of another token
            self.replace_bos = True
            self.new_boq_id = tokenizer.convert_tokens_to_ids('[')
            self.new_bod_id = tokenizer.convert_tokens_to_ids('{')
            self.eoq_id = tokenizer.convert_tokens_to_ids(']')
            self.eod_id = tokenizer.convert_tokens_to_ids('}')
            self.max_seq_length_be = self.max_seq_length - 2
        else:
            self.added_tokens = dict(boq='[BOQ]', eoq='[EOQ]', bod='[BOD]', eod='[EOD]')
            self.replace_bos = False
            self.max_seq_length_be = self.max_seq_length - 1
        tokenizer.add_tokens(list(self.added_tokens.values()), special_tokens=True)

        embedding: nn.Embedding = self._first_module().auto_model.get_input_embeddings()
        if len(tokenizer) > embedding.num_embeddings:
            logger.info(f"Resizing embedding from {embedding.num_embeddings} to {len(tokenizer)}")
            self._first_module().auto_model.resize_token_embeddings(len(tokenizer))
            embedding = self._first_module().auto_model.get_input_embeddings()

        grad_mask = torch.zeros((embedding.num_embeddings, 1), dtype=torch.long)

        for k, v in self.added_tokens.items():
            setattr(self, k + '_token', v)
            index = tokenizer.convert_tokens_to_ids(v)
            setattr(self, k + '_id', index)
            grad_mask[index] = 1

        if setting and setting != 'sgpt':
            logger.info('Registering embedding_grad_mask and mask_grad_hook')
            self.register_buffer('embedding_grad_mask', grad_mask)
            embedding.weight.register_hook(self.mask_grad_hook)
            # print(grad_mask[250678:250684])

    def mask_grad_hook(self, grad):
        # mask grads
        # print('before', grad[64][:20])
        grad.mul_(self.embedding_grad_mask)
        # print('after', grad[64][:20])
        return grad

    def encode(
        self,
        sentences,
        is_query=None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        **kwargs
    ):
        kwargs.update(show_progress_bar=False)
        if self.encode_batch_size > 0:
            kwargs.update(batch_size=self.encode_batch_size)
        if (
            self.accelerator is not None and
            self.accelerator.num_processes > 1 and
            isinstance(sentences, list) and
            len(sentences) >= self.accelerator.num_processes
        ):
            shard_size = len(sentences) // self.accelerator.num_processes
            if shard_size * self.accelerator.num_processes < len(sentences):
                shard_size += 1
            i = self.accelerator.process_index * shard_size
            j = i + shard_size
            selected = sentences[i:j]
            pad_num = shard_size - len(selected)
            if pad_num > 0:  # pad to same size for tensor gathering
                selected = selected + [sentences[-1] for _ in range(pad_num)]
        else:
            shard_size = 0
            selected = sentences
            kwargs.update(convert_to_numpy=convert_to_numpy, convert_to_tensor=convert_to_tensor)

        if self.setting:
            if is_query is None:
                is_query = self.default_query
            logger.debug(f'encode - is_query : {is_query}')
            symbol: str = self.boq_token if is_query else self.bod_token
            if isinstance(selected, str):
                selected = symbol + selected
            else:
                selected = [symbol + s for s in selected]
        # logger.info(f"-----{len(selected)}------")

        if shard_size > 0:
            shard_embeddings = super().encode(selected, convert_to_tensor=True, **kwargs)
            embeddings = [shard_embeddings.clone() for _ in range(dist.get_world_size())]
            dist.all_gather(embeddings, shard_embeddings)
            embeddings = torch.cat(embeddings)
            del shard_embeddings
            # Remove padded sentences
            total_pad_num = shard_size * self.accelerator.num_processes - len(sentences)
            if total_pad_num > 0:
                embeddings = embeddings[:-total_pad_num]
            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
        else:
            embeddings = super().encode(selected, **kwargs)
        return embeddings

    def encode_queries(self, queries: List[str], **kwargs):
        return self.encode(queries, is_query=True, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        # borrowed from mteb.abstasks.AbsTaskRetrieval.DRESModel
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        return self.encode(sentences, is_query=False, **kwargs)

    def encode_multi_process(
        self,
        sentences: List[str],
        pool: Dict[str, object],
        chunk_size: int = None,
        **kwargs
    ):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :param kwargs: other keyword arguments for model.encode() such as batch_size
        :return: Numpy matrix with all embeddings
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, chunk, kwargs])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=self._encode_multi_process_worker,
                args=(cuda_id, self, input_queue, output_queue),
                daemon=True
            )
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                id, sentences, kwargs = input_queue.get()
                kwargs.update(device=target_device, show_progress_bar=False, convert_to_numpy=True)
                embeddings = model.encode(sentences, **kwargs)
                results_queue.put([id, embeddings])
            except queue.Empty:
                break

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # lowercase
        if self._first_module().do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        # Transfomers never adds special tokens for GPT models
        # - https://github.com/huggingface/transformers/issues/3311
        # hence we add them manually here:
        tokenizer: PreTrainedTokenizerFast = self._first_module().tokenizer
        if self.setting:
            output.update(self.tokenize_bos_eos(to_tokenize, tokenizer))
        else:
            output.update(tokenizer(
                *to_tokenize, padding=True, truncation='longest_first',
                return_tensors="pt", max_length=self.max_seq_length
            ))

        return output

    def tokenize_bos_eos(self, to_tokenize, tokenizer):
        out = tokenizer(
            *to_tokenize, padding=False, truncation='longest_first',
            max_length=self.max_seq_length_be
        )
        for seq, att in zip(out["input_ids"], out["attention_mask"]):
            if seq[0] == self.bod_id:
                if self.replace_bos:  # Replace with a different doc token if given
                    seq[0] = self.new_bod_id
                seq.append(self.eod_id)
            elif seq[0] == self.boq_id:
                if self.replace_bos:  # Replace with a different query token if given
                    seq[0] = self.new_boq_id
                seq.append(self.eoq_id)
            else:
                raise ValueError(f"Did not find BOS Token in sequence: {tokenizer.decode(seq)}")
            att.append(1)

        return tokenizer.pad(out, padding=True, return_tensors="pt")

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels).to(self._target_device)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            batch_to_device(tokenized, self._target_device)
            sentence_features.append(tokenized)

        return sentence_features, labels