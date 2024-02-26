
import os

from transformers import AutoConfig, AutoTokenizer

from sentence_transformers import SentenceTransformer
from beir.retrieval import models

import transformers
from transformers import AutoConfig, AutoModel
from model.contriever.src.beir_utils import DenseEncoderContrieverModel
from model.CustomModels import SimLMDEModel, CoCondenserModel, AnglEModel, E5Model, SentenceTransformerSpec, \
    InstructorModel, JinaModel
from model.model_collection import ModelClass


class CustomModel(ModelClass):
    def __init__(self, model_dir="/opt/data/IR_models/",  specific_model=None):
        super().__init__(model_dir)

        if specific_model is not None:
            self.names = [specific_model]
        else:
            self.names = [# 23 models
                "SGPT-125M-weightedmean-msmarco-specb-bitfit",
                "SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
                "SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
                "SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                "instructor-xl", "instructor-large", "instructor-base",
                "e5-small", "e5-base", "e5-large",
                "multilingual-e5-small", "multilingual-e5-base", "multilingual-e5-large",
                "e5-small-v2", "e5-base-v2", "e5-large-v2",
                "bge-small-en-v1.5", "bge-base-en-v1.5", "bge-large-en-v1.5",
                "UAE-Large-V1",
                "contriever",
                "simlm-base-msmarco-finetuned",
                "co-condenser-marco-retriever",]

        self.score_function = {
            "instructor-xl": "cos_sim", "instructor-large": "cos_sim", "instructor-base": "cos_sim",
            "SGPT-125M-weightedmean-msmarco-specb-bitfit": "cos_sim",
            "SGPT-1.3B-weightedmean-msmarco-specb-bitfit": "cos_sim",
            "SGPT-2.7B-weightedmean-msmarco-specb-bitfit": "cos_sim",
            "SGPT-5.8B-weightedmean-msmarco-specb-bitfit": "cos_sim",
            "multilingual-e5-large": "cos_sim", "multilingual-e5-base": "cos_sim", "multilingual-e5-small": "cos_sim",
            "e5-large-v2": "cos_sim", "e5-base-v2": "cos_sim", "e5-small-v2": "cos_sim",
            "e5-large": "cos_sim", "e5-base": "cos_sim", "e5-small": "cos_sim",
            "bge-small-en-v1.5": "cos_sim", "bge-base-en-v1.5": "cos_sim", "bge-large-en-v1.5": "cos_sim",
            "UAE-Large-V1": "cos_sim",
            "contriever": "dot",
            "simlm-base-msmarco-finetuned": "dot",
            "co-condenser-marco-retriever": "dot",
        }

    def load_model(self, name, cuda=True, model_name_or_path=None):
        assert name in self.names

        if name == "contriever":
            if model_name_or_path is None:
                model_name_or_path = "facebook/contriever-msmarco"
            model = DenseEncoderContrieverModel(model_name_or_path=model_name_or_path,
                                                cache_dir=self.model_dir,
                                                max_length=350,
                                                cuda=cuda)

        elif name == "simlm-base-msmarco-finetuned":
            if model_name_or_path is None:
                model_name_or_path = 'intfloat/simlm-base-msmarco-finetuned'
            model = SimLMDEModel(model_name_or_path=model_name_or_path,
                                 cache_dir=self.model_dir,
                                 cuda=cuda)

        elif name == "co-condenser-marco-retriever":
            if model_name_or_path is None:
                model_name_or_path = 'Luyu/co-condenser-marco-retriever'
            model = CoCondenserModel(name=model_name_or_path,
                                     cache_dir=self.model_dir, cuda=cuda)
        # elif name == "character-bert-dr":
        #     model = CoCondenserModel(name=self.model_dir+'/models--arvin--character-bert-dr/',
        #                              cache_dir=self.model_dir, cuda=cuda)
        elif name == "UAE-Large-V1":
            model = AnglEModel("WhereIsAI/UAE-Large-V1", cache_dir=self.model_dir, cuda=cuda)
        elif "bge-" in name:
            # "bge-large-en-v1.5", "bge-base-en-v1.5", "bge-small-en-v1.5",
            # "bge-large-en", "bge-base-en", "bge-small-en"
            model = AnglEModel("BAAI/" + name, cache_dir=self.model_dir, cuda=cuda)
        elif "e5-" in name:
            # "e5-base", "e5-small", "e5-large-v2"
            # "e5-small-v2", "e5-base-v2", "e5-large-v2"
            # "multilingual-e5-small", "multilingual-e5-base", "multilingual-e5-large"
            model = E5Model("intfloat/" + name, cache_dir=self.model_dir, cuda=cuda)
        elif "weightedmean-msmarco-specb-bitfit" in name:
            model = SentenceTransformerSpec(f"Muennighoff/{name}",
                                            cache_dir=self.model_dir, cuda=cuda)
        elif "instructor-" in name:
            model = InstructorModel(f"hkunlp/{name}", cache_dir=self.model_dir, cuda=cuda)
        else:
            raise "Unknown model name"
        return model


class BeirModels(ModelClass):
    def __init__(self, model_dir, specific_model=None):
        super().__init__(model_dir)

        if specific_model is not None:
            self.names = [specific_model]
        else:
            self.names = [
                # 24 models
                "jina-embeddings-v2-base-en", "jina-embeddings-v2-small-en",
                "gte-tiny", "all-mpnet-base-v2",
                "gtr-t5-base", "gtr-t5-large",  "gtr-t5-xl",
                "sentence-t5-xl", "sentence-t5-large",
                "stella-base-en-v2",
                "ember-v1",
                "sf_model_e5",
                "gte-large", "gte-base", "gte-small",
                "all-MiniLM-L6-v2", "all-MiniLM-L12-v2",
                "msmarco-MiniLM-L-6-v3", "msmarco-MiniLM-L-12-v3",
                "msmarco-distilbert-base-v2", "msmarco-distilbert-base-v3",
                "msmarco-distilbert-base-dot-prod-v3",
                "msmarco-roberta-base-ance-firstp",
                "msmarco-distilbert-base-tas-b" ]

        model_name_or_path = []
        for name in self.names:
            if "gte-" in name:
                if name == "gte-tiny":
                    model_name_or_path.append(f"TaylorAI/{name}")
                else:
                    model_name_or_path.append(f"thenlper/{name}")
            elif name == "ember-v1":
                model_name_or_path.append(f"llmrails/{name}")
            elif name == "sf_model_e5":
                model_name_or_path.append(f"jamesgpt1/{name}")
            elif name == "stella-base-en-v2":
                model_name_or_path.append(f"infgrad/{name}")
            elif "jina-embeddings-" in name:
                model_name_or_path.append(f"jinaai/{name}")
            else:
                model_name_or_path.append(f"sentence-transformers/{name}")
        self.model_name_or_path = model_name_or_path
        score_function = {}

        for name in self.names:
            score_function[name] = "dot"
        names_with_cos = ["jina-embeddings-v2-small-en", "jina-embeddings-v2-base-en",
                          "stella-base-en-v2",
                          "ember-v1",
                          "gte-large", "gte-base", "gte-small", "gte-tiny",
                          "all-MiniLM-L6-v2", "all-MiniLM-L12-v2",
                          "msmarco-MiniLM-L-6-v3", "msmarco-MiniLM-L-12-v3",
                          "msmarco-distilbert-base-v3"]
        for name in names_with_cos:
            score_function[name] = "cos_sim"
        score_function["dpr"] = "dot"
        self.score_function = score_function

    def download_models(self):

        for name in self.model_name_or_path:
            # print("Downloading", name)
            if "jina-embeddings-" in name:
                continue
            SentenceTransformer(model_name_or_path=name,
                                cache_folder=self.model_dir)

            # print("Finished loading")

    def load_model(self,  model_name, cuda=True, model_name_or_path=None):

        # find model_name in self.model_name_or_path
        for name in self.model_name_or_path:
            if model_name in name:
                model_name = name
                break

        # replace "/" with "_"
        model_name = model_name.replace("/", "_")
        model_dir = os.path.join(self.model_dir, model_name)
        model = models.SentenceBERT(model_dir)

        model.config = AutoConfig.from_pretrained(model_dir)
        model.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if cuda:
            model.q_model = model.q_model.cuda()
            model.doc_model = model.doc_model.cuda()
        return model


