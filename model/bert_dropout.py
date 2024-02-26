import torch
import random
import numpy as np
from transformers import BertConfig


def bert_convert_to_mc_dropout(
        model: torch.nn.Module,
        config: BertConfig,
        p: float,
        start_from: int = 0,
        seed: int = None,
):
    shape = [config.max_length, config.hidden_size]  # shape for dropout in "BertEmbeddings", "BertSelfOutput", "BertOutput"
    # shape = [config.num_attention_heads, config.max_length, config.max_length]   # shape for dropout in "BertSelfAttention"

    assert 0 <= p <= 1, "p must be in [0, 1]"
    assert start_from < config.num_hidden_layers, "start_from must be < config.num_hidden_layers"

    for layer in model.encoder.layer[start_from:]:
        layer.output.dropout = DropoutMC(
                    p=p, activate=True, seed=seed, shape=shape
                ).to(model.device)
        if seed is not None:
            seed += 1


class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False, seed=None, shape=None):
        super().__init__()
        self.activate = activate
        self.p = p
        self.seed = seed
        if self.seed is not None:
            setup_seed(seed)
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            mask = binomial.sample([1] + shape)
            mask = mask.div_(1 - self.p)
            self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor):
        if self.activate:
            if 'mask' in self._buffers.keys():
                mask = self.mask.expand_as(x)
                return x * mask
            else:
                return torch.nn.functional.dropout(
                    x, self.p, training=self.activate
                )
        return x

    def extra_repr(self) -> str:
        return 'p={}, seed={} activate={}'.format(self.p, self.seed, self.activate)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
