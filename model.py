from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm 
import math
import random
from torch.distributions.categorical import Categorical


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freqs = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * \
                          (-math.log(10000) / embed_dim)).unsqueeze(0)

        arguments = positions * freqs
        pe[:, 0::2] = torch.sin(arguments)
        pe[:, 1::2] = torch.cos(arguments)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.2):
        """
        Args:
            embed_dim: dimensionality of embedding (total)
            num_heads: number of heads (must divide embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout = dropout

        self._reset_parameters()

    # original implementation uses this initialization
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        bs, l, _ = x.size()
        q = q.reshape(bs, l, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        k = k.reshape(bs, l, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        v = v.reshape(bs, l, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)
        
        
        #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                      attn_mask=mask,
                                                                      dropout_p=self.dropout)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(bs, l, self.embed_dim)

        res_mha = self.o_proj(output)
        return res_mha    
    

class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, feedforward_dim, activation=nn.ReLU, dropout=0.1):
        super().__init__()
        self.self_attention = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, feedforward_dim),
                activation(),
                nn.Linear(feedforward_dim, embed_dim)
        )

    def forward(self, x, mask=None):
        x = self.norm1(x)
        out = self.self_attention(x, mask) + x
        
        out = self.norm2(out)
        res = self.feedforward(out) + out

        return res

    
class LanguageModel(nn.Module):
    def __init__(self,
                 dataset,
                 num_encoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 vocab_size: int,
                 dim_feedforward: int = 512,
                 max_length: int = 512,
                 dropout: float = 0.1):
        super(LanguageModel, self).__init__()
        self.dataset = dataset
        self.emb_dim = emb_size
        encoder_blocks = [EncoderBlock(embed_dim=emb_size,
                                       num_heads=nhead,
                                       feedforward_dim=dim_feedforward,
                                       activation=nn.GELU,
                                       dropout=dropout) for _ in range(num_encoder_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_blocks)
        self.pred = nn.Linear(emb_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.max_length = max_length

    def forward(self, src: Tensor):
        src_emb = self.positional_encoding(self.embedding(src.long()) * math.sqrt(self.emb_dim))
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device)
        for block in self.transformer_encoder:
            src_emb = block(src_emb, mask=src_mask)
        return self.pred(src_emb)
    
    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        device = next(self.parameters()).device
        self.eval()
        tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        
        embs = self.positional_encoding(self.embedding(tokens.long()) * math.sqrt(self.emb_dim))
        for block in self.transformer_encoder:
            embs = block(embs)
        logits = self.pred(embs)
        new_tokens = Categorical(logits=logits[:, -1:] / temp).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)
        
        while tokens.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            embs = self.positional_encoding(self.embedding(new_tokens.long()) * math.sqrt(self.emb_dim))
            for block in self.transformer_encoder:
                embs = block(embs)
            logits = self.pred(embs)
            new_tokens = Categorical(logits=(logits[:, -1:]) / temp).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        return self.dataset.ids2text(tokens.squeeze())
