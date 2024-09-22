import torch
import torch.nn as nn
import math

#refer: https://nlp.seas.harvard.edu/annotated-transformer/


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (heads * self.head_dim == embed_size ), "Embedding size should be divisible by heads"

        self.queries = nn.Linear(embed_size, embed_size, bias=False) #对输入query进行线性变换
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)

        self.fc_out = nn.Linear(embed_size, embed_size)


    def transpose_for_score(self, x):
        N, len = x.shape[0], x.shape[1]
        x = x.reshape(N,len, self.heads, self.head_dim )
        return x.permute(0,2,1,3)

    def forward(self, values, keys, query, mask):
        '''
        :param values: (batch_size, value_len, d_v)
        :param keys: (batch_size, key_len, d_k)
        :param query: (batch_size, query_len, d_q)
        :param mask:(batch_size, 1,1, src_len)
        :return:
        '''

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        values = self.transpose_for_score(values)
        keys = self.transpose_for_score(keys)
        query = self.transpose_for_score(query)

        energy = torch.matmul(query, keys.transpose(-1,-2))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("1e-20"))

        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)
        out = torch.matmul(attention, values)
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out




class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_size,
                 heads,
                 forward_expansion,
                 dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, keys, values, query, mask):
        attention = self.attention(keys, values, query, mask)
        x = self.dropout(self.norm1(attention + query))
        fc = self.fc(x)
        out = self.dropout(self.norm2(x + fc))

        return out


class PositionEmbedding(nn.Module):
    def __init__(self,
                 max_length,
                 embed_size):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length).unsqueeze(1)
        #张量[0, 1, 2, 3, 4]，经过unsqueeze(1)操作后变为形状为(5, 1)的二维张量，即[[0], [1], [2], [3], [4]]
        div = torch.exp(torch.arange(0.,embed_size,2) * math.log(10000.0) / embed_size)
        pe[:,0::2] = torch.sin(position * div)
        pe[:,1::2] = torch.cos(position * div)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x = x + self.pe[:, x.size(1)]
        return x


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 heads,
                 forward_expansion,
                 dropout,
                 num_layers,
                 device,
                 mex_length):
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = PositionEmbedding(mex_length, embed_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, forward_expansion, dropout)
             for _ in range(num_layers)]
        )
        self.device = device
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask):
        #x:(batch_size, seq_len)
        x = self.word_embedding(x)
        out = self.position_embedding(x)

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_size,
                 heads,
                 forward_expansion,
                 dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.tranformer_block = TransformerBlock(embed_size, heads, forward_expansion, dropout)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keys, values, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.tranformer_block(keys, values, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 heads,
                 forward_expansion,
                 dropout,
                 num_layers,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionEmbedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.device = device


    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.embedding(x)
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        return x



class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=512,
                 heads = 8,
                 num_layers = 6,
                 forward_expansion=4,
                 dropout = 0,
                 max_length=100,
                 device="cpu"):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size,
                               embed_size,
                               heads,
                               forward_expansion,
                               dropout,
                               num_layers,
                               device,
                               max_length)

        self.decoder = Decoder(trg_vocab_size,
                               embed_size,
                               heads,
                               forward_expansion,
                               dropout,
                               num_layers,
                               device,
                               max_length)

        self.projection = nn.Linear(embed_size, trg_vocab_size)
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx


    def make_src_pad(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_pad(self, trg):
        trg_len = trg.shape[1]
        trg_mask = self.make_src_pad(trg)
        trg_mask = torch.tril(trg_mask.expand(-1,-1,trg_len,-1))
        return trg_mask.to(self.device)

    def forward(self, src, src_mask, trg, trg_mask):
        src_mask = self.make_src_pad(src)
        enc_out = self.encoder(src, src_mask)
        trg_mask = self.make_trg_pad(trg)
        dec_out = self.decoder(trg, enc_out, src_mask, trg_mask)
        out = self.projection(dec_out)
        return out



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)

    out = model(x, x, trg[:,:-1],x)
    print("x.shape:", x.shape)
    print("trg.shape:", trg.shape)
    print(out.shape)