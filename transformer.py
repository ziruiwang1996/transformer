# The implementation of attention all you need transformer network.

import torch
import torch.nn as nn
from embedding import Embedding

class Multihead_Attention(nn.Module):
        def __init__(self, d_model, heads):
                super(Multihead_Attention, self).__init__()
                self.d_model = d_model
                self.heads = heads
                self.d_k = d_model // heads
                assert (self.d_k * heads == d_model), 'Invalid d_model'
                self.queries = nn.Linear(self.d_model, self.d_model, bias=False)
                self.values = nn.Linear(self.d_model, self.d_model, bias=False)
                self.keys = nn.Linear(self.d_model, self.d_model, bias=False)
                self.fully_connected_out = nn.Linear(heads*self.d_k, d_model)

        def forward(self, value, key, query, mask):
                # dimension: samples x seq_len x d_model
                n = query.shape[0] #number of training samples
                v_len, k_len, q_len = value.shape[1], key.shape[1], query.shape[1] #seq len
                # linear transformation
                value = self.values(value.float())
                key = self.keys(key.float())
                query = self.queries(query.float())
                # split d_model by heads
                query = query.reshape(n, q_len, self.heads, self.d_k)
                value = value.reshape(n, v_len, self.heads, self.d_k)
                key = key.reshape(n, k_len, self.heads, self.d_k)
                # Q(SeqLen x d_model) @ K.T(d_model x SeqLen) -> SeqLen x SeqLen
                # n x heads x Q(q_len(seq) x d_k) @ n x heads x K.T(d_k x k_len(seq)) -> n x heads x q_len x k_len
                # multi-dimensional linear algebraic array operations
                qxkT = torch.einsum('nqhd, nkhd -> nhqk', [query, key])
                # attention become 0 where element = 0 in mask
                if mask is not None:
                        qxkT = qxkT.masked_fill(mask == 0, -1e10)
                # shape: (n, heads, q_len, k_len)
                attention = torch.softmax(qxkT/((self.d_k)**(1/2)), dim=-1)  # normalizing the last dimension k_len
                # matrix multiplication (n, q_len, heads, d_k), Merge: reshape and flatten the last two dimensions.
                out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(n, q_len, self.heads * self.d_k)
                out = self.fully_connected_out(out) # out shape: (n, q_len, d_model)
                return out


# multi-head attention + add&normalization + feed forward + add&normalization
class Encoder_Block(nn.Module):
        def __init__(self, d_model, heads, dropout, ff_expansion):
                super(Encoder_Block, self).__init__()
                self.attention = Multihead_Attention(d_model, heads)
                self.norm1 = nn.LayerNorm(d_model)
                # two linear transformations with a ReLU activation in between
                self.fc_feed_forward = nn.Sequential(nn.Linear(d_model, ff_expansion * d_model),
                                                  nn.GELU(), #nn.ReLU(),
                                                  nn.Linear(ff_expansion * d_model, d_model),)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)

        def forward(self, value, key, query, mask):
                sublayer_1 = self.attention(value, key, query, mask) # out shape: (n, q_len, d_model)
                sublayer_2 = self.dropout(self.norm1((query+sublayer_1).float()))
                sublayer_3 = self.fc_feed_forward(sublayer_2)
                out = self.dropout(self.norm2(sublayer_2+sublayer_3))
                return out


class Encoder(nn.Module):
        def __init__(self, d_model, heads, dropout, ff_expansion, num_layer, seq_len, device):
                super(Encoder, self).__init__()
                self.d_model = d_model
                self.device = device
                self.layers = nn.ModuleList([Encoder_Block(d_model=d_model,
                                                           heads=heads,
                                                           dropout=dropout,
                                                           ff_expansion=ff_expansion)
                                             for _ in range(num_layer)
                                             ])  # running block for num_layer times
                self.dropout = nn.Dropout(dropout)
                self.emb = Embedding(d_model=d_model, seq_len=seq_len)

        def forward(self, enc_input, mask):
                emb = self.emb.one_hot_and_pos(enc_input).to(self.device) #shape: sample, seq_len, d_model
                out = self.dropout(emb)
                for layer in self.layers:
                        out = layer(out, out, out, mask)  # out from previous block feed to the next block for num_layer times
                return out


class Decoder_Block(nn.Module):
        def __init__(self, d_model, heads, dropout, ff_expansion):
                super(Decoder_Block, self).__init__()
                self.attention = Multihead_Attention(d_model, heads)
                self.norm = nn.LayerNorm(d_model)
                self.cross_attention_block = Encoder_Block(d_model, heads, dropout, ff_expansion) # same architecture as encoder block
                self.dropout = nn.Dropout(dropout)

        def forward(self, enc_value, enc_key, dec_input, src_mask, trg_mask):
                # trg_mask: decoder mask future tokens
                # src_mask: encoder mask paddings
                self_attention = self.attention(dec_input, dec_input, dec_input, trg_mask)
                dec_query = self.dropout(self.norm((self_attention+dec_input).float())) # Q from decoder input
                out = self.cross_attention_block(enc_value, enc_key, dec_query, src_mask) # V, K from encoder
                return out


class Decoder(nn.Module):
        def __init__(self, d_model, heads, dropout, ff_expansion, num_layer, device, seq_len, trg_vocab_size):
                super(Decoder, self).__init__()
                self.d_model = d_model
                self.device = device
                self.layers = nn.ModuleList([Decoder_Block(d_model, heads, dropout, ff_expansion)
                                             for _ in range(num_layer)
                                             ]) # running block for num_layer times
                self.fc_feed_forward = nn.Linear(d_model, trg_vocab_size)
                self.dropout = nn.Dropout(dropout)
                self.emb = Embedding(d_model=d_model, seq_len=seq_len)

        def forward(self, dec_input, enc_out, src_mask, trg_mask):
                emb = self.emb.one_hot_and_pos(dec_input).to(self.device)  # shape: sample, seq_len, d_model
                dec_input = self.dropout(emb)
                for layer in self.layers:
                        out = layer(enc_out, enc_out, dec_input, src_mask, trg_mask) # out from dec_input
                out = self.fc_feed_forward(out) # Linear layer after decode blocks
                return out


class Transformer(nn.Module):
        def __init__(self, src_pad_idx, trg_pad_idx, d_model=20, heads=4, dropout=0.1, ff_expansion=4, num_layer=6, device="cpu", seq_len=59, trg_vocab_size=20):
                super(Transformer, self).__init__()
                self.encoder = Encoder(d_model, heads, dropout, ff_expansion, num_layer, seq_len, device)
                self.decoder = Decoder(d_model, heads, dropout, ff_expansion, num_layer, seq_len, device, trg_vocab_size)
                self.src_pad_idx = src_pad_idx
                self.trg_pad_idx = trg_pad_idx
                self.device = device
                self.softmax = nn.Softmax(dim=-1)

        def make_src_mask(self, src_seq):
                src_mask = (src_seq != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, src_len)
                return src_mask.to(self.device)

        def make_trg_mask(self, trg_seq):
                n, trg_len = len(trg_seq), len(trg_seq[0])
                trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(n, 1, trg_len, trg_len)
                return trg_mask.to(self.device)

        def forward(self, src_seq, trg_seq):
                src_mask = self.make_src_mask(src_seq)
                trg_mask = self.make_trg_mask(trg_seq)
                out_enc = self.encoder(src_seq, src_mask)
                out = self.decoder(trg_seq, out_enc, src_mask, trg_mask)
                out = self.softmax(out)
                return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    input = ['AEAKYAEENCNALSEIYYLPNLTSTQRCAFIKALCDDPSQSSELLSEAKKLNDSQAPK', 'AEAKYAEENCNACCSICSLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK']
    trg = ['AEAKYAEENCNACCSICSLPNLTISQRIAFVYALYDDPSQSSELLSEAKKLNDSQAPK', 'AEAKYAEENCNACCSICSLSNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK']
    src_pad_idx = 0
    trg_pad_idx = 0
    model = Transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, device=device)
    out = model(input, trg)
    print(out.shape)