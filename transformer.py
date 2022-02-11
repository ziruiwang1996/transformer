# The implementation of attention all you need transformer network.
# Embedding included segment embedding

import torch
import torch.nn as nn
from embedding import embedding

class multihead_attention(nn.Module):
        def __init__(self, d_model, heads):
                super(multihead_attention, self).__init__()
                self.d_model = d_model
                self.heads = heads
                self.d_k = d_model // heads
                assert (self.d_k * heads == d_model), 'Invalid d_model'
                self.queries = nn.Linear(self.d_model, self.d_model, bias=False)
                self.values = nn.Linear(self.d_model, self.d_model, bias=False)
                self.keys = nn.Linear(self.d_model, self.d_model, bias=False)
                self.fully_connected_out = nn.Linear(heads*self.d_k, d_model)

        def forward(self, value, key, query, mask):
                # dimension: samples x seqlen x embedding
                n = query.shape[0] #number of training samples
                v_len, k_len, q_len = value.shape[1], key.shape[1], query.shape[1] #seq len
                # linear transformation
                value = self.values(value.float())
                key = self.keys(key.float())
                query = self.queries(query.float())
                # split embedding by heads
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
                attention = torch.softmax(qxkT/((self.d_k)**(1/2)), dim=-1)  #normalizing the last dimention k_len
                out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(n, q_len, self.heads * self.d_k)
                # matrix multiplication D: (n, q_len, heads, d_k), Merge: reshape and flatten the last two dimensions.
                out = self.fully_connected_out(out) # out shape: (n, q_len, d_model)
                return out


# multi-head attention + add&normalization + feed forward + add&normalization
class encoder_block(nn.Module):
        def __init__(self, d_model, heads, dropout, ff_expansion):
                super(encoder_block, self).__init__()
                self.attention = multihead_attention(d_model, heads)
                self.norm1 = nn.LayerNorm(d_model)
                # two linear transformations with a ReLU activation in between
                self.fc_feed_forward = nn.Sequential(nn.Linear(d_model, ff_expansion * d_model),
                                                  nn.ReLU(),
                                                  nn.Linear(ff_expansion * d_model, d_model),)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)

        def forward(self, value, key, query, mask):
                sublayer_1 = self.attention(value, key, query, mask)
                sublayer_2 = self.dropout(self.norm1((query+sublayer_1).float()))
                sublayer_3 = self.fc_feed_forward(sublayer_2)
                out = self.dropout(self.norm2(sublayer_2+sublayer_3))
                return out


class encoder(nn.Module):
        def __init__(self, d_model, num_layer, heads, device, ff_expansion, dropout):
                super(encoder, self).__init__()
                self.d_model = d_model
                self.device = device
                self.layers = nn.ModuleList([encoder_block(d_model=d_model,
                                                           heads=heads,
                                                           dropout=dropout,
                                                           ff_expansion=ff_expansion)
                                             for _ in range(num_layer)
                                             ])  # running block for num_layer times
                self.dropout = nn.Dropout(dropout)

        def forward(self, input, segment_pos, mask):
                emb = embedding(input, self.d_model, segment_pos).to(self.device)
                out = self.dropout(emb)
                for layer in self.layers:
                        out = layer(out, out, out, mask)  # out from previous block feed to the next block for num_layer times
                return out


class decoder_block(nn.Module):
        def __init__(self, d_model, heads, dropout, ff_expansion):
                super(decoder_block, self).__init__()
                self.attention = multihead_attention(d_model, heads)
                self.norm = nn.LayerNorm(d_model)
                self.cross_attention_block = encoder_block(d_model, heads, dropout, ff_expansion) # same architecture as encoder block
                self.dropout = nn.Dropout(dropout)

        def forward(self, input, value, key, src_mask, trg_mask):
                # trg_mask: decoder self-attention, mask future tokens
                # src_mask: encoder self-attention & decoder cross-attention, mask paddings
                attention = self.attention(input, input, input, trg_mask)
                query = self.dropout(self.norm((attention+input).float())) # Q from target sequence (outputs)
                out = self.cross_attention_block(value, key, query, src_mask) # V, K from encoder
                return out


class decoder(nn.Module):
        def __init__(self, d_model, num_layer, heads, device, ff_expansion, dropout, trg_seq_len):
                super(decoder, self).__init__()
                self.d_model = d_model
                self.device = device
                self.layers = nn.ModuleList([decoder_block(d_model, heads, dropout, ff_expansion)
                                             for _ in range(num_layer)
                                             ]) # running block for num_layer times
                self.fc_feed_forward = nn.Linear(d_model, trg_seq_len) # Linear layer
                self.dropout = nn.Dropout(dropout)

        def forward(self, dec_input, enc_out, segment_pos, src_mask, trg_mask):
                emb = embedding(dec_input, self.d_model, segment_pos).to(self.device)
                out = self.dropout(emb)
                for layer in self.layers:
                        out = layer(out, enc_out, enc_out, src_mask, trg_mask) # out from dec_input
                out = self.fc_feed_forward(out)
                return out


class transformer(nn.Module):
        def __init__(self, src_pad_idx, trg_pad_idx, d_model=20,
                     num_layer=6, heads=4, ff_expansion=4, dropout=0.1,
                     device="cpu", trg_seq_len=58):
                super(transformer, self).__init__()
                self.encoder = encoder(d_model, num_layer, heads, device, ff_expansion, dropout)
                self.decoder = decoder(d_model, num_layer, heads, device, ff_expansion, dropout, trg_seq_len)
                self.src_pad_idx = src_pad_idx
                self.trg_pad_idx = trg_pad_idx
                self.device = device

        def make_src_mask(self, src_seq):
                pass
                #src_mask = (src_seq != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, src_len)
                #return src_mask.to(self.device)

        def make_trg_mask(self, trg_seq):
                n, trg_len = len(trg_seq), len(trg_seq[0])
                trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(n, 1, trg_len, trg_len)
                return trg_mask.to(self.device)

        def forward(self, src_seq, trg_seq, segment_pos):
                src_mask = self.make_src_mask(src_seq)
                trg_mask = self.make_trg_mask(trg_seq)
                out_enc = self.encoder(src_seq, segment_pos, src_mask)
                out = self.decoder(trg_seq, out_enc, segment_pos, src_mask, trg_mask)
                return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    input = ['AEAKYAEENCNALSEIYYLPNLTSTQRCAFIKALCDDPSQSSELLSEAKKLNDSQAPK', 'AEAKYAEENCNACCSICSLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK']
    trg = ['AEAKYAEENCNACCSICSLPNLTISQRIAFVYALYDDPSQSSELLSEAKKLNDSQAPK', 'AEAKYAEENCNACCSICSLSNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK']
    src_pad_idx = 0
    trg_pad_idx = 0
    model = transformer(src_pad_idx, trg_pad_idx, device=device)
    out = model(input, trg, {0:20, 20:38, 38:58})
    print(out.shape)