from embedding import onehot_emb, seg_emb, pos_emb
import torch
import torch.nn as nn
import numpy as np

seqs = ['AEAKYAEENCNACCSICPLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
        'AEAKYAEENCNACCSICSLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
        'AEAKYAEENCNACCSICSLPNLTISQRIAFVYALYDDPSQSSELLSEAKKLNDSQAPK',
        'AEAKYAEENCNACCSICSLSNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
        'AEAKYAEENCNALSEIYYLPNLTSTQRCAFIKALCDDPSQSSELLSEAKKLNDSQAPK']

seq_list=[]
for seq in seqs:
        emb = onehot_emb(seq) + seg_emb( [(0,20),(20,38),(38,58)], 58) + pos_emb(58)
        seq_list.append(emb)
x_data = torch.tensor(np.array(seq_list))
print(x_data.size())

class self_attention(nn.Module):
        def __init__(self, d_model, heads):
                super(self_attention, self).__init__()
                self.emb = d_model
                self.heads = heads
                self.d_k = d_model // heads

                self.values = nn.Linear(self.d_k, self.d_k, bias=False)
                self.keys = nn.Linear(self.d_k, self.d_k, bias=False)
                self.queries = nn.Linear(self.d_k, self.d_k, bias=False)
                self.fully_connected_out = nn.Linear(heads*self.d_k, d_model)

        def forward(self, values, keys, query, mask):
                n = query.shape[0] #number of training examples
                v_len, k_len, q_len = values.shape[1], keys.shape[1], query.shape[1]
                # split
                values = values.reshape(n, v_len, self.heads, self.d_k)
                keys = keys.reshape(n, k_len, self.heads, self.d_k)
                query = query.reshape(n, q_len, self.heads, self.d_k)

                qxkT = torch.einsum('nqhd, nkhd -> nhqk', [queries, keys])
                # queries shape: (N, query_len, heads, heads_dim),
                # keys shape: (N, key_len, heads, heads_dim)
                # energy: (N, heads, query_len, key_len)

                # Mask padded indices so their weights become 0
                if mask is not None:
                        qxkT = qxkT.masked_fill(mask == 0, float("-1e20"))

                attention = torch.softmax(qxkT/((self.d_k)**(1/2)), dim=3)  #why 3?
                # attention shape: (N, heads, query_len, key_len)

                out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
                        N, query_len, self.heads * self.head_dim
                )
                # attention shape: (N, heads, query_len, key_len)
                # values shape: (N, value_len, heads, heads_dim)
                # out after matrix multiply: (N, query_len, heads, head_dim), then
                # we reshape and flatten the last two dimensions.

                out = self.fully_connected_out(out)
                # Linear layer doesn't modify the shape, final shape will be
                # (N, query_len, embed_size)
                return out

class transformer_block(nn.Module):
        def __init__(self, d_model, heads, dropout, forward_expansion):
                super(transformer_block, self).__init__()
                self.attention = self_attention(d_model, heads)
                self.norm1 = nn.LayerNorm(d_model)
                self.feed_forward = nn.Sequential(nn.Linear(d_model, forward_expansion*d_model),
                                                  nn.ReLU(),
                                                  nn.Linear(forward_expansion*d_model, d_model))
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)

        def forward(self, value, key, query, mask):
                attention = self.attention(value, key, query, mask)
                x = self.dropout(self.norm1(attention+query))
                forward = self.feed_forward(x)
                out = self.dropout(self.norm2(forward+x))
                return out

class encoder(nn.Module):
        def __init__(self, src_vocab_size, d_model, num_layer, heads, device, forward_expansion, dropout, max_len):
                super(encoder, self).__init__()
                 

