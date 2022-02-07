import torch
import torch.nn as nn
from embedding import onehot_emb, seg_emb, pos_emb

class multihead_attention(nn.Module):
        def __init__(self, d_model, heads):
                super(multihead_attention, self).__init__()
                self.d_model = d_model
                self.heads = heads
                self.d_k = d_model // heads

                assert (self.d_k * heads == d_model), 'Invalid d_model'

                #linear transformation
                self.queries = nn.Linear(self.d_model, self.d_model, bias=False)
                self.values = nn.Linear(self.d_model, self.d_model, bias=False)
                self.keys = nn.Linear(self.d_model, self.d_model, bias=False)
                self.fully_connected_out = nn.Linear(heads*self.d_k, d_model)

        def forward(self, values, keys, queries, mask):
                # dimension: samples x seqlen x embedding
                n = queries.shape[0] #number of training samples
                v_len, k_len, q_len = values.shape[1], keys.shape[1], queries.shape[1] #seq len
                # split embedding by heads
                queries = self.queries(queries).reshape(n, q_len, self.heads, self.d_k)
                values = self.values(values).reshape(n, v_len, self.heads, self.d_k)
                keys = self.keys(keys).reshape(n, k_len, self.heads, self.d_k)
                # Q(SeqLen x d_model) @ K.T(d_model x SeqLen) -> SeqLen x SeqLen
                # n x heads x Q(q_len(seq) x d_k) @ n x heads x K.T(d_k x k_len(seq)) -> n x heads x q_len x k_len
                # multi-dimensional linear algebraic array operations
                qxkT = torch.einsum('nqhd, nkhd -> nhqk', [queries, keys])

                # Mask padded indices so their weights become 0
                if mask is not None:
                        qxkT = qxkT.masked_fill(mask == 0, -1e10)

                # shape: (n, heads, q_len, k_len)
                attention = torch.softmax(qxkT/((self.d_k)**(1/2)), dim=-1)  #normalizing the last dimention k_len

                out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(n, q_len, self.heads*self.d_k)
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
                self.fc_feed_forward = nn.Sequential(nn.Linear(d_model, ff_expansion*d_model),
                                                  nn.ReLU(),
                                                  nn.Linear(ff_expansion*d_model, d_model))
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)

        def forward(self, value, key, query, mask):
                sublayer_1 = self.attention(value, key, query, mask)
                sublayer_2 = self.dropout(self.norm1(query+sublayer_1))
                sublayer_3 = self.fc_feed_forward(sublayer_2)
                out = self.dropout(self.norm2(sublayer_2+sublayer_3))
                return out


class encoder(nn.Module):
        def __init__(self, sequence_string, d_model, num_layer, heads, device, ff_expansion, dropout, seq_len, list_of_tuples):
                super(encoder, self).__init__()
                self.d_model = d_model
                self.device = device
                self.token_emb = onehot_emb(sequence_string, d_model)
                self.segment_emb = seg_emb(list_of_tuples, seq_len, d_model)
                self.position_emb = pos_emb(seq_len, d_model)

                self.layers = nn.ModuleList([encoder_block(d_model=d_model,
                                                           heads=heads,
                                                           dropout=dropout,
                                                           ff_expansion=ff_expansion)
                                             for _ in range(num_layer)
                                             ])
                self.dropout = nn.Dropout(dropout)

        def forward(self, input, mask):
                #n, seq_len = input.shape
                emb = self.dropout(self.token_emb(input)+self.segment_emb(input)+self.position_emb(input))
                for layer in self.layers:
                        out = layer(emb, emb, emb, mask) # k, v, q are the same for now    #??????
                return out


class decoder_block(nn.Module):
        def __init__(self, d_model, heads, ff_expansion, dropout, device):
                super(decoder_block, self).__init__()
                self.attention = multihead_attention(d_model, heads)
                self.norm = nn.LayerNorm(d_model)
                self.encoder_block = encoder_block(d_model, heads, dropout, ff_expansion)
                self.dropout = nn.Dropout(dropout)

        def forward(self, input, value, key, src_mask, trg_mask):
                # trg_mask: decoder self-attention
                # src_mask: encoder self-attention & decoder encoder-decoder attention
                attention = self.attention(input, input, input, trg_mask)
                query = self.dropout(self.norm(attention+input)) # Q from target sequence (outputs)
                out = self.encoder_block(value, key, query, src_mask) # V, K from encoder
                return out


class decoder(nn.Module):
        def __init__(self, sequence_string, d_model, num_layer, heads, device, ff_expansion, dropout, trg_seq_len, list_of_tuples):
                super(decoder, self).__init__()
                self.d_model = d_model
                self.device = device
                self.token_emb = onehot_emb(sequence_string, d_model)
                self.segment_emb = seg_emb(list_of_tuples, seq_len, d_model)
                self.position_emb = pos_emb(seq_len, d_model)

                self.layers = nn.ModuleList([decoder_block(d_model, heads, ff_expansion, dropout, device)
                                             for _ in range(num_layer)
                                             ])     #????????????????????????
                self.fc_feed_forward = nn.Linear(d_model, trg_seq_len)
                self.dropout = nn.Dropout(dropout)

        def forward(self, input, enc_out, src_mask, trg_mask):
                n, seq_len = input.shape
                emb = self.dropout(self.token_emb(input) + self.segment_emb(input) + self.position_emb(input))
                for layer in self.layers:
                        out = layer(emb, enc_out, enc_out, src_mask, trg_mask) #??????
                out = self.fc_feed_forward(out)
                return out


class transformer(nn.Module):
        def __init__(self, src_vocab_size,
                     trg_vocab_size,
                     src_pad_idx,
                     trg_pad_idx,
                     embed_size=512,
                     num_layers=6,
                     forward_expansion=4,
                     heads=8,
                     dropout=0,
                     device="cpu",
                     max_length=100,):
                super(transformer, self).__init__()
                self.encoder = encoder(src_vocab_size,
                                       embed_size,
                                       num_layers,
                                       heads,
                                       device,
                                       forward_expansion,
                                       dropout,
                                       max_length,)
                self.decoder = decoder(trg_vocab_size,
                                       embed_size,
                                       num_layers,
                                       heads,
                                       forward_expansion,
                                       dropout,
                                       device,
                                       max_length,)
                self.src_pad_idx = src_pad_idx
                self.trg_pad_idx = trg_pad_idx
                self.device = device

        def make_src_mask(self):
                pass

        def make_trg_mask(self, target_seq):
                n, trg_len = target_seq.shape
                trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(n, 1, trg_len, trg_len)
                return trg_mask.to(self.device)  #?????????

        def forward(self, src, trg):
                src_mask = self.make_src_mask(src)
                trg_mask = self.make_trg_mask(trg)
                enc_src = self.encoder(src, src_mask)
                out = self.decoder(trg, enc_src, src_mask, trg_mask)
                return out