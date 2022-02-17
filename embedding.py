import numpy as np
import torch

# out Tensor shape: sample_size x seq_len x d_model
def embedding(seq_strs, d_model, segment_pos):
    embs = []
    for seq_str in seq_strs:
        seq_len = len(seq_str)
        # one hot embedding
        tokens = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        order = np.arange(len(tokens))
        look_up = dict(zip(tokens, order))
        one_hot_emb = np.zeros((seq_len, d_model))  # dimension: seq_len * d_model
        n = 0
        for aa in list(seq_str):
            if aa == '-':
                one_hot_emb[n, :] = 0 # masked token vector elements = 0
            else:
                one_hot_emb[n][look_up.get(aa)] = 1
            n += 1

        # segment embedding
        seg_id = np.arange(len(segment_pos))
        seg_emb = np.zeros((seq_len, d_model))
        n = 0
        for key, value in segment_pos.items():
            seg_emb[int(key):int(value), :] = seg_id[n]
            n += 1

        # positional embedding
        pos = np.arange(seq_len)
        min_freq = 1e-4
        freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)
        pos_emb = pos.reshape(-1, 1) * freqs.reshape(1, -1)
        pos_emb[:, ::2] = np.cos(pos_emb[:, ::2])
        pos_emb[:, 1::2] = np.sin(pos_emb[:, 1::2])
        embs.append(one_hot_emb+seg_emb+pos_emb)
    embs = np.array(embs)
    return torch.from_numpy(embs) #shape: n x q_len x d_model

'''
input = ['AEAKYAEENCNALSEIYYLPNLTSTQRCAFIKALCDDPSQSSELLSEAKKLNDSQAPK', 
         'AEAKYAEENCNACCSICSLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
         'AEAKYAEENCNACCSICSLSNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK']
a = embedding(input, 20, {0:20, 20:38, 38:58})
print(a.shape)'''