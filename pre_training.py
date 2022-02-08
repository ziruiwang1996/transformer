import numpy as np
import torch
from embedding import onehot_emb, seg_emb, pos_emb
from masking import token_masking


# loading data
seqs = ['AEAKYAEENCNACCSICPLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
        'AEAKYAEENCNACCSICSLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
        'AEAKYAEENCNACCSICSLPNLTISQRIAFVYALYDDPSQSSELLSEAKKLNDSQAPK',
        'AEAKYAEENCNACCSICSLSNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
        'AEAKYAEENCNALSEIYYLPNLTSTQRCAFIKALCDDPSQSSELLSEAKKLNDSQAPK']

one_hot = []
seg_pos_emb = []
for seq in seqs:
    one_hot.append(onehot_emb(seq))
    emb = seg_emb( [(0,20),(20,38),(38,58)], 58) + pos_emb(58)
    seg_pos_emb.append(emb)

#masking
data = np.array(one_hot)
masked_repr = token_masking(data)
#print(data.shape)

emb = []
for i in np.arange(58):
    emb = masked_repr[i] + seg_pos_emb[i]
    emb.append(emb)
#x_data = torch.tensor(np.array(seq_list))
print(emb)