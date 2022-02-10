import numpy as np
from embedding import onehot_emb, seg_emb, pos_emb
import random


def token_masking(matrix_after_onehot, d_model=21, seq_lenth=58, sample_size=5, perc=0.15):
    mask = np.random.random((sample_size, seq_lenth))
    matrix_after_onehot[mask<(perc*0.9), :]=0 #15% * (100%-10%) replace with [MASK]=[0,0...]
    coord = np.where(mask<(perc*0.1)) #15% * 10% replace with random token
    for index in list(zip(coord[0],coord[1])):
        matrix_after_onehot[index[0], index[1], random.randrange(d_model)] = 1
    return matrix_after_onehot

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

# token masking
data = np.array(one_hot)
masked_repr = token_masking(data)
#print(data.shape)

emb_lst = []
for i in np.arange(58):
    emb = masked_repr[i] + seg_pos_emb[i]
    emb_lst.append(emb)
#x_data = torch.tensor(np.array(seq_list))
print(emb_lst)

#encoder 12x

