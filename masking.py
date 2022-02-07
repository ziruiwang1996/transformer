import numpy as np
import random
from embedding import onehot_emb

seqs = ['AEAKYAEENCNACCSI',
        'SLPNLTISQRIAFIYA']

A = []
for seq in seqs:
    A.append(onehot_emb(seq))
data = np.array(A)
#print(data.shape)

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


def token_masking(matrix_after_onehot=data, d_model=21, seq_lenth=16, sample_size=2, perc=0.15):
    mask = np.random.random((sample_size, seq_lenth))
    matrix_after_onehot[mask<(perc*0.9), :]=0 #15% * (100%-10%) replace with [MASK]=[0,0...]
    coord = np.where(mask<(perc*0.1)) #15% * 10% replace with random token
    for index in list(zip(coord[0],coord[1])):
        matrix_after_onehot[index[0], index[1], random.randrange(d_model)] = 1
    return matrix_after_onehot

def decoder_self_attention_mask(seq_lenth=16):
    pass
