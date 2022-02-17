import numpy as np
from embedding import *
import random
from transformer import *

# load data
input_seqs = ['AEAKYAEENCNALSEIYYLPNLTSTQRCAFIKALCDDPSQSSELLSEAKKLNDSQAPK',
              'AEAKYAEENCNACCSICSLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
              'AEAKYAEENCNACCSICSLSNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK']
# build a look up table
one_hots = np.zeros((20, 20))
np.fill_diagonal(one_hots, 1)
tokens = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
order = list(range(1,len(tokens)+1))
look_up = dict(zip(tokens, order))
look_up['-'] = 0
#print(look_up)

def get_key(val):
    for key, value in look_up.items():
        if val == value:
            return key
    return "key doesn't exist"

def token_id(input_seqs):
    indices_all_seqs = []
    for seq in input_seqs:
        indices_seq = []
        for aa in list(seq):
            indices_seq.append(look_up.get(aa))
        indices_all_seqs.append(indices_seq)
    return np.array(indices_all_seqs)

def token_masking(input_seqs):
    n, seq_lenth = len(input_seqs), len(input_seqs[0])
    mask = []
    for i in range(n):
        mask.append(np.random.rand(seq_lenth) < 0.15)
    mask = torch.from_numpy(np.array(mask))

    masked_spots = []
    indices_all_seqs = token_id(input_seqs)
    for i in range(n):
        masked_spots.append(torch.flatten(mask[i].nonzero()).tolist())
    for i in range(n):
        indices_all_seqs[i, masked_spots[i]] = 0

    sequences = []
    for indices in indices_all_seqs:
        sequence = ''
        for index in indices:
            sequence = sequence + get_key(index)
        sequences.append(sequence)
    return masked_spots, indices_all_seqs, sequences


class bert(nn.Module):
    def __init__(self, d_model=20, num_layer=6, heads=4, device='cpu', ff_expansion=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder = encoder(d_model, num_layer, heads, device, ff_expansion, dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

    def actual_token(self, masked_spots, n):
        actual = []
        for i in range(n):
            trg = token_id(input_seqs)
            actual_tokens = trg[i, masked_spots[i]]
            for token in actual_tokens:
                actual.append(token)
        return torch.Tensor(actual)

    def forward(self, input_seqs, segment_pos, mask):
        n = len(input_seqs)
        masking = token_masking(input_seqs)
        masked_seq = masking[2]
        masked_spots = masking[0]
        y = self.actual_token(masked_spots, n)
        out = self.encoder(masked_seq, segment_pos, mask)
        y_pred = out[0, masked_spots[0]]
        for i in range(1, n):
            y_pred= torch.cat((y_pred,out[i, masked_spots[i]]), 0)
        loss = self.loss(y_pred, y.long())
        return loss

pre_train = bert()
out = pre_train(input_seqs, {0:20, 20:38, 38:58}, None)
out.backward()


'''def token_masking(input_seqs):
    n, seq_lenth = len(input_seqs), len(input_seqs[0])

    indices_all_seqs = []
    for seq in input_seqs:
        indices_seq = []
        for aa in list(seq):
            indices_seq.append(look_up.get(aa))
        indices_all_seqs.append(indices_seq)
    indices_all_seqs = np.array(indices_all_seqs)

    mask = []
    for i in range(n):
        mask.append(np.random.random(seq_lenth) < 0.15)
    mask = torch.from_numpy(np.array(mask))

    masked_spots = []
    for i in range(n):
        masked_spots.append(torch.flatten(mask[i].nonzero()).tolist())
    for i in range(n):
        indices_all_seqs[i, masked_spots[i]] = 0

    sequences = []
    for indices in indices_all_seqs:
        sequence = ''
        for index in indices:
            sequence = sequence + get_key(index)
        sequences.append(sequence)
    return sequences, masked_spots'''