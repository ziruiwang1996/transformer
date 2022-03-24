import torch
import numpy as np
import random


# look up table
one_hots = np.zeros((20, 20))
np.fill_diagonal(one_hots, 1)
# special token: -: mask, +: start, #: end
tokens = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "+", "#", "-"]
order = list(range(len(tokens)))
look_up = dict(zip(tokens, order))
#print(look_up)

def rand_aa_index():
    return random.randint(0, 19)

def get_key(val):
    for key, value in look_up.items():
        if val == value:
            return key
    return "key doesn't exist"

def tokenizer(input_seqs):
    indices_all_seqs = []
    for seq in input_seqs:
        indices_seq = []
        for aa in list(seq):
            indices_seq.append(look_up.get(aa))
        indices_all_seqs.append(indices_seq)
    return np.array(indices_all_seqs)

def text_corrupting(input_seqs):
    n = len(input_seqs)  # sample size
    seq_len = len(input_seqs[0])  # sequence length
    # generating random numbers from 0-1
    # 0     - 0.12  --> <mask> 80%
    # 0.12  - 0.135 --> random change 10%
    # 0.135 - 0.15  --> unchange 10%
    torch.manual_seed(100)
    a = torch.rand((n, seq_len))
    corrupted_spots = a < 0.15
    msk = a < 0.12
    rc = (a < 0.135 ) & (a >= 0.12)
    # uc = (a < 0.15 ) & (a >= 0.135) no action needed for unchange
    msk_spots = []
    rc_spots = []
    indices_all_seqs = tokenizer(input_seqs)
    for i in range(n):
        msk_spots.append(torch.flatten(msk[i].nonzero()).tolist())
        rc_spots.append(torch.flatten(rc[i].nonzero()).tolist())
    for i in range(n):
        indices_all_seqs[i, msk_spots[i]] = 22  # replace with <mask>
        for j in range(len(rc_spots[i])):
            indices_all_seqs[i, rc_spots[i][j]] = rand_aa_index()  # replace with random aa one by one
    return corrupted_spots, indices_all_seqs

def adjustment(corrupted_spots):
    n = len(corrupted_spots)
    a = torch.zeros((1, n)).bool()
    a_t = torch.transpose(a, 0, 1)
    b = torch.cat((a_t, corrupted_spots), 1)
    return b

def corrupted_seq(indices_all_seqs):
    sequences = []
    for indices in indices_all_seqs:
        sequence = '+'
        for index in indices:
            sequence = sequence + get_key(index)
        sequences.append(sequence)
    return sequences

def get_mask_label(input_seqs, masked_spots):
    n = len(input_seqs)  # sample size
    labels_all = []
    seq_tokens = tokenizer(input_seqs)
    for i in range(n):
        labels = seq_tokens[i, masked_spots[i]]
        labels_per_seq = []
        for label in labels:
            labels_per_seq.append(label)
        labels_all.append(labels_per_seq)
    return labels_all

def get_seq_label(trgs):
    indexes = []
    for seq in trgs:
        seq_index = []
        for token in seq:
            seq_index.append(look_up.get(token))
        indexes.append(seq_index)
    return torch.Tensor(indexes)