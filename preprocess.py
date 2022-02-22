import torch
import numpy as np

# look up table
one_hots = np.zeros((20, 20))
np.fill_diagonal(one_hots, 1)
tokens = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
order = list(range(len(tokens)))
look_up = dict(zip(tokens, order))
look_up['-'] = 20
#print(look_up)

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

def token_masking(input_seqs):
    n = len(input_seqs)  # sample size
    seq_len = len(input_seqs[0])  # sequence length

    # generating random numbers from 0-1
    torch.manual_seed(100)
    mask = torch.rand((n, seq_len)) < 0.15

    masked_spots = []
    indices_all_seqs = tokenizer(input_seqs)
    for i in range(n):
        masked_spots.append(torch.flatten(mask[i].nonzero()).tolist())
    for i in range(n):
        indices_all_seqs[i, masked_spots[i]] = 20
    return masked_spots, indices_all_seqs

def masked_seq(indices_all_seqs):
    sequences = []
    for indices in indices_all_seqs:
        sequence = ''
        for index in indices:
            sequence = sequence + get_key(index)
        sequences.append(sequence)
    return sequences

def get_label(input_seqs, masked_spots):
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