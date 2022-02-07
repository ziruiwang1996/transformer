import numpy as np

# d_model = 21
def onehot_emb (sequnece_string, d_model):
    tokens = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","-"]
    order = np.arange(len(tokens))
    look_up = dict(zip(tokens, order))
    #dimension: seq_len * d_model
    emb = np.zeros((len(sequnece_string), d_model))
    n =0
    for aa in list(sequnece_string):
        emb[n][look_up.get(aa)]=1
        n+=1
    return emb

def seg_emb (list_of_tuples, seq_len, d_model):
    seg_id = np.arange(len(list_of_tuples))
    emb = np.zeros((seq_len, d_model))
    n= 0
    for tuple in list_of_tuples:
        emb[tuple[0]:tuple[1],:]=seg_id[n]
        n+=1
    return emb

def pos_emb(seq_len, d_model, min_freq=1e-4):
    pos = np.arange(seq_len)
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    emb = pos.reshape(-1,1)*freqs.reshape(1,-1)
    emb[:, ::2] = np.cos(emb[:, ::2])
    emb[:, 1::2] = np.sin(emb[:, 1::2])
    return emb
