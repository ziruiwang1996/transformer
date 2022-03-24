import torch.nn as nn
import torch
from transformer import Encoder, Decoder
from functions import get_key
import torch.optim as optim
from functions import get_seq_label
import csv

class Generative_Model(nn.Module):
    def __init__(self, d_model=22, heads=2, dropout=0.1, ff_expansion=4, device='cpu'):
        super().__init__()
        self.device = device
        self.pre_train = Encoder(d_model, heads, dropout, ff_expansion, num_layer=4, seq_len=59, device='cpu')
        self.decoder = Decoder(d_model, heads, dropout, ff_expansion, num_layer=6, seq_len=59, device='cpu', trg_vocab_size=22)

    def make_trg_mask(self, trg_seq):
        n, trg_len = len(trg_seq), len(trg_seq[0])
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(n, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, dec_input, src_mask):
        self.pre_train.load_state_dict(torch.load('trained_model.pth'))
        self.pre_train.eval()
        enc_out = self.pre_train(dec_input, None)
        trg_mask = self.make_trg_mask(dec_input)
        out = self.decoder(dec_input, enc_out, src_mask, trg_mask)
        return out

if __name__ == "__main__":
    # import data
    file = csv.reader(open('All_affibody_greater_10.csv'))
    raw_data = []
    for row in file:
        raw_data.append(('+'+row[1], row[1]+'#'))
    raw_data = raw_data[1:]
    #print(raw_data)

    model = Generative_Model()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # make a iterable dataset
    dataset = []
    for seq_pair in raw_data:
        dataset.append([seq_pair[0], seq_pair[1]])
    batch_size = 30
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(30):  # loop over the dataset
        running_loss = 0.0
        for i, data in enumerate(trainloader, start=0):
            sequence, label = data

            optimizer.zero_grad()
            out = model(sequence, src_mask=None)
            # torch.save(out, 'attention_figure.pt')
            out = torch.transpose(out, 1, 2)
            # print(out.shape, get_seq_label(label).shape)
            loss = criterion(out, get_seq_label(label).long())
            loss.backward()
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    def generation():
        model = Generative_Model()
        out = model('+', src_mask=None)
        # print(out)
        all_aas = []
        for seq in out:
            tokens = torch.argmax(seq, dim=1)
            aas = []
            for token in tokens:
                aas.append(get_key(int(token)))
            aas_str = ''.join(aas)
            all_aas.append(aas_str)
        return all_aas
    print(generation())