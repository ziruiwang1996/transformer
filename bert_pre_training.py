import numpy as np
import csv
from transformer import *
from preprocess import *
import torch.optim as optim

#import data
file = csv.reader(open('All_affibody_greater_10.csv'))
input_seqs = []
for row in file:
    input_seqs.append(row[1])

input_seqs = input_seqs[1:20]
#print(input_seqs)


class Bert(nn.Module):
    def __init__(self, d_model=20, num_layer=6, heads=4, device='cpu', ff_expansion=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder = encoder(d_model, num_layer, heads, device, ff_expansion, dropout)

    def forward(self, input_seqs, segment_pos, mask):
        out = self.encoder(input_seqs, segment_pos, mask)
        return out

pre_train = Bert()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(pre_train.parameters(), lr=0.001, momentum=0.9)

masked_spots, indices_all_seqs = token_masking(input_seqs)

input = masked_seq(indices_all_seqs)
label = labels(input_seqs, masked_spots)
print(len(input), len(label))

trainloader = torch.utils.data.DataLoader(input, batch_size=2, shuffle=True)



for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, start=0):

        label = labels(data, masked_spots).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        output = pre_train(data, {0: 20, 20: 38, 38: 58}, None)

        # get predicted masked tokens
        out_pred = output[0, masked_spots[0]]
        for n in range(2):
            out_pred = torch.cat((out_pred, output[n, masked_spots[n]]), 0)

        loss = criterion(out_pred, label.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

#print('Finished Training')

'''for epoch in range(3):
    running_loss = 0.0
    output = pre_train(input, {0:20, 20:38, 38:58}, None)
    out_pred = output[0, masked_spots[0]]
    n = len(input_seqs)
    for i in range(1, n):
        out_pred = torch.cat((out_pred, output[i, masked_spots[i]]), 0)
    loss = criterion(out_pred, label)
    loss.backward()
    # print statistics
    running_loss += loss.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss))'''
