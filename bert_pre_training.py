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

input_seqs = input_seqs[1:61]
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
masked_seqs = masked_seq(indices_all_seqs)
labels_all = get_label(input_seqs, masked_spots)

# make a iterable dataset
dataset = []
n = 0
for seq in masked_seqs:
    dataset.append([seq, n])
    n += 1

batch_size = 6
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

for epoch in range(3):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, start=0):
        sequence, index = data

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = pre_train(sequence, {0: 20, 20: 38, 38: 58}, None)

        # get predicted masked tokens
        out_pred = outputs[0, masked_spots[index[0]]]
        for n in range(1,batch_size):
            out_pred = torch.cat((out_pred, outputs[n, masked_spots[index[n]]]), 0)

        # get masked token labels
        labels_per_batch = []
        for n in range(batch_size):
            labels_per_batch = labels_per_batch + labels_all[index[n]]
        labels_per_batch = torch.Tensor(labels_per_batch)
        # dimension check
        #print(out_pred.shape, labels_per_batch.shape)

        loss = criterion(out_pred, labels_per_batch.long())
        loss.backward()
        optimizer.step()

        # print loss statistics
        running_loss += loss.item()
        if i % 3 == 2:  # print every 3 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 3))
            running_loss = 0.0

print('Finished Training')