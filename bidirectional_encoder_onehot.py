import csv
import torch.nn as nn
from transformer import Encoder
from functions import *
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # import data
    file = csv.reader(open('All_affibody_greater_10.csv'))
    input_seqs = []
    for row in file:
        input_seqs.append(row[1])
    input_seqs = input_seqs[1:4019]
    # print(input_seqs)

    pretrain_model = Encoder(d_model=22, heads=2, dropout=0.1, ff_expansion=4, num_layer=4, seq_len=59, device='cpu')
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(pre_train.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(pre_train.parameters(), lr=0.001)
    optimizer = optim.Adadelta(pretrain_model.parameters())
    #optimizer = optim.Adagrad(pre_train.parameters(), lr=0.01)

    #checking number of parameters in model
    total_params = sum(p.numel() for p in pretrain_model.parameters())
    #print(total_params)

    corrupted_spots, indices_all_seqs = text_corrupting(input_seqs)
    corrupted_seqs = corrupted_seq(indices_all_seqs)
    labels_all = get_mask_label(input_seqs, corrupted_spots)
    adj_spots = adjustment(corrupted_spots)

    # make a iterable dataset
    dataset = []
    n = 0
    for seq in corrupted_seqs:
        dataset.append([seq, n])
        n += 1

    batch_size = 14
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    plot_x = []
    plot_y = []

    for epoch in range(20):  # loop over the dataset
        running_loss = 0.0
        for i, data in enumerate(trainloader, start=0):
            sequence, index = data

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = pretrain_model(enc_input=sequence, mask=None)
            # outputs.shape, (n, seq_len, d_model)

            # get predicted masked tokens
            out_pred = outputs[0, adj_spots[index[0]]]
            # out_pred.shape, (token, d_model)

            for n in range(1, batch_size):
                out_pred = torch.cat((out_pred, outputs[n, adj_spots[index[n]]]), 0)

            # get masked token labels
            labels_per_batch = []
            for n in range(batch_size):
                labels_per_batch = labels_per_batch + labels_all[index[n]]
            labels_per_batch = torch.Tensor(labels_per_batch)
            # dimension check
            # print(out_pred.shape, labels_per_batch.shape)

            loss = criterion(out_pred, labels_per_batch.long())
            loss.backward()
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if i % 40 == 39:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 40))
                plot_x.append('[%d, %5d]' % (epoch + 1, i + 1))
                plot_y.append(running_loss/40)
                running_loss = 0.0

    plt.plot(plot_x, plot_y)
    plt.xticks(rotation=90)
    plt.show()
    print('Finished Training')

    FILE = 'trained_model.pth'
    torch.save(pretrain_model.state_dict(), FILE)