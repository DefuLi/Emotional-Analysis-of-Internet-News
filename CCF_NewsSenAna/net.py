import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Net, self).__init__()

        # 单词=>[embedding_dim=300]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # [embedding_dim]=>[hidden_dim]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        # [hidden_dim*2]=>[3]
        self.fc = nn.Linear(hidden_dim * 2, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x:[seq_len,b,1]=>[seq_len,b,embedding_dim=300]
        embedding = self.dropout(self.embedding(x))

        # out: [seq_len, b, hid_dim*2]
        # h: [num_layers*2, b, hid_dim]=>[4,b,hidden_dim]
        # c: [num_layers*2, b, hid_dim]

        out, (h, c) = self.lstm(embedding)
        h = torch.cat([h[-2], h[-1]], dim=1)

        h = self.dropout(h)
        output = self.fc(h)

        return output
