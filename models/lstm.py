import torch
from torch import nn
import torch.nn.functional as F
import os

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=True)
        self.W_out = nn.Linear(hidden_dim,1)

    def forward(self, batch):
        """
        :param batch of size [b @ (seq x dim)]
        :return: array of size [b]
        """
        lengths = (batch!=0).sum(1)[:,0] # lengths of non-padded items
        lstm_outs, _ = self.lstm(batch) # [b x seq x dim]
        out = torch.stack([lstm_outs[i,idx-1] for i,idx in enumerate(lengths)])
        out = self.W_out(out)
        # print(out.size())
        out = out.squeeze(-1)
        # print(out.size())
        out = torch.sigmoid(out)
        return out

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(BiLSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.lstm_f = nn.LSTM(embedding_dim, hidden_dim,batch_first=True)
        self.lstm_b = nn.LSTM(embedding_dim, hidden_dim,batch_first=True)
        self.W_out = nn.Linear(hidden_dim*2,1)

    def forward(self, batch_f, batch_b):
        """
        :param batch of size [b @ (seq x dim)]
        :return: array of size [b]
        """
        lengths = (batch_f!=0).sum(1)[:,0] # lengths of non-padded items
        lstm_outs, _ = self.lstm_f(batch_f) # [b x seq x dim]
        lstm_outs2, _ = self.lstm_b(batch_b) # [b x seq x dim]
        out = torch.stack([torch.cat([lstm_outs[i,idx-1],lstm_outs2[i,idx-1]]) for i,idx in enumerate(lengths)])
        out = self.W_out(out).squeeze()
        # out = torch.sigmoid(out).squeeze()
        out = torch.sigmoid(out)
        return out