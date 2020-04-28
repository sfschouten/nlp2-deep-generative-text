import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNLM(nn.Module):

    NUM_LAYERS = 1
    NUM_CLASSES = 3

    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(RNNLM, self).__init__()

        self.gru = nn.GRU(
                input_size = input_dim,         # E
                hidden_size = hidden_dim,       # H
                num_layers = self.NUM_LAYERS,
                dropout = dropout
            )

        self.fc = nn.Linear(hidden_dim, self.NUM_CLASSES)


    def forward(self, sentences, lengths):
        """
        Args:
            x : vectors that represent the sentence     S x B x E

        Returns:
            
        """

        B, S, D = sentences.shape

        packed = nn.utils.rnn.pack_padded_sequence(
            sentences, 
            lengths
            #enforce_sorted=False
        )

        _, h_n = self.gru(packed)
        
        y = self.fc(h_n)    

        return F.softmax(y)
