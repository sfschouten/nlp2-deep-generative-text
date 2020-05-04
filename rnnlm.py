import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNLM(nn.Module):

    NUM_LAYERS = 1

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0):
        super(RNNLM, self).__init__()

        self.gru = nn.GRU(
                input_size = input_dim,         # E
                hidden_size = hidden_dim,       # H
                num_layers = self.NUM_LAYERS,
                dropout = dropout
            )

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sm = nn.LogSoftmax(dim=2)

    def forward(self, x):
        """
        Args:
            x : vectors that represent the sentence     S x B x E

        Returns:
            
        """
        S, B, E = x.shape

        out, _ = self.gru(x)    # S x B x H
        y = self.fc(out)        # S x B x C
        y = self.sm(y)
        return y 
