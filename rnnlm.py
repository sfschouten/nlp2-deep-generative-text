import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNLM(nn.Module):

    NUM_LAYERS = 1

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0):
        super(RNNLM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gru = nn.GRU(
                input_size = input_dim,         # E
                hidden_size = hidden_dim,       # H
                num_layers = self.NUM_LAYERS,
                dropout = dropout
            )

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sm = nn.LogSoftmax(dim=2)


    def forward(self, x, h = None):
        """
        Args:
            x : vectors that represent the sentence     S x B x E

        Returns:
            
        """
        S, B, E = x.shape

        out, _ = self.gru(x, h)     # S x B x H
        y = self.fc(out)            # S x B x C
        y = self.sm(y)

        return y


    def feed_back_forward(self, x, embedding, max_len, h = None, eos_idx = 2):
        """
        Forward the module, while feeding the output at each timestep back
        in as input for the following timestep.
        Args:
            x : the first input.
            embedding : the embedding used for the input.
            max_len : the maximum length of the output. 
            h : the initial hidden state. Default: zero vector.
            eos_idx : the index of the end-of-sequence token in the embedding. Default: 2.
        """

        # copy weights of GRU layer to a GRUCell module
        if not self.gru_cell:
            self.gru_cell = nn.GRUCell(
                    input_size = self.input_dim,
                    hidden_size = self.hidden_dim
                )
            self.gru_cell.weight_ih = self.gru.weight_ih_l[0]
            self.gru_cell.weight_hh = self.gru.weight_hh_l[0]
            self.gru_cell.bias_ih = self.gru.bias_ih_l[0]
            self.gru_cell.bias_hh = self.gru.bias_hh_l[0]

        
        B, E = x.shape
        device = x.device

        if not h:
            h = torch.zeros((B, self.hidden_dim), device = device)

        cntr = 0
        outputs = []
        while i != eos_idx and cntr <= max_len:
            h = self.gru_cell(x, h)
            outputs.append(h)

            y = self.fc(h)
            i = self.sm(y).argmax()
            x = embedding(i)

            cntr += 1


        outputs = torch.cat(outputs)
        return h, outputs



