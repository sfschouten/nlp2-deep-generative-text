import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class RNNLM(nn.Module):

    NUM_LAYERS = 1

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RNNLM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gru = nn.GRU(
            input_size = input_dim,         # E
            hidden_size = hidden_dim,       # H
            num_layers = self.NUM_LAYERS
        )

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sm = nn.LogSoftmax(dim=2)


    def forward(self, x, l, h = None):
        """
        Args:
            x : vectors that represent the sentence     S x B x E

        Returns:
            
        """
    
        packed = nn.utils.rnn.pack_padded_sequence(x, l, enforce_sorted=False)

        out, _ = self.gru(packed, h)     # S x B x H

        out, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=x.shape[0])

        y = self.fc(out)            # S x B x C
        y = self.sm(y)
        return y


    def get_additional_losses(self):
        return {}


    def _sample(self, sample_f, embedding, max_len, x, h, eos_idx):
        """
        Forward the module, while feeding the output at each timestep back
        in as input for the following timestep.
        Args:
            x : the index of the first input. (int)
            embedding : the embedding used for the input.
            max_len : the maximum length of the output. 
            h : the initial hidden state. Default: zero vector.
            eos_idx : the index of the end-of-sequence token in the embedding. Default: 2.
        """

        # copy weights of GRU layer to a GRUCell module
        if not hasattr(self, 'gru_cell'):
            self.gru_cell = nn.GRUCell(
                    input_size = self.input_dim,
                    hidden_size = self.hidden_dim
                )
            self.gru_cell.weight_ih = self.gru._parameters['weight_ih_l0'] 
            self.gru_cell.weight_hh = self.gru._parameters['weight_hh_l0']
            self.gru_cell.bias_ih = self.gru._parameters['bias_ih_l0'] 
            self.gru_cell.bias_hh = self.gru._parameters['bias_hh_l0'] 

        
        device = self.fc.weight.device 

        if not x:
            x = eos_idx 
  
        x = torch.LongTensor([x]).to(device)
        x = embedding(x)

        if h == None:
            h = torch.zeros((1, self.hidden_dim), device = device)

        cntr = 0
        i = torch.LongTensor([-1]).to(device).unsqueeze(0)
        outputs = []
        while i.item() != eos_idx and cntr <= max_len:
            h = self.gru_cell(x, h)
            y = self.fc(h)
            i = sample_f(y)
            outputs.append(i.item())
            x = embedding(i)
            cntr += 1

        return outputs


    def greedy_sample(self, embedding, max_len, x = None, h = None, eos_idx = 2):
        sample_f = lambda x : x.softmax(dim=1).argmax(dim=1)
        return self._sample(sample_f, embedding, max_len, x, h, eos_idx)


    def temperature_sample(self, embedding, max_len, temperature=2, x = None, h = None, eos_idx = 2):
        def sample_f(x):
            device = x.device
            probs = (x*temperature).softmax(dim=1).squeeze().cpu().detach().numpy()
            i = np.random.choice(a = range(len(embedding.weight)), p = probs)
            return torch.LongTensor([i]).to(device)
        
        return self._sample(sample_f, embedding, max_len, x, h, eos_idx)

    def multi_sample_estimates(self, x, l, t, K=4):
        """

        """
        with torch.no_grad():
            S, B, E = x.shape
    
            # repeat K times and merge with batch dimension
            x = x.unsqueeze(2).expand(-1, -1, K, -1)        # S x B x K x E
            l = l.unsqueeze(1).expand(-1, K)                # B x K
            t = t.unsqueeze(2).expand(-1, -1, K)            # S x B x K
            x = x.reshape(S, B*K, E)
            l = l.reshape(B*K)
            t = t.reshape(S, B*K)

            # encode
            y = self(x, l)
            V = y.shape[2] # vocabulary

            # calculate NLL
            y = y.view(-1, V)                               # S*B*K x V
            t = t.reshape(-1)                               # S*B*K
            nll = F.nll_loss(y, t, reduction='none')        # S*B*K
            nll = nll.view(S, B, K).sum(dim=0).mean(dim=1)

        return nll, nll, torch.FloatTensor([0])


            


