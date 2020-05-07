import torch
import torch.nn as nn


class SentenceVAEEncoder(nn.Module):

    NUM_LAYERS = 1
    BIDIRECTIONAL = True

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0):
        super(SentenceVAEEncoder, self).__init__()

        self.gru = nn.GRU(
                input_size = input_dim,         # E
                hidden_size = hidden_dim,       # H
                num_layers = self.NUM_LAYERS,
                dropout = dropout,
                bidirectional = self.BIDIRECTIONAL
            )

        self.mu = nn.Linear(2*hidden_dim, latent_dim)

        self.sigma = nn.Sequential(
                nn.Linear(2*hidden_dim, latent_dim),
                nn.Softplus()
            )


    def forward(self, x):
        """
        Args:
            x : vectors that represent the sentence     S x B x E

        Returns:
            
        """
        S, B, E = x.shape

        _, h_n = self.gru(x)        # 2 x B x H
        h_n = h_n.permute(1,0,2)    # B x 2 x H
        h_n = h_n.reshape(B,-1)        # B x 2H

        mu = self.mu(h_n)           # B x L
        sigma = self.sigma(h_n)     # B x L

        return mu, sigma 


from rnnlm import RNNLM
import torch.distributions as dist

class SentenceVAE(nn.Module):


    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SentenceVAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.prior = None

        self.encoder = SentenceVAEEncoder(
                input_dim,          # E
                hidden_dim,         # H
                hidden_dim          # H
            )

        self.decoder = RNNLM(
                input_dim,          # E
                hidden_dim,         # H
                num_classes         # C
            )

    def forward(self, x):
        """
        Args:
            x : vectors that represent the sentence     S x B x E

        Returns:
            
        """
        S, B, E = x.shape
        device = x.device

        # Use the encoder to obtain mean and standard deviation.
        mu, sigma = self.encoder(x)         # B x H, B x H

        # obtain the normal distribution and sample from it 
        q_z = torch.distributions.Normal(loc=mu, scale=sigma)
        z = q_z.rsample().unsqueeze(dim=0)
        
        # construct a standard normal as prior
        if not self.prior:
            p_mu = torch.zeros_like(mu)
            p_sigma = torch.ones_like(sigma)
            self.prior = dist.Normal(loc=p_mu, scale=p_sigma)

        # store the regularization loss
        self.additional_loss = dist.kl.kl_divergence(q_z, self.prior).sum()

        output = self.decoder(x, h = z)
        return output 



