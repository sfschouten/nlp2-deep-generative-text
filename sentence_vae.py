import torch
import torch.nn as nn

from rnnlm import RNNLM
import torch.distributions as dist


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

        _, h_n = self.gru(x)        # 2 x B x H

        _, B, H = h_n.shape
        h_n = h_n.permute(1,0,2)    # B x 2 x H
        h_n = h_n.reshape(B,-1)     # B x 2H

        mu = self.mu(h_n)           # B x L
        sigma = self.sigma(h_n)     # B x L

        return mu, sigma 



class SentenceVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, 
            fb_lambda = 0.5, wd_keep_prob=1, wd_unk=None, mu_f_beta=3):
        """
        Args:
            input_dim:
            hidden_dim:
            num_classes:

            fb_lambda: the freebits lambda parameter that acts as a minimum
                on the KL term.
            wd_keep_prob: the probability of any word in the decoder's input
                being kept.
            wd_unk_i: the index of the unkown token in the embedding.
            mu_f_beta: the margin for the mu-forcing.
        """
        super(SentenceVAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.prior = None

        self.fb_lambda = fb_lambda
        self.mu_f_beta = mu_f_beta

        self.wd_keep_prob = wd_keep_prob
        if wd_keep_prob < 1:
            self.wd_unk = wd_unk.unsqueeze(dim=0)

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


    def calc_regularization_loss(self, q_z):

        # construct a standard normal as prior
        if not self.prior:
            p_mu = torch.zeros_like(q_z.mean)
            p_sigma = torch.ones_like(q_z.stddev)
            self.prior = dist.Normal(loc=p_mu, scale=p_sigma)

        kl = dist.kl.kl_divergence(q_z, self.prior)
        kl = kl.mean(dim=0)

        # FREEBITS 
        self.fb_loss = (self.fb_lambda - kl).clamp(min = 0).sum()
    
        self.kl_loss = kl.sum()


    def get_additional_losses(self):
        return {
            'kl-divergence' : self.kl_loss,
            'fb-loss' : self.fb_loss,
            'mu-loss' : self.mu_loss
        } 


    def forward(self, x):
        """
        Args:
            x : vectors that represent the sentence     S x B x E

        Returns:
            
        """
        S, B, E = x.shape
        device = x.device

        # Use the encoder to obtain mean and standard deviation.
        mu, sigma = self.encoder(x)                     # B x H, B x H

        # obtain the normal distribution and sample from it 
        q_z = dist.Normal(loc=mu, scale=sigma)
        z = q_z.rsample().unsqueeze(dim=0)

        # store the regularization loss
        self.calc_regularization_loss(q_z)

        # mu-FORCING
        mu_term = torch.bmm(mu.view(B, 1, -1), mu.view(B, -1, 1)).sum() / (2*B)
        self.mu_loss = (self.mu_f_beta - mu_term).clamp(min=0)

        # WORD DROPOUT
        if self.wd_keep_prob < 1 and self.training:
            # sample words to drop/keep
            prob = torch.Tensor(S, B).to(device)        # S x B
            prob.fill_(self.wd_keep_prob)
            drop_d = dist.bernoulli.Bernoulli(prob) 
            drops = drop_d.sample().unsqueeze(dim=2)    # S x B x 1
            
            # get unkowns to replace with
            unks = self.wd_unk.expand(S, B, -1)         # S x B x E
            
            # apply
            x = drops * x + (1-drops) * unks 

        output = self.decoder(x, h = z)
        return output 


    def perplexity(self, x, t):
        

        S, B, E = x.shape
        device = x.device

        mu, sigma = self.encoder(x)
        mu = mu.unsqueeze(1).expand(-1, K, -1)          # B x K x H
        sigma = sigma.unsqueeze(1).expand(-1, K, -1)    # B x K x H
      
        q_z = dist.Normal(loc=mu, scale=sigma)
        z = q_z.sample().unsqueeze(dim=0)               # 1 x B x K x H

        x = x.unsqueeze(2).expand(-1, -1, K, -1)        # S x B x K x E
        t = t.unsqueeze(2).expand(-1, -1, K)            # S x B x K

        y = self.decoder(x, h = z)                      # S x B x K x V

        y = y.view(-1, y.shape[3])                      # S*B*K x V
        t = t.view(-1)                                  # S*B*K
        nll = F.cross_entropy(y, t, reduction='none')   # S*B*K
        ll = -nll.view(S, B, K).sum(dim=0)              # B x K

        p_mu = torch.zeros_like(mu)
        p_sigma = torch.ones_like(sigma)
        prior = dist.Normal(loc=p_mu, scale=p_sigma)

        z = z.squeeze()
        p_z = prior.log_prob(z).sum(dim=2)              # B x K
        log_joint = ll + p_z
        
        q = q_z.log_prob(z).sum(2)                      # B x K
        marginal = (log_joint - q).exp().sum(dim=1)     # B
        n_log_marg = -marginal.log()





        

