import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from scipy.special import logsumexp as LSE

from rnnlm import RNNLM


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


    def forward(self, x, l):
        """
        Args:
            x : vectors that represent the sentence     S x B x E
            l : the lengths of those sentences.
        Returns:
            The mean and stdev to parameterize our standard Gaussian with. 
        """

        packed = nn.utils.rnn.pack_padded_sequence(x, l, enforce_sorted=False)

        _, h_n = self.gru(packed)   # 2 x B x H

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

    def _sample_prior(self, device):
        p_mu = torch.zeros((1, self.hidden_dim)).to(device)
        p_sigma = torch.ones((1, self.hidden_dim)).to(device)
        prior = dist.Normal(loc=p_mu, scale=p_sigma)
        return prior.sample()

    def greedy_sample(self, embedding, max_len, x = None, eos_idx = 2):
        h = self._sample_prior(embedding.weight.device)
        return self.decoder.greedy_sample(embedding, max_len, x, h, eos_idx) 

    def temperature_sample(self, embedding, max_len, temperature = 2, x = None, eos_idx = 2):
        h = self._sample_prior(embedding.weight.device)
        return self.decoder.temperature_sample(embedding, max_len, temperature, x, h, eos_idx) 
        

    def calc_regularization_loss(self, q_z):

        # construct a standard normal as prior
        
        p_mu = torch.zeros_like(q_z.mean)
        p_sigma = torch.ones_like(q_z.stddev)
        prior = dist.Normal(loc=p_mu, scale=p_sigma)

        kl = dist.kl.kl_divergence(q_z, prior)
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


    def forward(self, x, l):
        """
        Args:
            x : vectors that represent the sentence     S x B x E
            l : the lengths of the sentences
        Returns:
            
        """
        S, B, E = x.shape
        device = x.device

        # Use the encoder to obtain mean and standard deviation.
        mu, sigma = self.encoder(x, l)                  # B x H, B x H

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

        output = self.decoder(x, l, h = z)
        return output 


    def multi_sample_estimates(self, x, l, t, K=4):
        """
        Returns:
        multi-sample estimates of:
            the negative marginal log-likelihood
            the negative conditional log-likelihood
            the kl-divergence between prior and posterior
        """
        with torch.no_grad():
            S, B, E = x.shape
            device = x.device
    
            # encode
            mu, sigma = self.encoder(x, l)

            mu = mu.unsqueeze(1).expand(-1, K, -1)          # B x K x H
            sigma = sigma.unsqueeze(1).expand(-1, K, -1)    # B x K x H
        
            # get distributions
            q = dist.Normal(loc=mu, scale=sigma)
            z = q.sample()                                  # B x K x H
            
            p_mu = torch.zeros_like(mu)
            p_sigma = torch.ones_like(sigma)
            prior = dist.Normal(loc=p_mu, scale=p_sigma)
           
            # get prior and posterior probabilities of z
            log_p_z = prior.log_prob(z).sum(dim=2)          # B x K
            log_q_z = q.log_prob(z).sum(2)                  # B x K

            # repeat K times and merge with batch dimension
            x = x.unsqueeze(2).expand(-1, -1, K, -1)        # S x B x K x E
            l = l.unsqueeze(1).expand(-1, K)                # B x K
            t = t.unsqueeze(2).expand(-1, -1, K)            # S x B x K
            x = x.reshape(S, B*K, E)
            l = l.reshape(B*K)
            t = t.reshape(S, B*K)
            z = z.reshape(1, B*K, -1)

            # decode every sample in batch for each z_k
            y = self.decoder(x, l, h = z)                   # S x B*K x V
            V = y.shape[2] # vocabulary

            # calculate conditional NLL
            y = y.view(-1, V)                               # S*B*K x V
            t = t.reshape(-1)                               # S*B*K
            nll = F.nll_loss(y, t, reduction='none')        # S*B*K

            # log likelihood of sentences ( p(x|z_k) )
            ll = -nll.view(S, B, K).sum(dim=0)              # B x K
           
            # p(x,z) = p(x|z) p(z)
            log_joint = ll + log_p_z
            ll = ll.mean(dim=1)

            # add -log(K) to get logmeanexp 
            log_marginal = torch.logsumexp(log_joint - log_q_z - math.log(K), 1)    # B

            kl = dist.kl.kl_divergence(q, prior)
            kl = kl.mean(dim=1).sum(dim=1) 

            return -log_marginal, -ll, kl 


            

