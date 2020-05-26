import sys
import os
import time
import argparse
import itertools
import math
import pprint
import numpy
import random

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import torchtext
import torchtext.data as data 
import torchtext.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import load_data

from rnnlm import RNNLM
from sentence_vae import SentenceVAE

from train import *

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--embed_dim', type=int, default=300, help="The amount of dimensions of the word embeddings.")
    parser.add_argument('--hidden_dim', type=int, default=512, help="The amount of hidden dimensions.")
   
    parser.add_argument('--freebits_lambda', type=float, default=0, help="")
    parser.add_argument('--wdropout_prob', type=float, default=1, help="")
    parser.add_argument('--mu_forcing_beta', type=float, default=0, help="")

    # Training params
    parser.add_argument('--use_bptt', type=bool, default=False, help='')
    parser.add_argument('--seq_len', type=int, default=int(25), help='The length of the sequences to train on.')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples to process in a batch')
    parser.add_argument('--device', type=str, default="cuda", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.95, help='Learning rate decay fraction')
    parser.add_argument('--train_steps', type=int, default=int(20000), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='Gradient clipping maximum norm.')

    # Misc params
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--sw_log_dir', type=str, default='finalruns', help='The directory in which to create the default logdir.')
    parser.add_argument('--save_file', type=str, default='model.pt', help='Filename under which to store the model.')

    config = parser.parse_args()

    def set_seed(seed=42):
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
    set_seed()
    
    vocab = torchtext.vocab.FastText()
    
    _, (_,_,test,field) = load_data(embeddings=vocab) 
    test_set = test.dataset 
    embedding = nn.Embedding.from_pretrained(field.vocab.vectors)

    sentence_1a = field.process([test_set[1981].text])
    sentence_1b = field.process([test_set[2119].text])
    sentence_2a = field.process([test_set[1417].text])
    sentence_2b = field.process([test_set[1977].text])
    sentence_3  = field.process([test_set[672].text])
    sentence_4  = field.process([test_set[625].text])
    sentence_5  = field.process([test_set[555].text])

    
    def train_model(config):
        # summarywriter 
        logdir = logloc(dir_name=config.sw_log_dir, comment="")
        sw = SummaryWriter(log_dir=logdir)

        # Train and save the model
        best_model, _, metrics = train(config, sw)
        best_model = best_model.cpu()
           
        return best_model

    def experiment(best_model):
        mu_1a, s_1a = best_model.encoder(embedding(sentence_1a[0]), sentence_1a[1])
        mu_1b, s_1b = best_model.encoder(embedding(sentence_1b[0]), sentence_1b[1])
        mu_2a, s_2a = best_model.encoder(embedding(sentence_2a[0]), sentence_2a[1])
        mu_2b, s_2b = best_model.encoder(embedding(sentence_2b[0]), sentence_2b[1])

        def print_sample(sample):
            text = ' '.join(field.vocab.itos[w] for w in sample)
            print(f'{text}')

        def interpolate(z_a, z_b, steps=6):
            vectors = []
            for i in range(steps):
                z = z_a + (i/steps)*(z_b - z_a)
                vectors.append(z)
            vectors.append(z_b)
           
            for i,z in enumerate(vectors):
                sampled = best_model.decoder.greedy_sample(embedding, 50, h = z)
                print_sample(sampled)
            print('------')
                       
        interpolate(mu_1a, mu_1b)
        interpolate(mu_2a, mu_2b)

        print("#############################")

        mu_3, s_3 = best_model.encoder(embedding(sentence_3[0]), sentence_3[1])
        mu_4, s_4 = best_model.encoder(embedding(sentence_4[0]), sentence_4[1])
        mu_5, s_5 = best_model.encoder(embedding(sentence_5[0]), sentence_5[1])

        def replicate(mu, sigma, sample = best_model.decoder.greedy_sample):
            sampled = sample(embedding, 50, h = mu)
            print_sample(sampled)
            print('-')
            p_z = dist.Normal(loc=mu, scale=sigma)
            for _ in range(3):
                z = p_z.sample()
                sampled = sample(embedding, 50, h = z)
                print_sample(sampled)
            print('------------')

        replicate(mu_3, s_3)
        replicate(mu_4, s_4)
        replicate(mu_5, s_5)

        sample = best_model.decoder.temperature_sample
        replicate(mu_3, s_3, sample = sample)
        replicate(mu_4, s_4, sample = sample)
        replicate(mu_5, s_5, sample = sample)

        print("#############################")
         
    
    # Model 2 (vanilla)
    config.model = 's-vae'
    model = train_model(config)
    experiment(model)

    # Model 3 (best PP)
    config.wdropout_prob = 0.55
    config.mu_forcing_beta = 1
    model = train_model(config)
    experiment(model)

    # Model 4 (best NLL)
    config.mu_forcing_beta = 25
    model = train_model(config)
    experiment(model)

    # Model 1
    config.model = 'rnnlm'
    train_model(config)

