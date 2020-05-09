import sys
import os
import time
import argparse
import itertools
import math
import pprint

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import torchtext.data as data 
import torchtext.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import load_data

from rnnlm import RNNLM
from sentence_vae import SentenceVAE


################################################################################

def save_model(label, model, config):
    name = "{}_{}e_{}h_{}_{}".format(
            config.model,
            config.embed_dim,
            config.hidden_dim,
            label,
            config.save_file
    )
    torch.save(model, name)

def test_model(model, embedding, criterion, valid_iter, device):
    with torch.no_grad():
        model.eval()
        nll = 0
        batch_acc = 0
        for step, batch in enumerate(valid_iter):

            batch_input = embedding(batch.text.to(device))
            batch_target = batch.target 

            batch_output = model(batch_input)

            # merge batch and sequence dimension for evaluation
            batch_output = batch_output.view(-1, batch_output.shape[2])
            batch_target = batch_target.view(-1)

            batch_acc += batch_output.argmax(dim=1).eq(batch_target).double().mean()
            nll += criterion(batch_output, batch_target).item()

    nll_per_sample = nll / (step * valid_iter.batch_size) 
    return batch_acc/(step + 1), nll_per_sample 


def train(config, sw):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    vocab = torchtext.vocab.FastText()
    #vocab = torchtext.vocab.GloVe()

    # get data iterators
    lm_iters, s_iters, vocab = load_data(
            embeddings=vocab, 
            device=device, 
            batch_size=config.batch_size,
            bptt_len=config.seq_len
        )
    #train_iter, _, _ = lm_iters
    #_, valid_iter, test_iter = s_iters
    train_iter, valid_iter, test_iter = lm_iters

    print("Vocab size: {}".format(vocab.vectors.shape))

    # create embedding layer
    embedding = nn.Embedding.from_pretrained(vocab.vectors).to(device)

    num_classes = vocab.vectors.shape[0]
    # Initialize the model that we are going to use
    if config.model == "rnnlm":
        model = RNNLM(
            config.embed_dim, 
            config.hidden_dim,
            num_classes
        )
    elif config.model == "s-vae":
        model = SentenceVAE(
            config.embed_dim,
            config.hidden_dim,
            num_classes,
            fb_lambda = config.freebits_lambda, 
            wd_keep_prob = config.wdropout_prob, 
            wd_unk = embedding(torch.LongTensor([vocab.stoi["<unk>"]]).to(device))
        )
    else: raise Error("Invalid model parameter.")
    model = model.to(device)

    # Setup the loss, optimizer, lr-scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) 
    criterion = torch.nn.NLLLoss(reduction="sum").to(config.device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.learning_rate_decay)
    lr = config.learning_rate
  
    global_step = 0
    best_acc = 0
    for epoch in itertools.count():
        for batch in train_iter:
            batch_text = embedding(batch.text.to(device))   # 
            batch_target = batch.target.to(device)
            batch_output = model(batch_text)

            B = batch_text.shape[1]
            # merge batch and sequence dimension for evaluation
            batch_output = batch_output.view(-1, batch_output.shape[2])
            batch_target = batch_target.view(-1)

            batch_acc = batch_output.argmax(dim=1).eq(batch_target).double().mean()
            loss = criterion(batch_output, batch_target) / B
            sw.add_scalar('Train/NLL', loss.item(), global_step)

            if hasattr(model, 'additional_loss'):
                kl = model.additional_loss
                loss += kl
                sw.add_scalar('Train/KL-divergence', kl, global_step)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            sw.add_scalar('Train/Loss', loss, global_step)
            sw.add_scalar('Train/Accuracy', batch_acc, global_step)

            if global_step % config.print_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, "
                        "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), global_step,
                        config.train_steps, batch_acc, loss
                ), flush=True)
            
            global_step += 1
            
        epoch_acc, epoch_nll = test_model(model, embedding, criterion, valid_iter, device) 
        print("Valid NLL: {}".format(epoch_nll))
        model.train()
        sw.add_scalar('Valid/NLL', epoch_nll, global_step)
        sw.add_scalar('Valid/Accuracy', epoch_acc, global_step)
        sw.flush()
        
        if epoch_acc > best_acc:
            save_model("best", model, config)
        
        if global_step >= config.train_steps:
                break

        scheduler.step() 
        print("Learning Rate: {}".format([group['lr'] for group in optimizer.param_groups]))


    print('Done training.')

    epoch_acc, epoch_nll = test_model(model, embedding, criterion, test_iter, device)
    print("Test NLL: {}".format(epoch_nll))

    return model

################################################################################

def logloc(comment='',dir_name='runs'):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(dir_name, current_time + '_' + socket.gethostname() + comment)
    return log_dir

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model', choices=['rnnlm', 's-vae'], default='rnnlm', help="Which model to train.")
    parser.add_argument('--embed_dim', type=int, default=300, help="The amount of dimensions of the word embeddings.")
    parser.add_argument('--hidden_dim', type=int, default=512, help="The amount of hidden dimensions.")
   
    parser.add_argument('--freebits_lambda', type=float, default=0, help="")
    parser.add_argument('--wdropout_prob', type=float, default=1, help="")
    parser.add_argument('--mu_forcing_beta', type=float, default=0, help="")

    # Training params
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples to process in a batch')
    parser.add_argument('--device', type=str, default="cuda", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.95, help='Learning rate decay fraction')
    parser.add_argument('--train_steps', type=int, default=int(10000), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='Gradient clipping maximum norm.')
    parser.add_argument('--seq_len', type=int, default=int(25), help='The length of the sequences to train on.')

    # Misc params
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sw_log_dir', type=str, default='runs', help='The directory in which to create the default logdir.')
    parser.add_argument('--save_file', type=str, default='model.pt', help='Filename under which to store the model.')

    config = parser.parse_args()

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    # summarywriter 
    comment = "_fblambda={}_wdropout={}".format(config.freebits_lambda, config.wdropout_prob)
    logdir = logloc(dir_name=config.sw_log_dir, comment=comment)
    sw = SummaryWriter(log_dir=logdir)

    # Train and save the model
    model = train(config, sw)
    model = model.cpu()

    sw.close()

    save_model("last", model, config)

