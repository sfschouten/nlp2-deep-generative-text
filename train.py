import sys
import os
import time
import argparse
import itertools
import math

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

################################################################################

def save_model(label, model, config):
    name = "{}_{}e_{}h_{}_{}".format(
            config.model,
            config.embed_dims,
            config.hidden_dims,
            label,
            config.save_file
    )
    torch.save(model, name)


def train(config, sw):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    vocab = torchtext.vocab.FastText()
    #vocab = torchtext.vocab.GloVe()

    # get data iterators
    train_iter, valid_iter, test_iter, vocab = load_data(
            embeddings=vocab, device=device, batch_size=config.batch_size)

    print("vocab size: {}".format(vocab.vectors.shape))

    # Initialize the model that we are going to use
    if config.model == "rnnlm":
        model = RNNLM(
            config.embed_dim, 
            config.hidden_dim,
            vocab.vectors.shape[0]
        )
    else: raise Error("Invalid model parameter.")
    model = model.to(device)

    print("transferring embedding")
    # create embedding layer
    embedding = nn.Embedding.from_pretrained(vocab.vectors).to(device)


    # Setup the loss, optimizer, lr-scheduler
    criterion = torch.nn.NLLLoss().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) 
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.learning_rate_decay)
    lr = config.learning_rate
   
    best_acc = 0
    for epoch in itertools.count():
        for step, batch in enumerate(train_iter):
            global_step = epoch * math.floor( len(train_iter.dataset) / config.batch_size ) + step

            batch_text = embedding(batch.text.to(device))
            batch_target = batch.target.to(device)
            batch_output = model(batch_text)

            # merge batch and sequence dimension for evaluation
            batch_output = batch_output.view(-1, batch_output.shape[2])
            batch_target = batch_target.view(-1)

            batch_acc = batch_output.argmax(dim=1).eq(batch_target).double().mean()
            loss = criterion(batch_output, batch_target)
        
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            sw.add_scalar('Train/Loss', loss, global_step)
            sw.add_scalar('Train/Accuracy', batch_acc, global_step)

            if global_step % config.print_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, "
                        "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), global_step,
                        config.train_steps, batch_acc, loss
                ), flush=True)
            
            if global_step == config.train_steps:
                break

        #epoch_acc, epoch_loss = test_model(model, embedding, criterion, global_step, valid_iter, device) 
        #sw.add_scalar('Valid/Loss', epoch_loss, global_step)
        #sw.add_scalar('Valid/Accuracy', epoch_acc, global_step)
        #sw.flush()

        scheduler.step() 
        
        print("Learning Rate: {}".format([group['lr'] for group in optimizer.param_groups]))
        if scheduler.get_last_lr()[0] < 0.00001 or global_step >= config.train_steps:
            break

    print('Done training.')

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
    parser.add_argument('--model', required=True, choices=['rnnlm'], default='rnnlm', help="Which sentence encoder to use.")
    parser.add_argument('--embed_dim', type=int, default=300, help="")
    parser.add_argument('--hidden_dim', type=int, default=512, help="")
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of samples to process in a batch')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.99, help='Learning rate decay fraction')
    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='Gradient clipping maximum norm.')

    # Misc params
    parser.add_argument('--print_every', type=int, default=25, help='How often to print training progress')
    parser.add_argument('--sw_log_dir', type=str, default='runs', help='The directory in which to create the default logdir.')
    parser.add_argument('--save_file', type=str, default='model.pt', help='Filename under which to store the model.')

    config = parser.parse_args()

    print(config)

    # summarywriter 
    logdir = logloc(dir_name=config.sw_log_dir)
    sw = SummaryWriter(log_dir=logdir)

    # Train and save the model
    model = train(config, sw)
    model = model.cpu()

    sw.close()

    save_model("last", model, config)

