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

def file_name(label, config):
    return "{}_{}_{}".format(
            config.model,
            label,
            config.save_file
    )

def save_model(label, model, config):
    name = file_name(label, config)
    torch.save(model, name)

def load_model(label, config):
    name = file_name(label, config) 
    return torch.load(name)


def test_model(model, embedding, criterion, iter_, device):
    """
    Tests/Validates model on supplied iterator.
    """
    with torch.no_grad():
        model.eval()
        kl = 0
        nll = 0
        marg = 0
        lens = 0
        additional_losses = model.get_additional_losses()
        for step, batch in enumerate(iter_):
            batch_text, txt_len = batch.text
            batch_target, tgt_len = batch.target
            batch_text = embedding(batch_text.to(device))
            batch_target = batch_target.to(device)
            batch_output = model(batch_text, txt_len)
           
            K = 10
            splits = int(K/2) 
            itr = zip(
                torch.split(batch_text, splits, dim=1), 
                torch.split(txt_len, splits, dim=0), 
                torch.split(batch_target, splits, dim=1)
            )

            for txt, ln, tgt in itr:
                m, c, d = model.multi_sample_estimates(txt, ln, tgt, K=K)
                marg += m.sum()
                nll += c.sum().item()
                kl += d.sum().item()

            lens += txt_len.sum()

            for loss_name, additional_loss in model.get_additional_losses().items():
                additional_losses[loss_name] += additional_loss

    pp = torch.exp(marg / lens)

    for loss_name, additional_loss in model.get_additional_losses().items():
        additional_losses[loss_name] /= step

    nr_samples = (step * iter_.batch_size)
    nll_per_sample = nll / nr_samples 
    kl_per_sample = kl / nr_samples
    return nll_per_sample, pp, kl_per_sample, additional_losses 


def train(config, sw):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    vocab = torchtext.vocab.FastText()
    #vocab = torchtext.vocab.GloVe()

    # get data iterators
    lm_iters, s_iters = load_data(
            embeddings=vocab, 
            device=device, 
            batch_size=config.batch_size,
            bptt_len=config.seq_len
        )

    _, valid_iter, test_iter, field = s_iters
    vocab = field.vocab

    if config.use_bptt:
        train_iter,_,_,_ = lm_iters
    else:
        train_iter,_,_,_ = s_iters

    print("Vocab size: {}".format(vocab.vectors.shape))

    # create embedding layer
    embedding = nn.Embedding.from_pretrained(vocab.vectors).to(device)
    EMBED_DIM = 300


    num_classes = vocab.vectors.shape[0]
    # Initialize the model that we are going to use
    if config.model == "rnnlm":
        model = RNNLM(
            EMBED_DIM,
            config.hidden_dim,
            num_classes
        )
    elif config.model == "s-vae":
        model = SentenceVAE(
            EMBED_DIM,
            config.hidden_dim,
            num_classes,
            fb_lambda = config.freebits_lambda, 
            wd_keep_prob = config.wdropout_prob, 
            wd_unk = embedding(torch.LongTensor([vocab.stoi["<unk>"]]).to(device)),
            mu_f_beta = config.mu_forcing_beta
        )
    else: raise Error("Invalid model parameter.")
    model = model.to(device)

    # Setup the loss, optimizer, lr-scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) 
    criterion = torch.nn.NLLLoss(reduction="sum").to(config.device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.learning_rate_decay)
    lr = config.learning_rate

    global_step = 0
    best_nll = sys.maxsize
    best_pp = sys.maxsize
    best_kl = None
    for epoch in itertools.count():
        for batch in train_iter:
            
            # [1] Get data
            if config.use_bptt:
                batch_text = batch.text
                batch_target = batch.target
                txt_len = torch.full((batch_text.shape[1],), batch_text.shape[0], device=device)
                tgt_len = txt_len
            else:
                batch_text, txt_len = batch.text
                batch_target, tgt_len = batch.target

            batch_text = embedding(batch_text.to(device))
            batch_target = batch_target.to(device)
           
            # [2] Forward & Loss
            batch_output = model(batch_text, txt_len)

            # merge batch and sequence dimension for evaluation
            batch_output = batch_output.view(-1, batch_output.shape[2])
            batch_target = batch_target.view(-1)

            B = batch_text.shape[1]
            nll = criterion(batch_output, batch_target) / B
            sw.add_scalar('Train/NLL', nll.item(), global_step)

            loss = nll.clone()
            for loss_name, additional_loss in model.get_additional_losses().items():
                loss += additional_loss
                sw.add_scalar('Train/'+loss_name, additional_loss, global_step)

            # [3] Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            sw.add_scalar('Train/Loss', loss.item(), global_step)

            if global_step % config.print_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, "
                        "NLL = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), global_step,
                        config.train_steps, nll.item(), loss.item()
                ), flush=True)
            
            global_step += 1
            
        epoch_nll, epoch_pp, epoch_kl, additional_losses = test_model(model, embedding, criterion, valid_iter, device) 
        model.train()

        print("Valid NLL: {}".format(epoch_nll))
        print("Valid Perplexity: {}".format(epoch_pp))
        print("Valid KL: {}".format(epoch_kl))
        sw.add_scalar('Valid/NLL', epoch_nll, global_step)
        sw.add_scalar('Valid/Perplexity', epoch_pp, global_step)
        sw.add_scalar('Valid/KL', epoch_kl, global_step)

        # the additional_loss below will also have kl but not multisample
        for loss_name, additional_loss in additional_losses.items():
            sw.add_scalar('Valid/'+loss_name, additional_loss, global_step)
      
        # sample some sentences
        MAX_LEN = 50
        for _ in range(5):
            text = model.temperature_sample(embedding, MAX_LEN)
            text = ' '.join(vocab.itos[w] for w in text)
            print(text)
            sw.add_text('Valid/Sample-text', text, global_step)

        if epoch_nll < best_nll:
            best_nll = epoch_nll
            save_model("best", model, config)
        if epoch_pp < best_pp:
            best_pp = epoch_pp
        
        if global_step >= config.train_steps:
            break

        scheduler.step() 
        print("Learning Rate: {}".format([group['lr'] for group in optimizer.param_groups]))


    print('Done training.')

    best_model = load_model("best", config)
    test_nll, test_pp, test_kl, test_additional_losses = test_model(best_model, embedding, criterion, test_iter, device)
    print("Test NLL: {}".format(test_nll))
    print("Test PP: {}".format(test_pp))
    print("Test KL: {}".format(test_kl))
    print("{}".format(test_additional_losses))

    return best_model, model, {'hparam/nll':best_nll ,'hparam/pp':best_pp}

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
    parser.add_argument('--hidden_dim', type=int, default=512, help="The amount of hidden dimensions.")
   
    parser.add_argument('--freebits_lambda', type=float, default=0, help="")
    parser.add_argument('--wdropout_prob', type=float, default=1, help="")
    parser.add_argument('--mu_forcing_beta', type=float, default=0, help="")

    # Training params    
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples to process in a batch')
    parser.add_argument('--device', type=str, default="cuda", help="Training device 'cpu' or 'cuda'")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.95, help='Learning rate decay fraction')
    parser.add_argument('--train_steps', type=int, default=int(14000), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='Gradient clipping maximum norm.')

    parser.add_argument('--use_bptt', type=bool, default=False, help='')
    parser.add_argument('--seq_len', type=int, default=int(25), help='The length of the sequences to train on.')

    # Misc params
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sw_log_dir', type=str, default='runs', help='The directory in which to create the default logdir.')
    parser.add_argument('--save_file', type=str, default='model.pt', help='Filename under which to store the model.')

    config = parser.parse_args()
    print(config)

    # summarywriter 
    comment = "" 
    logdir = logloc(dir_name=config.sw_log_dir, comment=comment)
    sw = SummaryWriter(log_dir=logdir)

    # Train and save the model
    best_model, model, metrics = train(config, sw)
    model = model.cpu()

    config_dict = dict(vars(config))
    for key in ['use_bptt','seq_len','device','train_steps','print_every','sw_log_dir','save_file']:
        del config_dict[key]

    sw.add_hparams(config_dict, metrics)
    sw.close()

    #save_model("last", model, config)

