"""
Procedures for loading the PennTreebank data for the language modelling task.
"""

import re
import os
import io

import torchtext
import torchtext.data as data

def load_data(embeddings=None, device='cpu', batch_size=32, bptt_len=35, path_to_data="../data",
        train = "02-21.10way.clean", valid = "22.auto.clean",
        test = "23.auto.clean", bos_token='<bos>'):
    """
    Args:
        bptt_len: the length of the sequences in the batches
    """ 

    # fields
    # already tokenized so use identity function
    TEXT = data.Field(lower=True, tokenize=lambda x:x)
    fields = [("text", TEXT)]
    
    # extract sentences from files; turn into examples
    splits = []
    for f in [train, valid, test]:
        path = os.path.join(path_to_data, f)
       
        # remove POS tags and concatenate into one list for language modelling.
        nr_lines = 0
        total_tokens = 0
        example = []
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                nr_lines += 1
                tokens = [bos_token] + re.sub(r"\([0-9] |\)", "", line).split()
                tokens = [token for token in tokens if not token.startswith('(')]
                total_tokens += len(tokens)
                example.extend(tokens)

        avg_length = total_tokens / nr_lines
        print("Average Sentence Length: {}".format(avg_length))

        example = data.Example.fromlist([example], fields)
        dataset = data.Dataset([example], fields)
        splits.append(dataset)
  
    specials = ['<unk>', '<pad>', bos_token]
    if embeddings:
        TEXT.build_vocab(*splits, vectors=embeddings, specials=specials)
    else:
        TEXT.build_vocab(*splits, specials=specials)

    train, valid, test = splits
    train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
            (train, valid, test),
            batch_size = batch_size,
            bptt_len = bptt_len,
            device = device,
            repeat = False
        )

    return train_iter, valid_iter, test_iter, TEXT.vocab


if __name__ == "__main__":

    print("Testing dataloader.")

    train_iter, valid_iter, test_iter, vocab = load_data()

    for i, batch in enumerate(valid_iter):
        print(batch)
        print(batch.text)
        print(batch.target)
        
        print("==========================")

        if i > 15:
            break
