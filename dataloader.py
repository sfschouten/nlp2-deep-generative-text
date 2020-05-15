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
    lm_fields = [("text", TEXT)]
    s_fields = [("text", TEXT), ("target", TEXT)]
   
    print("Loading data...")

    # extract sentences from files; turn into examples
    splits_langmodel = []
    splits_sentences = []
    for f in [train, valid, test]:
        path = os.path.join(path_to_data, f)
       
        # remove POS tags and concatenate into one list for language modelling.
        nr_lines = 0
        total_tokens = 0
        lm_example = []
        s_examples = []
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                nr_lines += 1
                tokens = [bos_token] + re.sub(r"\([0-9] |\)", "", line).split()
                tokens = [token for token in tokens if not token.startswith('(')]
                total_tokens += len(tokens)
                lm_example.extend(tokens)
                s_examples.append([tokens, tokens[1:]+[bos_token]])

        avg_length = total_tokens / nr_lines
        print("Average Sentence Length: {}".format(avg_length))

        # The language model datasets are one big example with all sentences.
        lm_example = data.Example.fromlist([lm_example], lm_fields)
        dataset = data.Dataset([lm_example], lm_fields)
        splits_langmodel.append(dataset)

        # 
        examples = [ data.Example.fromlist(example, s_fields) for example in s_examples ]
        dataset = data.Dataset(examples, s_fields)
        splits_sentences.append(dataset)

    print("Done loading.")
  
    specials = ['<unk>', '<pad>', bos_token]
    if embeddings:
        TEXT.build_vocab(*splits_langmodel, min_freq=2, vectors=embeddings, specials=specials)
    else:
        TEXT.build_vocab(*splits_langmodel, min_freq=2, specials=specials)

    train, valid, test = splits_langmodel
    lm_train_iter, lm_valid_iter, lm_test_iter = data.BPTTIterator.splits(
            (train, valid, test),
            batch_size = batch_size,
            bptt_len = bptt_len,
            shuffle = True,
            device = device
        )

    #TODO create labels for these dataiterators?

    train, valid, test = splits_sentences
    s_train_iter, s_valid_iter, s_test_iter = data.BucketIterator.splits(
            (train, valid, test),
            batch_size = 1,#(batch_size, 1, 1),
            shuffle = True,
            sort = False,
            device = device
        )

    return (lm_train_iter, lm_valid_iter, lm_test_iter), \
           (s_train_iter, s_valid_iter, s_test_iter), \
           TEXT.vocab


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
