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
    Function that loads and prepares the Penn Treebank data in two different ways.
    The first takes fixed length pieces of the entire training text.
    The second takes each sentence as is.

    Args:
        embeddings: the pre-trained word embeddings to use.
        batch_size: size of batches.
        bptt_len: the length of the sequences in the batches
        path_to_data: where the Penn Treebank data is located.
        train, valid, test: the files to use as train/valid/test.
    """ 

    # Already tokenized so use identity function.
    TEXT = data.Field(lower=True, tokenize=lambda x:x)
    SENTENCE = data.Field(lower=True, tokenize=lambda x:x, include_lengths=True)

    lm_fields = [("text", TEXT)]
    s_fields = [("text", SENTENCE), ("target", SENTENCE)]
   
    print("Loading data...")

    # Extract sentences from files; turn into examples.
    splits_langmodel = []
    splits_sentences = []
    for f in [train, valid, test]:
        path = os.path.join(path_to_data, f)
       
        # Remove POS tags and concatenate into one list for language modelling.
        nr_lines = 0
        total_tokens = 0
        lm_example = []
        s_examples = []
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                nr_lines += 1
                # remove POS tags and tree structure.
                tokens = [bos_token] + re.sub(r"\([0-9] |\)", "", line).split()
                tokens = [token for token in tokens if not token.startswith('(')] 
                total_tokens += len(tokens)

                lm_example.extend(tokens)
                s_examples.append([tokens, tokens[1:]+[bos_token]])

        avg_length = total_tokens / nr_lines
        print("Average Sentence Length: {}".format(avg_length))

        # The language model datasets are one big Example with all sentences.
        lm_example = data.Example.fromlist([lm_example], lm_fields)
        dataset = data.Dataset([lm_example], lm_fields)
        splits_langmodel.append(dataset)

        # the sentence datasets contain each sentence as a separate Example.
        examples = [ data.Example.fromlist(example, s_fields) for example in s_examples ]
        dataset = data.Dataset(examples, s_fields)
        splits_sentences.append(dataset)

    print("Done loading.")
  
    # To reduce vocabulary to roughly 22.000 .
    MIN_FREQ = 2

    specials = ['<unk>', '<pad>', bos_token]
    if embeddings:
        TEXT.build_vocab(*splits_langmodel, min_freq=MIN_FREQ, vectors=embeddings, specials=specials)
    else:
        TEXT.build_vocab(*splits_langmodel, min_freq=MIN_FREQ, specials=specials)

    # Use BPTTIterator for LM variant.
    train, valid, test = splits_langmodel
    lm_train_iter, lm_valid_iter, lm_test_iter = data.BPTTIterator.splits(
            (train, valid, test),
            batch_size = batch_size,
            bptt_len = bptt_len,
            shuffle = True,
            device = device
        )

    # Make validation/test fit in memory (multi-sample estimates required a bit more).
    VALID_TEST_BATCH_SIZE = 16

    train, valid, test = splits_sentences
    s_train_iter = data.BucketIterator(
            train,
            batch_size = batch_size,
            sort_key = lambda x:x,
            shuffle = True,
            sort = False,
            device = device
        )
    s_valid_iter, s_test_iter = data.BucketIterator.splits(
            (valid, test),
            batch_size = VALID_TEST_BATCH_SIZE,
            shuffle = True,
            sort = False,
            device = device
        )


    SENTENCE.vocab = TEXT.vocab
    
    return (lm_train_iter, lm_valid_iter, lm_test_iter, TEXT), \
           (s_train_iter, s_valid_iter, s_test_iter, SENTENCE) 


if __name__ == "__main__":

    print("Testing dataloader.")

    (_,_,_,field2), (train, valid, test, field) = load_data()

    print(field.vocab.vectors)
    print(field2.vocab.vectors)

    print('\n'.join([f'{i}: ' + ' '.join(example.text[1:]) for i,example in enumerate(test.dataset)]))
    
    
