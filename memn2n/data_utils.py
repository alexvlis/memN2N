from __future__ import absolute_import
from __future__ import print_function

from sklearn import model_selection
from itertools import chain
from six.moves import range, reduce

import os
import re
import numpy as np

def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which 
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)

def get_data_info(data_dir, memory_size):
    """Assigns unique ID to each word in the vocabulary

       Retrieves some data about the train/valid/test sets.
    """
    
    # Load the train/test data
    ids = range(1, 21)
    train, test = [], []
    for i in ids:
        tr, te = load_task(data_dir, i)
        train.append(tr)
        test.append(te)
    data = list(chain.from_iterable(train + test))
    
    # Assign an ID to each word
    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))        
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        
    # Add time words and IDs
    for i in range(memory_size):
            word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)
            
    vocab_size = len(word_idx) + 1 # +1 for null word embedding
    
    # Get sentence size
    max_sentence_size_q = max(map(len, (q for _, q, _ in data)))
    max_sentence_size_s = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    sentence_size = max(max_sentence_size_s, max_sentence_size_q) + 1 # for time words
    
    return word_idx, vocab_size, sentence_size
    
def split_train_valid_test(data_dir, word_idx, sentence_size, memory_size):
    """Splits the data in 'data_dir' into training, validation, and test sets"""
    # Load the train/test data
    ids = range(1, 21)
    train, test = [], []
    for i in ids:
        tr, te = load_task(data_dir, i)
        train.append(tr)
        test.append(te)
    data = list(chain.from_iterable(train + test))

    # Create train/validation/test sets
    train_stories = []
    train_queries = []
    train_answers = []
    val_stories = []
    val_queries = []
    val_answers = []
    test_stories, test_queries, test_answers = vectorize_data(list(chain.from_iterable(test)), word_idx, sentence_size, memory_size)

    for task in train:
        S, Q, A = vectorize_data(task, word_idx, sentence_size, memory_size)
        ts, vs, tq, vq, ta, va = model_selection.train_test_split(S, Q, A, test_size=0.1, random_state=None)
        train_stories.append(ts)
        train_queries.append(tq)
        train_answers.append(ta)
        val_stories.append(vs)
        val_queries.append(vq)
        val_answers.append(va)

    train_stories = reduce(lambda a,b : np.vstack((a,b)), (x for x in train_stories))
    train_queries = reduce(lambda a,b : np.vstack((a,b)), (x for x in train_queries))
    train_answers = reduce(lambda a,b : np.vstack((a,b)), (x for x in train_answers))
    val_stories = reduce(lambda a,b : np.vstack((a,b)), (x for x in val_stories))
    val_queries = reduce(lambda a,b : np.vstack((a,b)), (x for x in val_queries))
    val_answers = reduce(lambda a,b : np.vstack((a,b)), (x for x in val_answers))

    return train_stories, train_queries, train_answers, \
           val_stories, val_queries, val_answers, \
           test_stories, test_queries, test_answers

def generate_batches(batch_size, n_train):
    batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start,end in batches]
    return batches