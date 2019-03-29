import random
import glob
import sys

#Set seed for reproducibility.
random.seed(42)

def unicode_convert(c):
    '''
    Convert character to acceptable Unicode value.
    0 "null"
    10 line feed LF
    32-64 numbers and punctuation
    65-90 upper-case letters
    91-97 more punctuation
    97-122 lower-case letters
    123-126 more punctuation
    '''
    if c == 9:
        return 1
    if c == 10:
        return 127 - 30
    elif 32 <= c <= 126:
        return c - 30
    else:
        return 0

def ascii_convert(c):
    '''
    Convert characters to ASCII.
    Unknown = 0
    Tab = 1
    Space = 2
    All chars from 32 to 126 = c-30
    LF mapped to 127-30
    '''
    if c == 1:
        return 9
    if c == 127 - 30:
        return 10
    if 32 <= c + 30 <= 126:
        return c + 30
    else:
        return 0

def encode_text(s):
    '''
    Convert text to unicode and remove troublesome characters.
    '''
    return list(map(lambda c: unicode_convert(ord(c)), s))

def load_corpus(directory):
    '''
    Load text corpus and return it with validation set and file ranges.
    '''
    corpus = []
    file_ranges = []
    file_list = glob.glob(directory, recursive = True)
    random.shuffle(file_list)

    for txtfile in file_list:
        file_text = open(txtfile,"r")
        start = len(corpus)
        corpus.extend(encode_text(file_text.read()))
        end = len(corpus)
        file_ranges.append({"start": start, "end": end,
            "name": txtfile.rsplit("/",1)[-1]})

    file_text.close()
    
    #Put aside 10% of files for validation.
    nval_files = len(file_ranges) // 10
    cutoff = file_ranges[-nval_files]["start"]
    valitxt = corpus[cutoff:]
    corpus = corpus[:cutoff]

    return corpus, valitxt, file_ranges
