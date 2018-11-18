import spacy
import numpy as np
from underthesea import word_tokenize
import pdb
import os

def tokenize_vi(f_names, f_out_names):
    for f_name, f_out_name in zip(f_names, f_out_names):
        lines = open(f_name, 'r').readlines()
        tok_lines = open(f_out_name, 'w')
        for i, sentence in enumerate(lines):
            if i > 0 and i % 100 == 0:
                print (f_name.split('/')[-1], i, len(lines))
            tok_lines.write(word_tokenize(sentence, format="text") + '\n')
        tok_lines.close()

def tokenize_en(f_names, f_out_names):
    tokenizer = spacy.load('en_core_web_sm')

    for f_name, f_out_name in zip(f_names, f_out_names):
        lines = open(f_name, 'r').readlines()
        tok_lines = open(f_out_name, 'w')
        for i, sentence in enumerate(lines):
            if i > 0 and i % 100 == 0:
                print (f_name.split('/')[-1], i, len(lines))
            tok_lines.write(' '.join(tokenizer(sentence)) + '\n')
        tok_lines.close()


if __name__ == "__main__":
    root = '/Users/mansimov/datasets/iwslt-vi-en'
    #tokenize_vi([os.path.join(root, 'train.vi'), os.path.join(root, 'dev.vi'), os.path.join(root, 'test.vi')],\
    #            [os.path.join(root, 'train.tok.vi'), os.path.join(root, 'dev.tok.vi'), os.path.join(root, 'test.tok.vi')])

    tokenize_en([os.path.join(root, 'train.en'), os.path.join(root, 'dev.en'), os.path.join(root, 'test.en')],\
                [os.path.join(root, 'train.tok.en'), os.path.join(root, 'dev.tok.en'), os.path.join(root, 'test.tok.en')])
