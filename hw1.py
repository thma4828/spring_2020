# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:05:05 2020

@author: tsmar
"""

import nltk

f = open('textread.txt', encoding='utf8')

data_tokens = nltk.word_tokenize(f.read())

print(len(set(data_tokens)))
print(data_tokens)