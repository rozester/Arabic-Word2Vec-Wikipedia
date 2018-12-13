# -*- coding: utf-8 -*-

from nltk.tokenize import TreebankWordTokenizer
from gensim.models import Word2Vec
from gensim.models import FastText
import re
import os
import json

# Arabic Text Cleansing
accents = re.compile(r'[\u064b-\u0652\u0640]') # harakaat and tatweel
puncs = re.compile(r'[\u061b\u061f\u060c\u003A\u003D\u002E\u002F\u007C\u002D]')
arabic = re.compile(r'[\u0621-\u063A\u0641-\u064A]+') # Arabic letters only
nonArabic = re.compile(r'[^\u0621-\u063A\u0641-\u064A]')
def ArTokenizer(text,token_min_len=2, token_max_len=15, lower=False):
    tokens = TreebankWordTokenizer().tokenize(accents.sub('',puncs.sub(' ', text)))
    # keep only Ar words between min/max len and remove other characters if any
    return [nonArabic.sub('',token) for token in tokens if arabic.findall(token) and token_min_len <= len(token) <= token_max_len]

# Reading from Wikipedia extracted directory
folders = ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL']
wikis = []
for ff in folders:
    for i in range(0, 100):
        ext = str(i) if i >= 10 else '0' + str(i)
        if os.path.exists("wikiextractor/text/" + ff + "/wiki_" + ext):
            with open("wikiextractor/text/" + ff + "/wiki_" + ext, encoding="utf-8") as f:
                for line in f.readlines():
                    l = json.loads(line)
                    wikis.append({"title": l["title"], "size": len(l["text"]), "text": l["text"]})

# Calling Arabic Text Wrangling            
sentences = []
for wiki in wikis:
    text = wiki['text'].replace("\\n\\n", "\n\n")
    text = ArTokenizer(text)
    sentences.append(text)
  
# Start Training the Model using Word2Vec
Word2Vec_model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=8, sg=0)
Word2Vec_model.save('ar_wiki_word2vec')

# Start Training the Model using FastText
FastText_model = FastText(sentences, size=300, window=5, min_count=5, workers=8, sg=0)
FastText_model.save('ar_wiki_FastText')

