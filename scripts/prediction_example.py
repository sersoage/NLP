#!/usr/bin/python
# Predicition example on a txt file
from __future__ import print_function
import nltk
from util.preprocess import addChar, createMtx, addCasing
from layers.SeqTagger import SeqTagger
import sys

if len(sys.argv) < 3:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]

#Read input
with open(inputPath, 'r') as f:
    text = f.read()

#Load the model ::
SeqModel = SeqTagger.loadModel(modelPath)


#Prepare the input
sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
addChar(sentences)
addCasing(sentences)
dataMatrix = createMtx(sentences, SeqModel.mappings, True)

#Tag the input
tags = SeqModel.tagSentences(dataMatrix)

#Output
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']

    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

        print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    print("")


#Dict of tags
labels=[(k) for k in tags.items()]

