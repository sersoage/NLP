from __future__ import (division, absolute_import, print_function, unicode_literals)
import os
import numpy as np
import gzip
import os.path
import nltk
import logging
from nltk import FreqDist

from .wordemb import wordNormalize
from .CoNLL import readCoNLL

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
    from io import open

def prepareDataset(embeddingsPath, datasets, frequencyThresholdUnknownTokens=50, reducePretrainedEmbeddings=False, valTransformations=None, padOneTokenSentence=True):
    """
        Prepares dataset to be consumed by the network
    """
    embeddingsName = os.path.splitext(embeddingsPath)[0]
    pklName = "_".join(sorted(datasets.keys()) + [embeddingsName])
    outputPath = 'pkl/' + pklName + '.pkl'

    if os.path.isfile(outputPath):
        logging.info("Using existent pickle file: %s" % outputPath)
        return outputPath

    casing2Idx = getCasingVocab()
    embeddings, word2Idx = readEmbed(embeddingsPath, datasets, frequencyThresholdUnknownTokens, reducePretrainedEmbeddings)
    
    mappings = {'tokens': word2Idx, 'casing': casing2Idx}
    pklObjects = {'embeddings': embeddings, 'mappings': mappings, 'datasets': datasets, 'data': {}}

    for datasetName, dataset in datasets.items():
        datasetColumns = dataset['columns']
        commentSymbol = dataset['commentSymbol']

        trainData = 'data/%s/train.txt' % datasetName 
        devData = 'data/%s/dev.txt' % datasetName 
        testData = 'data/%s/test.txt' % datasetName 
        paths = [trainData, devData, testData]

        logging.info(":: Transform "+datasetName+" dataset ::")
        pklObjects['data'][datasetName] = createPkl(paths, mappings, datasetColumns, commentSymbol, valTransformations, padOneTokenSentence)

    
    f = open(outputPath, 'wb')
    pkl.dump(pklObjects, f, -1)
    f.close()
    
    logging.info("DONE - Embeddings file saved: %s" % outputPath)
    
    return outputPath


def loadPickle(embeddingsPickle):
    """ Loads the cPickle file """
    f = open(embeddingsPickle, 'rb')
    pklObjects = pkl.load(f)
    f.close()

    return pklObjects['embeddings'], pklObjects['mappings'], pklObjects['data']



def readEmbed(embeddingsPath, datasetFiles, frequencyThresholdUnknownTokens, reducePretrainedEmbeddings):
    """
        Load embeddings
    """

    logging.info("Generate new embeddings files for a dataset")

    neededVocab = {}
    if reducePretrainedEmbeddings:
        logging.info("Compute which tokens are required for the experiment")

        def createDict(filename, tokenPos, vocab):
            for line in open(filename):
                if line.startswith('#'):
                    continue
                splits = line.strip().split()
                if len(splits) > 1:
                    word = splits[tokenPos]
                    wordLower = word.lower()
                    wordNormalized = wordNormalize(wordLower)

                    vocab[word] = True
                    vocab[wordLower] = True
                    vocab[wordNormalized] = True

        for dataset in datasetFiles:
            dataColumnsIdx = {y: x for x, y in dataset['cols'].items()}
            tokenIdx = dataColumnsIdx['tokens']
            datasetPath = 'data/%s/' % dataset['name']

            for dataset in ['train.txt', 'dev.txt', 'test.txt']:
                createDict(datasetPath + dataset, tokenIdx, neededVocab)

    # :: Read in word embeddings ::
    logging.info("Read file: %s" % embeddingsPath)
    word2Idx = {}
    embeddings = []

    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
                                                                                               encoding="utf8")

    embeddingsDimension = None

    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]

        if embeddingsDimension == None:
            embeddingsDimension = len(split) - 1

        if (len(
                split) - 1) != embeddingsDimension:  
            print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue

        if len(word2Idx) == 0:  
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension)
            embeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)  
            embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])

        if len(neededVocab) == 0 or word in neededVocab:
            if word not in word2Idx:
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)

    # Extend embeddings file with new tokens
    def createFD(filename, tokenIndex, fd, word2Idx):
        for line in open(filename):
            if line.startswith('#'):
                continue

            splits = line.strip().split()

            if len(splits) > 1:
                word = splits[tokenIndex]
                wordLower = word.lower()
                wordNormalized = wordNormalize(wordLower)

                if word not in word2Idx and wordLower not in word2Idx and wordNormalized not in word2Idx:
                    fd[wordNormalized] += 1

    if frequencyThresholdUnknownTokens != None and frequencyThresholdUnknownTokens >= 0:
        fd = nltk.FreqDist()
        for datasetName, datasetFile in datasetFiles.items():
            dataColumnsIdx = {y: x for x, y in datasetFile['columns'].items()}
            tokenIdx = dataColumnsIdx['tokens']
            datasetPath = 'data/%s/' % datasetName
            createFD(datasetPath + 'train.txt', tokenIdx, fd, word2Idx)

        addedWords = 0
        for word, freq in fd.most_common(10000):
            if freq < frequencyThresholdUnknownTokens:
                break

            addedWords += 1
            word2Idx[word] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)

            assert (len(word2Idx) == len(embeddings))

        logging.info("Added words: %d" % addedWords)
    embeddings = np.array(embeddings)

    return embeddings, word2Idx


def addChar(sentences):
    """Breaks every token into the characters"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['characters'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            chars = [c for c in token]
            sentences[sentenceIdx]['characters'].append(chars)

def addCasing(sentences):
    """Adds information of the casing of words"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['casing'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            sentences[sentenceIdx]['casing'].append(getCasing(token))
       
       
def getCasing(word):   
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing

def getCasingVocab():
    entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    return {entries[idx]:idx for idx in range(len(entries))}


def createMtx(sentences, mappings, padOneTokenSentence):
    data = []
    numTokens = 0
    numUnknownTokens = 0    
    missingTokens = FreqDist()
    paddedSentences = 0

    for sentence in sentences:
        row = {name: [] for name in list(mappings.keys())+['raw_tokens']}
        
        for mapping, str2Idx in mappings.items():    
            if mapping not in sentence:
                continue
                    
            for entry in sentence[mapping]:                
                if mapping.lower() == 'tokens':
                    numTokens += 1
                    idx = str2Idx['UNKNOWN_TOKEN']
                    
                    if entry in str2Idx:
                        idx = str2Idx[entry]
                    elif entry.lower() in str2Idx:
                        idx = str2Idx[entry.lower()]
                    elif wordNormalize(entry) in str2Idx:
                        idx = str2Idx[wordNormalize(entry)]
                    else:
                        numUnknownTokens += 1    
                        missingTokens[wordNormalize(entry)] += 1
                        
                    row['raw_tokens'].append(entry)
                elif mapping.lower() == 'characters':  
                    idx = []
                    for c in entry:
                        if c in str2Idx:
                            idx.append(str2Idx[c])
                        else:
                            idx.append(str2Idx['UNKNOWN'])                           
                                      
                else:
                    idx = str2Idx[entry]
                                    
                row[mapping].append(idx)
                
        if len(row['tokens']) == 1 and padOneTokenSentence:
            paddedSentences += 1
            for mapping, str2Idx in mappings.items():
                if mapping.lower() == 'tokens':
                    row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
                    row['raw_tokens'].append('PADDING_TOKEN')
                elif mapping.lower() == 'characters':
                    row['characters'].append([0])
                else:
                    row[mapping].append(0)
            
        data.append(row)
    
    if numTokens > 0:           
        logging.info("Unknown-Tokens: %.2f%%" % (numUnknownTokens/float(numTokens)*100))
        
    return data
    
  
  
def createPkl(datasetFiles, mappings, cols, commentSymbol, valTransformation, padOneTokenSentence):
    trainSentences = readCoNLL(datasetFiles[0], cols, commentSymbol, valTransformation)
    devSentences = readCoNLL(datasetFiles[1], cols, commentSymbol, valTransformation)
    testSentences = readCoNLL(datasetFiles[2], cols, commentSymbol, valTransformation)    
   
    extendMappings(mappings, trainSentences+devSentences+testSentences)

                
    
    charset = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        charset[c] = len(charset)
    mappings['characters'] = charset
    
    addChar(trainSentences)
    addCasing(trainSentences)
    
    addChar(devSentences)
    addCasing(devSentences)
    
    addChar(testSentences)   
    addCasing(testSentences)

    logging.info(":: Create Train Matrix ::")
    trainMatrix = createMtx(trainSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Dev Matrix ::")
    devMatrix = createMtx(devSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Test Matrix ::")
    testMatrix = createMtx(testSentences, mappings, padOneTokenSentence)

    
    data = {
                'trainMatrix': trainMatrix,
                'devMatrix': devMatrix,
                'testMatrix': testMatrix
            }        
       
    
    return data

def extendMappings(mappings, sentences):
    sentenceKeys = list(sentences[0].keys())
    sentenceKeys.remove('tokens') #No need to map tokens

    for sentence in sentences:
        for name in sentenceKeys:
            if name not in mappings:
                mappings[name] = {'O':0} #'O' is also used for padding

            for item in sentence[name]:              
                if item not in mappings[name]:
                    mappings[name][item] = len(mappings[name])


