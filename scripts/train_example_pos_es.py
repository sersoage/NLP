# Example script to train POS tagging

from __future__ import print_function
import os
import logging
import sys
from layers.SeqTagger import SeqTagger
from util.preprocess import prepareDataset, loadPickle



# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'es':                 #Name of the dataset
        {'columns': {0:'tokens', 1:'POS'},   #Input format
         'label': 'POS',                     #Prediction
         'evaluate': True,                   #Evaluation
         'commentSymbol': None}              #Skip lines if
}


#Embeddings to use
embeddingsPath = 'emb/emb_es.txt'

#Dataset format required by the net
pickleFile = prepareDataset(embeddingsPath, datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadPickle(pickleFile)

#Hyperparams setting
params = {'classifier': ['CRF'], 'LSTM-Size': [128,128], 'dropout': (0.25, 0.25)}

model = SeqTagger(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/pos_es_results.csv') #Store performance
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" #Store models
model.fit(epochs=25)



