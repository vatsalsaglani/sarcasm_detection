from tokenizers import BertWordPieceTokenizer
import json
import pickle
import torch 
import torch.nn as nn
import torch.nn.functional as F

from AttentionTransformer.utilities import count_model_parameters, device



def pad_sequences(arr, shape = (200)):

    if len(arr) > shape:
        arr = arr[:shape]
    
    arr = torch.tensor(arr)
    op = torch.zeros((shape))
    op[:arr.size(0)] = arr
    return op.long().unsqueeze(0), torch.zeros((shape)).long().unsqueeze(0)



def getTextPrediction(text, model, tokenizer):

    labels = ['Not Sarcasm', 'Sarcasm']
    tokens = tokenizer.encode(text).ids
    tnsr, trg = pad_sequences(tokens)
    tnsr, trg = tnsr.to(device()), trg.to(device())
    output = model(tnsr, trg)
    return labels[output.max(1)[-1].item()]
