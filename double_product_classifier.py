import argparse
import logging

import numpy as np

import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss


from transformers.modeling_utils import SequenceSummary

class Double_Product_Classifier(nn.Module):
    """
    Left and righ product for classification
    """
    def __init__(self, config):
        print("************ THIS MODEL COMES FROM CS224N PROJECT ************")
        super(Double_Product_Classifier, self).__init__()
        self.middle_weight = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.bias = nn.Parameter(torch.tensor([0.0]))
        #self.multiplier = nn.Parameter(torch.tensor([1.1]))

    def forward(
        self,
        sentence_embed_1,
        sentence_embed_2,
        mc_labels = None
    ):
        tmp = self.middle_weight(sentence_embed_1)
        tmp = tmp.view(tmp.shape[0], 1, tmp.shape[1])
        
        reshape_embed_2 = sentence_embed_2.view(sentence_embed_2.shape[0], sentence_embed_2.shape[1], 1)
        mc_logits = torch.bmm(tmp, reshape_embed_2)
        
        #reshaped_embed_1 = sentence_embed_1.view(sentence_embed_1.shape[0], 1, sentence_embed_1.shape[1])
        #mc_logits += torch.bmm(reshaped_embed_1, reshape_embed_2)
        
        mc_logits.squeeze(-1)

        #mc_logits *= self.multiplier 
        mc_logits += self.bias

        outputs = (mc_logits,) 
        if mc_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(mc_logits.view(-1), mc_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  


