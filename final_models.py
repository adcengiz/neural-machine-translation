from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import spacy
import pdb
import os
from underthesea import word_tokenize
import jieba
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle as pkl
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# ENOCDER - DECODER MODELS
# --------------------------------------------------------------------------------------------------------------
# :RNNencoder: Maps the source sentence to a single vector given the source sentence, source lengths. 
# 			   RNN inside is a GRU.
# :RNNdecoder: Decodes one token at a time, given the input tokens and the hidden vector. Initial hidden vector
#			   is encoder's final hidden vector.
# :Attention:  Attention module. 
# :AttnDecoderRNN: (instead of AttnDecoderRNN_) 
# :CNNencoder: Encoder network with 2 1d convolutional layers. Hidden out is fed into one of the RNN decoders. 
# --------------------------------------------------------------------------------------------------------------

# RNNencoder
class RNNencoder(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_gru_layers=1):
        
        super(RNNencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_gru_layers

        self.embedding = nn.Embedding(self.input_size, 
                                      self.hidden_size,
                                      padding_idx=0)
        
        self.GRU = nn.GRU(self.hidden_size, 
                          self.hidden_size,
                          batch_first = True,
                          bidirectional = False)
        
        if self.GRU.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
    def init_hidden(self, batch_size):
        
        return torch.zeros(self.num_layers*self.num_directions, 
                           batch_size, self.hidden_size).to(device)

    def forward(self, 
    		`	source_sentence, 
    			source_lengths, 
    			hidden):
        
        sort_original_source = sorted(range(len(source_lengths)), 
                               key=lambda sentence: -source_lengths[sentence])
        unsort_to_original_source = sorted(range(len(source_lengths)), 
                                    key=lambda sentence: sort_original_source[sentence])
        
        source_sentence = source_sentence[sort_original_source]
        source_lengths = source_lengths[sort_original_source]
        batch_size, seq_len_source = source_sentence.size()
        
        embeds_source = self.embedding(source_sentence)
        
        embeds_source = torch.nn.utils.rnn.pack_padded_sequence(embeds_source, 
                                                                source_lengths, 
                                                                batch_first=True)
        output = embeds_source
        
        output, hidden = self.GRU(output, hidden)
        
        hidden = hidden.view(batch_size, self.hidden_size)
        
        hidden = hidden[unsort_to_original_source] ## back to original indices
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        hidden = hidden.view(1, batch_size, self.hidden_size)
        
        return hidden, output[unsort_to_original_source]



# RNNdecoder
class RNNdecoder(nn.Module):
    
    def __init__(self, 
                 hidden_size, 
                 vocab_size):
        
        super(RNNdecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, 
                                      self.hidden_size,
                                      padding_idx=0)
        
        
        self.dropout = nn.Dropout(p=0.1)
        
        self.GRU = nn.GRU(self.hidden_size, 
                          self.hidden_size,
                          batch_first=True)
        

        self.linear_layer = nn.Linear(self.hidden_size, self.vocab_size)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

    def forward(self, 
                input_, 
                decoder_hidden,
                encoder_outputs=None):
        
        # seq_len will always be 1 in the decoder at each time step
        batch_size = input_.size(0)
        output = self.embedding(input_)
        output = self.dropout(output)
        
#         cat_out = torch.cat((output, decoder_hidden), 2)

        output, decoder_hidden = self.GRU(output, decoder_hidden)

        output = self.linear_layer(output.squeeze(dim=1))

        output = self.log_softmax(output)

        return output, decoder_hidden


# AttnDecoderRNN_ - not using this anymore
class AttnDecoderRNN_(nn.Module):
    def __init__(self, 
                 hidden_size = 300, 
                 output_size = vien_en_.n_words, 
                 bidirectional = True):
        
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, # vocab size
                                      self.hidden_size) # embed_size = hidden_size
        
        self.dropout = nn.Dropout(p=0.1)
        
        self.GRU = nn.GRU(self.hidden_size, 
                          self.hidden_size,
                          batch_first=True)
        
        self.attn = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.attn_drop = nn.Dropout(p = 0.5)
        
        self.attn_combine = nn.Linear(self.hidden_size*2, 
                                      self.hidden_size)

        self.out = nn.Linear(self.hidden_size, 
                             self.output_size) # feed into softmax over vocabulary
        
        self.log_softmax = nn.LogSoftmax(dim=1) 
        
    def init_hidden(self, batch_size):
        
        hidden_ = torch.zeros(1, batch_size, self.hidden_size).to(device)
        
        return hidden_

    def forward(self, 
                input_, 
                hidden, 
                encoder_outputs):
        
        # decoder seq-len will always be 1
        batch_size = input_.size(0)
        
        out = self.embedding(input_)
        out = self.dropout(out)
        
        hidden = hidden.view(batch_size, 1, self.hidden_size)
        cat = torch.cat((output, hidden),2)

        attention_out = self.attn_drop(self.attn(cat))
        
        attention_weights = F.softmax(torch.bmm(encoder_outputs,
        										attention_out.transpose(1,2)),dim = 1)
        
        attn_applied = torch.sum(encoder_outputs*attention_weights, 
        						 dim = 1).unsqueeze(1)
        
        attn_cat = torch.cat((out, attn_applied), 2)
        
        attn_comb = self.attn_combine(attn_cat)
        
        out = F.relu(attn_comb)

        out, hidden = self.GRU(out, hidden.view(1, batch_size, 
                                                      self.hidden_size)[0].unsqueeze(0))

        out = self.out(out.squeeze(dim=1))

        decoder_out = self.log_softmax(out)

        return decoder_out, hidden

def initLSTM(input_size, 
		 hidden_size, 
		 **kwargs):

    model = nn.LSTM(input_size, 
    				hidden_size,
    				**kwargs)

    for name, param in model.named_parameters():

        if ("weight" in name) or ("bias" in name):
            param.data.uniform_(-0.1, 0.1)

    return model


def initLSTMCell(input_size, 
		     	 hidden_size,
		     	 **kwargs):

    model = nn.LSTMCell(input_size, 
    					hidden_size,
    					**kwargs)

    for name, param in model.named_parameters():

        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)

    return model


def initGRUCell(input_size,
				hidden_size,
				**kwargs):
	
	model = nn.GRUCell(input_size, 
    					hidden_size,
    					**kwargs)

	for name, param in model.named_parameters():

        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)

    return model


# Attention: attention module
class Attention(nn.Module):
    def __init__(self, 
    			 hidden_size, 
    			 attn_size):

        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.attn_size = attn_size

        self.linear_layer1 = nn.Linear(self.hidden_size, self.attn_size)

        self.linear_layer2 = nn.Linear(self.hidden_size + self.attn_size, self.attn_size)
        
    def forward(self, 
    			hidden, 
    			encoder_outs, 
    			source_lengths):

    	# hidden_size -> attn_size
        attn_hidden = self.linear_layer1(hidden)

        # get scores
        attn_score = torch.sum((encoder_outs.transpose(0,1) * attn_hidden.unsqueeze(0)),2)

        attn_mask = torch.transpose(seq_mask(source_lengths, 
        							max_len = max(source_lengths).item()),
        							0,1)

        masked_attn = attn_mask*attn_score
        masked_attn[masked_attn==0] = -1e10

        # softmax over attention to get weights
        attn_scores = F.softmax(masked_attn, dim=0)
        # compute weighted sum according to attention scores
        attn_hidden = torch.sum(attn_scores.unsqueeze(2)*encoder_outs.transpose(0,1), 0)

        attn_hidden = self.linear_layer2(torch.cat((attn_hidden, hidden), dim=1))
        attn_hidden = torch.tanh(attn_hidden)

        return attn_hidden, attn_scores

# AttnDecoderRNN
class AttnDecoderRNN(nn.Module):

    def __init__(self, 
    			 vocab_size, 
    			 embed_size, 
    			 hidden_size, 
    			 num_rnn_layers = 1, 
    			 attention = True,
    			 dropout_percent=0.1):

        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        encoder_output_size = self.hidden_size

        self.embedding = nn.Embedding(vocab_size, 
        							  embed_size, 
        							  PAD_IDX)

        self.dropout_f = nn.Dropout(p=dropout_percent)

        self.num_layers = num_rnn_layers

        if attention:
        	self.attention = Attention(self.hidden_size, 
        						   	   encoder_output_size)
        else:
        	self.attention = None

        self.layers = nn.ModuleList([initLSTMCell(input_size=self.hidden_size+self.embed_size if ((layer == 0) and attention) \
        									  else self.embed_size if layer == 0 else self.hidden_size,
                							  hidden_size=self.hidden_size,)for layer in range(self.num_layers)])

        self.linear_layer = nn.Linear(self.hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, 
                decoder_inputs,
                context, 
                prev_hiddens,
                prev_context,
                encoder_outputs,
                source_lengths):
        
        batch_size = decoder_inputs.size(0)

        # embed
        embed_target = self.embedding(decoder_inputs)
        out = self.dropout_f(embed_target)
        
        if self.attention is not None:
            input_ = torch.cat([out.squeeze(1), context], dim = 1)
        else:
            input_ = out.squeeze(1)

        context_ = []
        decoder_hiddens_ = []

        for layer, rnn in enumerate(self.layers):
            hidden, con = rnn(input_, (prev_hiddens[layer], 
            						   prev_context[layer]))
            input_ = self.dropout_f(hidden)
            decoder_hiddens_.append(hidden.unsqueeze(0))
            context_.append(con.unsqueeze(0))

        decoder_hiddens_ = torch.cat(decoder_hiddens_, dim = 0)
        context_ = torch.cat(context_, dim = 0)

        if self.attention is not None:
            out, attn_score = self.attention(hidden, 
            								 encoder_outputs, 
            								 source_lengths)
        else:
            out = hidden
            attn_score = None

        context_vec = out
        out = self.dropout_f(out)

        # linear: hidden_size -> vocab_size
        deco_out = self.linear_layer(out)
        deco_out = self.log_softmax(deco_out)

        return out_vocab, context_vec, decoder_hiddens_, context_, attn_score


# CNNencoder
class CNNencoder(nn.Module):

    def __init__(self, 
                 vocab_size, 
                 embed_size, 
                 hidden_size, 
                 kernel_size, 
                 num_layers,
                 percent_dropout=0.3):
        
        super(CNNencoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, 
                                      self.embed_size, 
                                      padding_idx=0)
        
        self.dropout_f = nn.Dropout(percent_dropout)
        
        in_channels = self.embed_size
        
        self.conv = nn.Conv1d(in_channels, 
        					  self.hidden_size, 
        					  kernel_size, 
                              padding=kernel_size//2)
        
        # todo
        self.conv2 = nn.Conv1d(60, self.hidden_size, kernel_size,
        					   padding=kernel_size//2)
        
        self.ReLU = nn.ReLU()

    def forward(self, source_sentence):
        
        batch_size, seq_len = source_sentence.size()
        
        embeds_source = self.embedding(source_sentence)
        
        out = self.conv(embeds_source.transpose(1, 2)).transpose(1,2)
        out = self.ReLU(out)
        out = F.max_pool1d(out, kernel_size=5, stride=5)
        
        out = self.conv2(out.transpose(1, 2)).transpose(1,2)
        out = self.ReLU(out)
        out = torch.mean(out, dim=1).view(1, batch_size, self.hidden_size)
    
        return out


class LSTMencoder(nn.Module):

    def __init__(self, 
    			 input_size, 
    			 embed_size, 
    			 hidden_size,
    			 num_lstm_layers):

        super(LSTMencoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = Embedding(input_size, 
        						   self.embed_size, 
        						   padding_idx=0)

        self.dropout_ = nn.Dropout(p = 0.1)
        self.num_layers = num_lstm_layers

        self.lstm = LSTM(self.embed_size, self.hidden_size, 
        				 batch_first=True, bidirectional=True, 
        				 num_layers = self.num_layers, 
        				 dropout = 0.15)

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers*2,
                           batch_size,
                           self.hidden_size).to(device),\
               torch.zeros(self.num_layers*2,
                           batch_size,
                           self.hidden_size).to(device)

    def forward(self, 
    			encoder_inputs, 
    			source_lengths):

        sort_original_source = torch.sort(source_lengths, descending=True)[1]
        unsort_to_original_source = torch.sort(sort_original_source)[1]

        embeds_source = self.embedding(encoder_inputs)
        
        lstm_out = self.dropout_(embeds_source)

        batch_size, seq_len = embeds_source.size()

        hidden, context = self.initHidden(batch_size)
        sorted_output = lstm_out[sort_original_source]
        sorted_len = source_lengths[sort_original_source]

        packed_output = nn.utils.rnn.pack_padded_sequence(sorted_output, 
                                                          sorted_lengths, 
                                                          batch_first = True)

        packed_outs, (hiddden, context) = self.lstm(packed_output,(hidden, context))
        hidden = hidden[:,unsort_to_original_source,:]
        context = context[:,unsort_to_original_source,:]

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, 
        												padding_value=PAD_IDX, 
        												batch_first = True)
        # UNSORT OUTPUT
        lstm_out = lstm_out[unsort_to_original_source]
        hidden = hidden.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous().view(self.num_layers, batch_size, -1)
        context = context.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous().view(self.num_layers, batch_size, -1)

        return output, hidden, context





