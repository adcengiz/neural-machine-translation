import numpy as np
import copy
from sacreBLEU.sacreBLEU import corpus_bleu

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


# :translate_rnn:  trains an encoder-decoder rnn pair given the encoder model, the decoder model, source sentence,
#                  target sentence, and the source lengths
# :translate_cnn:  trains an encoder-decoder pair of CNN + RNN, given the encoder model, the decoder model, source sentence,
#                  target sentence, and the source lengths
# :translate_attn: trains the attention-based RNN encoder-decoer (LSTMencoder + AttnDecoderRNN from encoder_decoder_models.py)
#                  given  the encoder model, the decoder model, source sentence,
#                  target sentence, source lengths, and target lengths
# :eval_:          for everything besides training!
# :id2text_:       converts index outputs to word strings
# :BeamSearch:     conducts beam search given a trained encoder, a trained decoder, and validation/test data


# encoder_model, decoder_model = trained_encoder, trained_decoder

PAD_IDX = 0
SOS_token = 1
UNK_IDX = 2
EOS_token = 3

def translate_rnn(encoder_model,
                  decoder_model,
                  source_sentence,
                  target_sentence,
                  source_lengths):
    
    use_teacher_forcing = True if random.random() < 0.6 else False
    
    batch_size = source_sentence.size(0)
    encoder_hidden = encoder_model.init_hidden(batch_size)
    
    encoder_hidden, encoder_output = encoder_model(source_sentence,
                                                   source_lengths,
                                                   encoder_hidden)
    
    decoder_hidden = encoder_hidden
    
    decoder_input = torch.FloatTensor([[SOS_token]]*batch_size).to(device)

    if use_teacher_forcing:
        
        decoder_out = []
         
        for time_step in range(MAX_SENTENCE_LENGTH):
            
            decoder_output, decoder_hidden = decoder_model(decoder_input,
                                                           decoder_hidden,
                                                           encoder_outputs=None)
            decoder_out.append(decoder_output.unsqueeze(-1))
            decoder_input = target_sentence[:,time_step].view(-1,1)
            
        decoder_out = torch.cat(decoder_out,
                                dim=-1)
    else:
        
        decoder_out = []
        for time_step in range(MAX_SENTENCE_LENGTH):
            
            decoder_output, decoder_hidden = decoder_model(decoder_input,
                                                           decoder_hidden,
                                                           encoder_output)
            
            decoder_out.append(decoder_output.unsqueeze(-1))
            top_scores, top_indices = decoder_output.topk(1)
            decoder_input = top_indices.squeeze().detach().view(-1,1)
            
        decoder_out = torch.cat(decoder_out,
                                dim=-1)
        
    return decoder_out, decoder_hidden

def translate_cnn(encoder_model,
                  decoder_model,
                  source_sentence,
                  target_sentence,
                  source_lengths):
    
    use_teacher_forcing = True if random.random() < 0.6 else False
    
    batch_size = source_sentence.size(0)
    
    encoder_hidden = encoder_model(source_sentence)
    
    decoder_hidden = encoder_hidden
    
    decoder_input = torch.tensor([[SOS_token]]*batch_size).to(device)

    if use_teacher_forcing:
        
        decoder_out = []
         
        for time_step in range(MAX_SENTENCE_LENGTH):
            
            decoder_output, decoder_hidden = decoder_model(decoder_input,
                                                           decoder_hidden,
                                                           encoder_outputs=None)
            decoder_out.append(decoder_output.unsqueeze(-1))
            decoder_input = target_sentence[:,time_step].view(-1,1)
            
        decoder_out = torch.cat(decoder_out,
                                dim=-1)
    else:
        
        decoder_out = []
        for time_step in range(MAX_SENTENCE_LENGTH):
            
            decoder_output, decoder_hidden = decoder_model(decoder_input,
                                                           decoder_hidden,
                                                           encoder_output)
            
            decoder_out.append(decoder_output.unsqueeze(-1))
            top_scores, top_indices = decoder_output.topk(1)
            decoder_input = top_indices.squeeze().detach().view(-1,1)
            
        decoder_out = torch.cat(decoder_out,
                                dim=-1)
        
    return decoder_out, decoder_hidden

def translate_attn(encoder_model,decoder_model,
                   source_sentence, target_sentence,
                   source_lengths, target_lengths,
                   val=False):
    
    if val == False:
        
        teacher_forcing = True if random.random() < 0.6 else False

        batch_size, seq_len_source = source_sentence.size()
        
        encoder_out, encoder_hidden, encoder_context = encoder_model(source_sentence, source_lengths)
        
        max_source_length = max(source_lengths).item()
        max_target_length = max(target_lengths).item()
        
        prev_hiddens = encoder_hidden
        prev_context = encoder_context
        
        prev_ys = torch.zeros((batch_size, encoder_out.size(-1))).to(device)
        
        # decoder should start with SOS tokens at the first timestep
        decoder_input = torch.tensor([[SOS_token]]*batch_size).to(device)
        
        if teacher_forcing:
            
            decoder_out = []
            
            for time_step in range(max_target_length):
                
                out_, prev_ys, prev_hiddens,\
                prev_context, attn_score = decoder_model(decoder_input,
                                                         prev_ys,
                                                         prev_hiddens,
                                                         prev_context,
                                                         encoder_out,
                                                         source_lengths)

                decoder_out.append(out_.unsqueeze(-1))
                decoder_input = target_sentence[:,time_step].view(-1,1)
                
            decoder_out = torch.cat(decoder_out,
                                    dim=-1)

        else:
            
            decoder_out = []
            
            for time_step in range(max_target_length):
                
                out_, prev_ys, prev_hiddens,\
                prev_context, attn_score = decoder_model(decoder_input,
                                                         prev_ys,
                                                         prev_hiddens,
                                                         prev_context, 
                                                         encoder_out,
                                                         source_lengths)
                
                decoder_out.append(out_.unsqueeze(-1))
                top_scores, top_indices = out_.topk(1)
                decoder_input = top_indices.squeeze().detach().view(-1,1)

            decoder_out = torch.cat(decoder_out,
                                    dim=-1)
        return decoder_out
    
    else: # Val
        
        encoder_model.eval()
        decoder_model.eval()
        batch_size, seq_len_source = source_sentence.size()
        
        encoder_out, encoder_hidden, encoder_context = encoder_model(source_sentence, 
                                                                     source_lengths)
        max_source_length = max(source_lengths).item()
        max_target_length = max(target_lengths).item()
        
        prev_hiddens = encoder_hidden
        prev_context = encoder_context
        
        prev_ys = torch.zeros((batch_size, encoder_out.size(-1))).to(device)
        
        # SOS
        decoder_input = torch.tensor([[SOS_token]]*batch_size).to(device)
        
        decoder_out = []
        
        for i in range(max_target_length):
            
            out_, prev_ys, prev_hiddens, \
            prev_context, attn_score = decoder_model(decoder_input,
                                                     prev_ys,
                                                     prev_hiddens,
                                                     prev_context, 
                                                     encoder_out,
                                                     source_lengths)
            
            decoder_out.append(out_.unsqueeze(-1))
            top_scores, top_indices = out_.topk(1)
            decoder_input = top_indices.squeeze().detach().view(-1,1)

        decoder_out = torch.cat(decoder_out,dim=-1)
        
        return decoder_out

def eval_(encoder, 
          decoder, 
          val_dataloader, 
          vien_en_, # change with zhen_en_ for chinese -> english
          m_type):
    
    encoder.eval()
    decoder.eval()
    
    pred_corpus = []
    ref_corpus = []

    for data in val_dataloader:
        
        encoder_input = data[0].to(device)
        source_lengths = data[2].to(device)
        
        batch_size, seq_len = encoder_input.size()[:2]
        
        encoder_out, encoder_hidden, encoder_context = encoder(encoder_input,
                                                               source_lengths)
        max_source_length = max(source_lengths).item()
        
        prev_hiddens = encoder_hidden
        prev_context = encoder_context
        decoder_input = torch.tensor([[SOS_token]]*batch_size).to(device)
        prev_output = torch.zeros((batch_size, encoder_out.size(-1))).to(device)
        
        decoder_out = []
        
        for i in range(seq_len*2):
            
            out_, prev_output, prev_hiddens,\
            prev_context, attention_score = decoder_model(decoder_input,
                                                          prev_output,
                                                          prev_hiddens,
                                                          prev_context, 
                                                          encoder_out,
                                                          source_lengths)
            top_scores, top_indices = out_.topk(1)
            decoder_out.append(top_indices.item())
            decoder_input = top_indices.squeeze().detach().view(-1,1)
            
            if top_indices.item() == EOS_token:
                break
        
        ref_corpus.append(data[-1])
        
        pred_sent = id2text_(decoder_out,vien_en_)
        pred_corpus.append(pred_sent)

    print ("true corpus", ref_corpus[:5])
    print ("pred corpus", pred_corpus[:5])
    
    # import above: from sacreBLEU.sacreBLEU import corpus_bleu
    score = corpus_bleu((" ").join(pred_corpus),
                        (" ").join(ref_corpus))[0]
    return score

def train(encoder_optimizer,
          decoder_optimizer, 
          encoder_model, decoder_model, 
          loss_function,
          data_loader, 
          en_lang, # "vien_en_" for vietnamese -> eng, "zhen_en_" for chinese -> eng
          num_epochs=10, val_interval=1, rm = 0.8, 
          enc_scheduler=None, 
          dec_scheduler=None):

    mode_list = ["train","val_train"] # val_train, val every val_interval train epochs
    loss_hist = {"train": [], "val_train": []}
    BLEU_hist = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print ("epoch", epoch)

        for ex, mode in enumerate(mode_list):
            
            start = time.time()
            total = 0
            top1_correct = 0
            running_loss = 0
            running_total = 0
            
            if mode == "train":
                encoder.train()
                decoder.train()
                
            elif mode == "val_train":
                encoder.eval()
                decoder.eval()
            else:
                raise ValueError
                
            for data in data_loader[mode]:
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_input, decoder_input = data[0].to(device), data[1].to(device)
                source_lengths, target_lengths = data[2].to(device), data[3].to(device)

                if mode == "val_train":                
                    output = encode_decode_attn(encoder_model, decoder_model,
                                                encoder_input, decoder_input,
                                                source_lengths, target_lengths,
                                                rand_num=rm, val=True)
                else:
                    output = encode_decode_attn(encoder_model, decoder_model,
                                                encoder_input, decoder_input,
                                                source_lengths, target_lengths,
                                                rand_num=rm, val=False)
                    
                loss = loss_function(output.float(), 
                                     decoder_input[:,:output.size(-1)].long())
                
                batch = decoder_input.size(0)
                
                running_loss += loss.item()*batch
                
                total += batch
                
                if mode == "train":
                    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.15)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.15)
                    
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    
            epoch_loss = running_loss / total 
            loss_hist[mode].append(epoch_loss)
            print("epoch {} {} loss = {}, time = {}".format(epoch, mode, epoch_loss,
                                                                           time.time() - start))
        if (enc_scheduler is not None) and (dec_scheduler is not None):
            enc_scheduler.step(epoch_loss)
            dec_scheduler.step(epoch_loss)
            
        if epoch % val_interval == 0:
            val_bleu_score = eval_(encoder_model, decoder_model, data_loader["val"], en_lang)
            BLEU_hist["val"].append(val_bleu_score)
            print("validation BLEU = ", val_bleu_score)

    return encoder_model, decoder_model, loss_hist, BLEU_hist
    

def train_cnn(encoder_optimizer,
              decoder_optimizer,
              encoder_model, decoder_model,
              loss_function,
              data_loader,
              en_lang, # "vien_en_" for vietnamese -> eng, "zhen_en_" for chinese -> eng
              num_epochs=10, val_interval=1, rm = 0.8, 
              enc_scheduler=None, 
              dec_scheduler=None):

    mode_list = ["train","val_train"] # val_train, val every val_interval train epochs
    loss_hist = {"train": [], "val_train": []}
    BLEU_hist = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print ("epoch", epoch)

        for ex, mode in enumerate(mode_list):
            
            start = time.time()
            total = 0
            top1_correct = 0
            running_loss = 0
            running_total = 0
            
            if mode == "train":
                encoder.train()
                decoder.train()
                
            elif mode == "val_train":
                encoder.eval()
                decoder.eval()
            else:
                raise ValueError
                
            for data in data_loader[mode]:
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_input, decoder_input = data[0].to(device), data[1].to(device)
                source_lengths, target_lengths = data[2].to(device), data[3].to(device)

                if mode == "val_train":                
                    
                    output = encode_decode_cnn(encoder_model, decoder_model,
                                               encoder_input, decoder_input,
                                               source_lengths)
                else:
                    output = encode_decode_cnn(encoder_model, decoder_model,
                                               encoder_input, decoder_input,
                                               source_lengths)
                    
                loss = loss_function(output.float(), 
                                     decoder_input[:,:output.size(-1)].long())
                
                batch = decoder_input.size(0)
                
                running_loss += loss.item()*batch
                
                total += batch
                
                if mode == "train":
                    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.15)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.15)
                    
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    
            epoch_loss = running_loss / total 
            loss_hist[mode].append(epoch_loss)
            print("epoch {} {} loss = {}, time = {}".format(epoch, mode, epoch_loss,
                                                                           time.time() - start))
        if (enc_scheduler is not None) and (dec_scheduler is not None):
            enc_scheduler.step(epoch_loss)
            dec_scheduler.step(epoch_loss)
            
        if epoch % val_interval == 0:
            val_bleu_score = eval_(encoder_model, decoder_model, data_loader["val"], en_lang)
            BLEU_hist["val"].append(val_bleu_score)
            print("validation BLEU = ", val_bleu_score)

    return encoder_model, decoder_model, loss_hist, BLEU_hist



def id2text_(index_list, en_):
    """Params
    :index_tensor: final output hypothesis of the BeamSearch. A list of tokens.
    :en_: the English language object of the validation set we are using.
          - zhen_en_ for Chinese -> English
          - vien_en_ for Vietnamese -> English
    """
    strings = []

    if type(index_list) == list:

        for i in index_list:
            if i not in set([EOS_token]):
                strings.append(en_.index2word[i])
    else:

        for i in index_list:
            if i.item() not in [*set([PAD_IDX, SOS_token, EOS_token])]:
                strings.append(en_.index2word[i.item()])

    return (" ").join(strings)


# BeamSearch
def BeamSearch(encoder_model, 
               decoder_model, 
               val_loader,
               en_, # vien_en_ or zhen_en_
               beam_size, 
               device = "cuda"):
    
    """
    Params
    :encoder_model: Trained RNN or CNN encoder model.
    :decoder_model: Trained RNN or RNN w/attention decoder model.
    :val_loader: Validation dataloader object.
    :en_: The English language (target) object of the passed val_loader language.
    :beam_size: The function selects beam_size-many hypotheses with the highest log prob
                at each timestep.

    Returns
    :4-gram precision BLEU score for the given validation data.
    """

    encoder_model.eval()
    decoder_model.eval()

    model_corpus = []
    reference_corpus = []

    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)

    running_loss = 0
    running_total = 0

    # iterate over val_loader until computing the final
    # corpus-level BLEU score
    for sentence_pair in val_loader:
        # encoder input = source sentence
        encoder_input = sentence_pair[0].to(device)
        source_lengths = sentence_pair[2].to(device)

        seqlen_ = torch.max(source_lengths) # max_len
        batch_size, seq_len = encoder_input.size()[:2]
        
        encoder_out, encoder_hidden, encoder_context = encoder_model(encoder_input,
                                                                     source_lengths)
        
        prev_hiddens, prev_context = encoder_hidden, encoder_context

        # first input to the decoder should be SOS tokens (1)
        decoder_input = torch.tensor([[SOS_token]]*batch_size).to(device)

        prev_output = torch.zeros((batch_size, 
                                   encoder_out.size(-1))).to(device)

        decoder_input_list = [None]*beam_size

        end_beam = [False]*beam_size

        # init beam scores - batch_size x beam_size
        beam_scores = torch.zeros((batch_size,beam_size)).to(device)

        decoder_out_list = [[]]*beam_size

        for t in range(seq_len+20):

            if t == 0:

                outs, prev_output, prev_hiddens, \
                prev_context, attn_score = decoder_model(decoder_input,
                                                              prev_output,
                                                              prev_hiddens,
                                                              prev_context, 
                                                              en_out,
                                                              source_lengths)
                
                # get top beam_size-many scores and their indices
                top_scores, top_indices = outs.topk(beam_size)
                out_s, vocab_size = outs.size()

                prev_out_list = [prev_output]*beam_size
                prev_hidden_list = [prev_hiddens]*beam_size
                prev_context_list = [prev_context]*beam_size

                for beam_i in range(beam_size):

                    beam_scores[0][beam_i] = top_scores[0][beam_i].item()
                    decoder_input_list[beam_i] = top_indices[0][beam_i].squeeze().detach().\
                                                                                    view(-1,1)
                    decoder_out_list[beam_i].append(top_indices[0][beam_i].item())

                    if top_indices[0][beam_i].item() == EOS_token:
                        end_beam[beam_i] = True

            else:

                out_t, hidden_t, context_t, hold_beam = beam_size*[None], beam_size*[None], \
                                                        beam_size*[None], beam_size*[None]
                
                prev_ys = copy.deepcopy(decoder_out_list)

                for beam_i in [*range(beam_size)]:

                    if not end_beam[beam_i]:

                        hold_beam[beam_i], out_t[beam_i], hidden_t[beam_i], \
                        context_t[beam_i], attn_score = decoder_model(decoder_input_list[beam_i],
                                                                        prev_out_list[beam_i],
                                                                        prev_hidden_list[beam_i],
                                                                        prev_context_list[beam_i],
                                                                        encoder_out,
                                                                        source_lengths)

                        hold_beam[beam_i] = hold_beam[beam_i] + beam_scores[0][beam_i]

                    if end_beam[beam_i]:

                        hold_beam[beam_i] = torch.zeros(out_s, vocab_size).fill_(-np.inf).to(device)

                hold_beam = torch.cat(hold_beam, dim=1)
                top_scores, top_indices = hold_beam.topk(beam_size)

                hidden_id = top_indices//vocab_size
                top_indices_ = top_indices%vocab_size

                for beam_i in range(beam_size):

                    if not end_beam[beam_i]:

                        beam_scores[0][beam_i] = top_scores[0][beam_i].item()
                        list_decoder_input[beam_i] = top_indices_[0][beam_i].squeeze().detach().view(-1,1)
                        decoder_out_list[beam_i] = copy.deepcopy(prev_ys[hidden_id[0][beam_i]])
                        decoder_out_list[beam_i].append(top_indices_[0][beam_i].item())

                        # <EOS>
                        if top_indices_[0][beam_i].item() == EOS_token:
                            end_beam[beam_i] = True

                        else:
                            prev_out_list[beam_i] = out_t[hidden_id[0][beam_i]]
                            prev_context_list[beam_i] = context_t[hidden_id[0][beam_i]]
                            prev_hidden_list[beam_i] = hidden_t[hidden_id[0][beam_i]]
                            
                # all batch <EOS>
                if all(end_beam):
                    break

        max_score_id = np.argmax(beam_scores)

        decoder_out = decoder_out_list[max_score_id]

        reference_corpus.append(sentence_pair[-1]) # true/reference sentence
        pred_sentence = id2text_(decoder_out, en_) # predicted sentence
        model_corpus.append(pred_sentence)
    
    # import above - from sacreBLEU.sacreBLEU import corpus_bleu
    # WARNING: Do not forget to join the lists before computing BLEU score!
    # Otherwise your BLEU score will be far below the true one.
    bleu_score = corpus_bleu((" ").join(model_corpus),
                             (" ").join(reference_corpus))[0]

    print ("BLEU score calculated on the validation set is ", score)

    return bleu_score

def seq_mask(seq_len,
             max_len=None):

    if max_len is None:
        max_len = torch.max(seq_len).item()

    batch_size = seq_len.size(0)
    
    range_ = torch.arange(0, max_len).long()
    seq_range_unsq = range_.unsqueeze(0).repeat([batch_size,1])
    seq_range_unsq = seq_range_unsq.to(device)
    seq_length_unsq = (seq_len.unsqueeze(1).expand_as(seq_range_unsq))
    
    mask = (seq_range_unsq < seq_length_unsq).float()

    return mask



