# glu from https://arxiv.org/pdf/1612.08083.pdf

# ENCODER

class CNNencoder(nn.Module):
    def __init__(self, 
                 embedding_size, # in channels
                 hidden_size, 
                 kernel_size, 
                 padding = 1,
                 stride = 2,
                 percent_dropout = 0.3,
                 vocab_size = len(zhen_zh_train.index2word),
                 max_sentence_len=350):
        
        super(CNNencoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.vocab_size = vocab_size
        self.stride = stride
        self.dropout = nn.Dropout(percent_dropout)
        self.max_sentence_len = max_sentence_len
        
        self.embedding = nn.Embedding(self.vocab_size, 
                                      self.embedding_size)
        
        self.conv1 = nn.Conv1d(self.embedding_size, self.hidden_size, 
                               kernel_size=self.kernel_size, padding=self.padding,
                               stride=self.stride)

        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, 
                               kernel_size=self.kernel_size, padding=self.padding,
                               stride = self.stride)
        
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(3, 1)
        self.maxpool_2 = nn.MaxPool1d(5, 2)
        
        self.sigmoid = nn.Sigmoid()
 

    def forward(self, input_):
        
        # input size = 1'e uydurmaya calis
        
        batch_size, seq_len = input_.size()
        
        embed = self.dropout(self.embedding(input_))
        # print ("embed size = "+str(embed.size()))
        # 32, 350, 300 check
        
        hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
        hidden = self.relu(hidden)
        hidden = self.maxpool_1(hidden.transpose(1,2)).transpose(1,2)
        
        # second conv layer
        hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
        hidden = self.relu(hidden)
        hidden = self.maxpool_2(hidden.transpose(1,2)).transpose(1,2)

        # print ("hidden size = "+str(hidden.size()))
        hidden = nn.functional.glu(hidden)
        
        # sum 
        hidden = torch.mean(hidden, 1).view(batch_size, 1, hidden.size(-1))
        # sigmoid
        hidden = self.sigmoid(hidden)
        
        return hidden

class RNNdecoder_CNN(nn.Module):
    def __init__(self,
                 vocab_size=len(zhen_en_train_token2id), # for chinese-english's english
                 embedding_size=300,
                 percent_dropout=0.3, 
                 hidden_size=512,
                 num_gru_layers=1,
                 max_sentence_len=300):
        
        super(RNNdecoder_CNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embedding_size
        self.dropout = percent_dropout
        self.max_sentence_len = max_sentence_len

        self.hidden_size = hidden_size
        self.num_layers = num_gru_layers
        
        self.GRU = nn.GRU(self.embed_size, 
                          self.hidden_size, 
                          self.num_layers, 
                          batch_first=True, 
                          bidirectional=False)
        
        self.ReLU = nn.ReLU
        
        self.drop_out_function = nn.Dropout(self.dropout)
        
        self.embed_target = nn.Embedding(self.vocab_size,
                                         self.embed_size, padding_idx=0)
        
        self.sigmoid = nn.Sigmoid()
        
        # *2 because we are concating hidden with embedding plus context
        self.linear_layer = nn.Linear(self.hidden_size*2, self.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)
        
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, 
                             batch_size, self.hidden_size).to(device)
        
        return hidden

    def forward(self,
                decoder_hidden, ## decoder_hidden = encoder_hidden at first time_step
                input_, # input
                target_lengths,
                target_mask,
                time_step):
        
        self.input = input_
        print ("input size = "+str(self.input.size()))
        
        sort_original_target = sorted(range(len(target_lengths)), 
                             key=lambda sentence: -target_lengths[sentence])
        unsort_to_original_target = sorted(range(len(target_lengths)), 
                             key=lambda sentence: sort_original_target[sentence])
        
        self.input = self.input[sort_original_target]
        _target_mask = target_mask[sort_original_target]
        target_lengths = target_lengths[sort_original_target]
        
        # seq_len_target is always 1 in the decoder since we are 
        # passing the tokens for only 1 time_step at a time
        batch_size, seq_len_target = self.input.size()
        
        if self.GRU.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        # hidden => initial hidden will be the same as the context
        # vector, which is the hidden_source tensor
        # then as we update the hidden state at each time step, this will be 
        # updated as well
        self.hidden = decoder_hidden.view(self.num_layers*self.num_directions,
                                          batch_size, self.hidden_size)
        
        # the following should print (1, 32, 256) for this config
        # print ("self.hidden size = "+str(self.hidden.size()))
        
        self.input = self.input.unsqueeze(1)
        
        embeds_target = self.drop_out_function(self.embed_target(self.input.long())).view(batch_size,
                                                                                   seq_len_target,
                                                                                   -1)
    
        embeds_target = target_mask[:,time_step,:].unsqueeze(1)*embeds_target + \
                        (1-_target_mask[:,time_step,:].unsqueeze(1))*embeds_target.clone().detach()


        gru_out_target, self.hidden = self.GRU(embeds_target.data.view(batch_size, 1, self.embed_size),
                                               self.hidden)
        
        # ref: pytorch documentation
        # hidden source : h_n of shape 
        # (num_layers * num_directions, batch_size, hidden_size)
        # the following should print (1, 32, 256) for this config
        # print ("hidden size after GRU = "+str(self.hidden.size()))


        hidden = self.hidden.view(self.num_layers, self.num_directions,
                                  batch_size, self.hidden_size)
        hidden = torch.sum(hidden, dim=0) # we don't divide here, just sum
        
        if self.GRU.bidirectional:
            # separate layers
            gru_out_target = gru_out_target.contiguous().view(seq_len_target,
                                                              batch_size,
                                                              self.num_directions,
                                                              self.hidden_size)
        else:
            gru_out_target = gru_out_target
        
#         print ("gru out size = "+str(gru_out_target.size()))
        
        # sum along sequence
        gru_out_target = torch.sum(gru_out_target, dim=1) # we don't divide here, just sum
        
        if self.GRU.bidirectional:
            hidden = torch.cat([hidden[:,i,:] for i in range(self.num_directions)], 
                               dim=0)
            gru_out_target = torch.cat([gru_out_target[:,i,:] for i in range(self.num_directions)], 
                                       dim=1)
        else:
            hidden = hidden.view(batch_size, 
                                 self.num_directions, self.hidden_size)
            gru_out_target = gru_out_target.view(batch_size,
                                                 self.num_directions, self.hidden_size)
        
        hidden = hidden[unsort_to_original_target] ## back to original indices
        gru_out_target = gru_out_target[unsort_to_original_target] ## back to original indices

        gru_out_target = self.sigmoid(gru_out_target)
        # concating embedding + context = gru_out_target with hidden
        out = torch.cat([gru_out_target,hidden], dim=2)
        
#         print ("out size after concat = "+str(out.size()))
        
        out = self.linear_layer(out)
        
        # softmax over vocabulary
        pred = self.log_softmax(out)

        return pred, hidden


# chinese -> english
enc = CNNencoder(300, # embed size
                 1024, # hidden size
                 3, # kernel size
                 padding = 1,
                 stride = 2,
                 percent_dropout = 0.3,
                 vocab_size = len(zhen_zh_train.index2word),
                 max_sentence_len=50)
    
dec = RNNdecoder_CNN()


# model = Translate(enc, dec).to(device)

loss_hist = []
# train

BATCH_SIZE = 32
def train(encoder, decoder, loader=zhen_train_loader,
          optimizer = torch.optim.Adam([*enc.parameters()] + [*dec.parameters()], lr=1e-4),
          epoch=None, teacher_forcing=False, criterion = torch.nn.NLLLoss()):


    optimizer.zero_grad()
    
    loss = 0
    
    for batch_idx, (source_sentence, source_mask, source_lengths, 
                    target_sentence, target_mask, target_lengths)\
                    in enumerate(loader):
        
        source_sentence, source_mask = source_sentence.to(device), source_mask.to(device) 
        target_sentence, target_mask = target_sentence.to(device), target_mask.to(device)
        
        encoder_hidden, source_lengths = encoder(source_sentence,
                                               source_mask,
                                               source_lengths)
        
        decoder_hidden = encoder_hidden.to(device)
        
        # decoder should start with SOS tokens 
        # ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        input_ = SOS_token*torch.ones(BATCH_SIZE,1).view(-1,1).to(device)
        
        
        if teacher_forcing:
            
            decoder_outputs = torch.zeros(BATCH_SIZE, torch.max(torch.from_numpy(target_lengths)), decoder.vocab_size)
            
            for t in range(0, target_sentence.size(1)):

                decoder_out, decoder_hidden = decoder(decoder_hidden, # = gru_out_source - instead of encoded_source[0]
                                                     input_, # instead of target sentence up to t 
                                                     target_lengths,  # target lengths
                                                     target_mask,
                                                     t)
                
                decoder_outputs[:,t,:] = decoder_out.view(BATCH_SIZE, decoder.vocab_size)
            
                input_ = target_sentence[:,t].view(-1,1)
               
            loss_tensor = torch.zeros(BATCH_SIZE, 1)
            
            for i in range(BATCH_SIZE):
                loss_tensor[i] = torch.sum(criterion(decoder_outputs[i], target_sentence[i]))/torch.from_numpy(target_lengths).float()[i]
                
                
            loss = torch.sum(loss_tensor)
            loss.backward(retain_graph = True)
#             loss += criterion(F.sigmoid(decoder_out), target_tokens)
            

            print ("loss = "+str('{0:.16f}'.format(loss)))
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

            optimizer.step()
            
        
        else:
            
            decoder_outputs = torch.zeros(BATCH_SIZE, torch.max(torch.from_numpy(target_lengths)), decoder.vocab_size)
            
            for t in range(0, target_sentence.size(1)):

                decoder_out, decoder_hidden = decoder(decoder_hidden, # = gru_out_source - instead of encoded_source[0]
                                                     input_, # instead of target sentence up to t 
                                                     target_lengths,  # target lengths
                                                     target_mask,
                                                     t)

                decoder_outputs[:,t,:] = decoder_out.view(BATCH_SIZE, decoder.vocab_size)
                input_ = decoder_out.topk(1)[1].view(BATCH_SIZE, 1)
#                 print ("input_ = "+ str(input_))
#                 print ("input_ size = "+str(input_.size()))

#                 loss += criterion(F.sigmoid(decoder_out), target_tokens)

            loss_tensor = torch.zeros(BATCH_SIZE, 1)
    
            for i in range(BATCH_SIZE):
                loss_tensor[i] = torch.sum(criterion(decoder_outputs[i], target_sentence[i]))/torch.from_numpy(target_lengths).float()[i]
                
            loss = torch.sum(loss_tensor)/BATCH_SIZE
            loss.backward(retain_graph = True)
#             loss += criterion(F.sigmoid(decoder_out), target_tokens)
            

            print ("loss = "+str('{0:.16f}'.format(loss)))
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

            optimizer.step()
        
    torch.save(encoder.state_dict(), "rnn_encoder_state_dict")
    torch.save(decoder.state_dict(), "rnn_decoder_state_dict")
            
    return loss
        


