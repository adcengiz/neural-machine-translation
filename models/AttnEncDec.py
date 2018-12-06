# from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# same as 1st model's RNN encoder except that works on one token at a time
# the different part is the attention decoder in model 2

class attnRNNencoder(nn.Module):
    def __init__(self,
                 vocab_size=len(zhen_zh_train_token2id), # for chinese
                 embedding_size=300,
                 percent_dropout=0.3, 
                 hidden_size=256,
                 num_gru_layers=4,
                 max_sentence_len=350):
        
        super(attnRNNencoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_gru_layers
        
        self.vocab_size = vocab_size
        self.embed_size = embedding_size
        self.dropout = percent_dropout
        self.embed_source = nn.Embedding(self.vocab_size,
                                         self.embed_size,
                                         padding_idx=0
                                        )
        
        self.max_sentence_len = max_sentence_len
        
        self.GRU = nn.GRU(self.embed_size, 
                          self.hidden_size, 
                          self.num_layers, 
                          batch_first=True, 
                          bidirectional=False)
        
        self.drop_out_function = nn.Dropout(self.dropout)
        
    def init_hidden(self, batch_size):
        
        hidden_ = torch.zeros(self.num_layers*self.num_directions, 
                             batch_size, self.hidden_size).to(device)
        return hidden_

    def forward(self, source_sentence, source_mask, source_lengths,
                time_step):
        """Returns source lengths to feed into the decoder, since we do not want
        the translation length to be above/below a certain treshold*source sentence length."""
        
        source_sentence = source_sentence.view(-1,1)
        # print ("source size = "+str(source_sentence.size()))
        # (batch_size, 1)
        
        sort_original_source = sorted(range(len(source_lengths)), 
                             key=lambda sentence: -source_lengths[sentence])
        unsort_to_original_source = sorted(range(len(source_lengths)), 
                             key=lambda sentence: sort_original_source[sentence])
        
        source_sentence = source_sentence[sort_original_source]
        _source_mask = source_mask[sort_original_source]
        source_lengths = source_lengths[sort_original_source]
        batch_size, seq_len_source = source_sentence.size()
        
        if self.GRU.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
            
        self.hidden_source = self.init_hidden(batch_size)
        # (self.num_layers*self.num_directions, batch_size, self.hidden_size)
        # (1, 32, 256)
        # https://pytorch.org/docs/stable/nn.html
        # print ("self hidden size. = "+str(self.hidden_source.size()))
        
        # If batch_first == True, then the input and output tensors are provided as 
        # (batch_size, seq_len, feature)
        # https://pytorch.org/docs/stable/nn.html
        # print ("seq len source = "+str(seq_len_source))
        
        source_sentence = source_sentence.unsqueeze(1)
        
        embeds_source = self.embed_source(source_sentence).view(batch_size, seq_len_source,
                                                               self.embed_size)
        
        # print ("embeds source size = "+str(embeds_source.size()))
        
        embeds_source = source_mask[:,time_step,:].unsqueeze(1)*embeds_source + \
                        (1-_source_mask[:,time_step,:].unsqueeze(1))*embeds_source.clone().detach()
        
        # print ("embeds source after mask size = "+str(embeds_source.size()))
        
        
#         embeds_source = torch.nn.utils.rnn.pack_padded_sequence(embeds_source, 
#                                                                 source_lengths, 
#                                                                 batch_first=True)
        
        gru_out_source, self.hidden_source = self.GRU(embeds_source, self.hidden_source)
        
        # print ("gru out source size = "+str(gru_out_source.size()))
        
        # print ("hidden source size = "+str(self.hidden_source.size()))
        # print ("gru out source size = "+str(gru_out_source.size()))
        
        # hidden source size = torch.Size([1, 32, 256])
        # gru out source size = torch.Size([32, 350, 256])
        
        # ref: pytorch documentation
        # hidden source : h_n of shape 
        # (num_layers * num_directions, batch_size, hidden_size)
        # print ("hidden source size = "+str(self.hidden_source.size()))
        
        # ref: pytorch documentation
        # Like output, the layers can be separated using 
        # h_n.view(num_layers, num_directions, batch_size, hidden_size)
        hidden_source = self.hidden_source.view(self.num_layers, self.num_directions, 
                                                batch_size, self.hidden_size)
        
        # print ("hidden source size = "+str(hidden_source.size()))
        # hidden source size = torch.Size([1, 1, 32, 256])
        
        # the following should print (1, 1, 32, 256) for this config
        # print ("hidden source size after view = "+str(hidden_source.size()))
        
        # get the mean along 0th axis (over layers)
        hidden_source = torch.mean(hidden_source, dim=0) ## mean instead of sum for source representation as suggested in the class
        # the following should print (1, 32, 256)
        # print ("hidden source size after mean = "+str(hidden_source.size()))
        
        if self.GRU.bidirectional:
            hidden_source = torch.cat([hidden_source[:,i,:] for i in range(self.num_directions)], dim=1)
            gru_out_source = gru_out_source
        else:
            hidden_source = hidden_source
            gru_out_source = gru_out_source
            
        # view before unsort
        hidden_source = hidden_source.view(batch_size, self.hidden_size)
        
        # the following should print (32, 256)
        # print("hidden source size before unsort = "+str(hidden_source.size()))
        # UNSORT HIDDEN
        hidden_source = hidden_source[unsort_to_original_source] ## back to original indices
        
#         gru_out_source, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out_source,
#                                                                   batch_first=True)
        
        ### UNSORT GRU OUT
        # get the mean for the GRU output (batch_size, output size, hidden_size)
        gru_out_source = gru_out_source.view(batch_size, seq_len_source, self.hidden_size)
        # gru_out_source = torch.mean(gru_out_source, dim=1).view(batch_size, 1, self.hidden_size)
        gru_out_source = gru_out_source[unsort_to_original_source]
        # print ("gru_out_source size = "+str(gru_out_source.size()))
        
        source_lengths = source_lengths[unsort_to_original_source]
        
        # here we return both hidden and out since we will pass both to
        # the attention decoder
        return hidden_source, gru_out_source, source_lengths


# from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

class AttnDecoderRNN(nn.Module):
    def __init__(self,
                 vocab_size=len(zhen_zh_train_token2id), 
                 embedding_size=300,
                 percent_dropout=0.3, 
                 hidden_size=256,
                 max_sentence_len=350, 
                 num_gru_layers=1):

        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = percent_dropout
        self.max_sentence_len = max_sentence_len
        self.num_layers = num_gru_layers
        self.embed_size = embedding_size
        
        self.embed_target = nn.Embedding(self.vocab_size,
                                         self.hidden_size,
                                         padding_idx=0
                                        )
        
        self.GRU = nn.GRU(self.hidden_size, 
                          self.hidden_size,
                          self.num_layers, 
                          batch_first=True, 
                          bidirectional=False)
        
        # we concat embeds with hidden before attention, thus the input size
        # of the linear attn layer is embed + hidden, and the output is hidden.
        self.attn = nn.Linear(self.hidden_size*2, 
                              self.max_sentence_len)
        
        # we combine embeds with attention applied (self.attn out) before attn_combine
        # so the input size of the linear attn_combine layer is embed_size + hidden_size 
        # 
        self.attn_combine = nn.Linear(self.hidden_size*2, 
                                      self.hidden_size)
        
        self.dropout = nn.Dropout(self.dropout)
        
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        
        self.log_softmax = nn.LogSoftmax()
        
    def forward(self,
                hidden, ## decoder_hidden = encoder_hidden at first time_step
                input_, # input (batch_size, seq_len = 1)
                encoder_outputs, # (encoder hidden and encoder out)
                target_lengths,
                target_mask,
                time_step):
        
        # input (batch_size, seq_len = 1)
        self.input = input_
        print ("input size. ="+str(self.input.size()))
        
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
            
        self.hidden = hidden.view(batch_size, 
                                  self.num_layers*self.num_directions, 
                                  self.hidden_size)
        
        self.input = self.input.unsqueeze(1)
        
        
        embeds_target = self.dropout(self.embed_target(self.input.long()))\
                                                                .view(batch_size,
                                                                      seq_len_target, -1)
        
        embeds_target = target_mask[:,time_step,:].unsqueeze(1)*embeds_target + \
                        (1-_target_mask[:,time_step,:].unsqueeze(1))*embeds_target.clone().detach()
        
        # print ("embeds target size = "+str(embeds_target.size()))

        attn_weights = F.softmax(self.attn(torch.cat((embeds_target, self.hidden), 2)), dim=2)
#         print ("attn_weights size = "+str(attn_weights.size()))
        
        # try for loop and bmm and see if these are the same 
        # print ("enc out size = "+str(encoder_outputs.size()))
        attn_applied = torch.zeros(batch_size, self.max_sentence_len, self.hidden_size)
        
        for i in range(batch_size):
#             print ("attn_weights[i] = "+str(attn_weights[i]))
#             print ("encoder_outputs[i] = "+str(encoder_outputs[i]))
            apply = torch.bmm(attn_weights[i].unsqueeze(0),
                              encoder_outputs[i].unsqueeze(0))
            
            attn_applied[i] = apply
        
        print ("attn_applied size = "+str(attn_applied.size()))
        print ("embeds target size = "+ str(embeds_target.size()))
#         print ("encoder outputs = "+str(encoder_outputs))
        print ("attn_applied[:,time_step,:] size = "+str(attn_applied[:,time_step,:].view(batch_size,
                                                                                          1, self.hidden_size).size()))

        output = torch.cat((embeds_target,
                            attn_applied[:,time_step,:].view(batch_size,1,
                                                             self.hidden_size)),2)
        
        output = self.attn_combine(output)
        
        output = F.relu(output)
        
        self.hidden = self.hidden.view(self.num_layers*self.num_directions,
                                       batch_size,
                                       self.hidden_size)

        output, self.hidden = self.GRU(output, self.hidden)
        
        self.hidden = self.hidden.view(batch_size,
                                       self.num_layers*self.num_directions,
                                       self.hidden_size)
        
        output = output[unsort_to_original_target]
        self.hidden = self.hidden[unsort_to_original_target]
        
        print ("output size = "+str(output.size()))
        print ("hidden size = "+str(self.hidden.size()))
        
        output = self.out(output)
        print ("out after linear size = "+str(output.size()))
        
        output = F.log_softmax(output, dim=2)
        print ("logsoft size = "+str(output.size()))
        
        return output, hidden, attn_weights
    
    
BATCH_SIZE = 32

class AttnTranslate(nn.Module):
    def __init__(self, encoder, decoder, use_teacher_forcing=False):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.use_teacher_forcing = use_teacher_forcing
        self.max_length = self.encoder.max_sentence_len
        
    def forward(self, source_sentence, target_sentence, 
                source_mask, target_mask, source_lengths,
                target_lengths):
        
        # following should print (batch_size, max_sentence_len) = (32, 350)
        # print ("target_sentence size = "+str(target_sentence.size()))
        
        # to hold previously decoded ys
        y_outputs = torch.zeros(batch_size, 
                                target_sentence.size(1), 
                                len(zhen_en_train_token2id)).to(device)
        
        encoder_outputs = torch.zeros(BATCH_SIZE,
                                      self.max_length, 
                                      self.encoder.hidden_size, 
                                      device=device)
        
        for i in range(self.max_length):
            #last hidden state of the encoder is the context
            encoder_hidden, encoder_output, source_lengths = self.encoder(source_sentence[:,i],
                                                                          source_mask,
                                                                          source_lengths,
                                                                          i) # i as time_step
            # doing what we want, uncomment the prints below to check
            # i-th time_step token of each sentence in batch is filled with the corresponding
            # encoder output
            encoder_outputs[:,i,:] = encoder_output.unsqueeze(1)[:,0,0]
            
            # print ("encoder_outputs[:,i,:] size = "+str(encoder_outputs[:,i,:].size()))
            # print ("encoder outputs size = "+str(encoder_outputs.size()))
            
            # print ("encoder outputs = "+str(encoder_outputs))
        
#         print ("encoder outputs size = "+str(encoder_outputs.size()))
#         print ("enc outs = "+str(encoder_outputs))

        # encoder hidden also used as the initial hidden state of the decoder
        decoder_hidden = encoder_hidden

        # decoder should start with SOS tokens 
        # ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        input_ = SOS_token*torch.ones(BATCH_SIZE,1).view(-1,1)
        
        # TODO
        # Obtain target tensor using convert_to_softmax (debug function first)  
        # target tensor -> batch_size, max_sent_len, vocab_size = 32, 350, vocab_size
#         target_tensor = torch.zeros()
        # append it to y_outputs

        target_length = target_sentence.size(1)

        if self.use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                # target tensor -> (batch_size, vocab_size) of t-th time step tokens 
                # from each sentence, converted to softmax (binary)
                target_tensor = convert_to_softmax(target_sentence[:,di],32)
#                 print ("target_tensor = "+str(target_tensor))
                print ("target tensor size = "+str(target_tensor.size()))
                
                print (breajs)
                # take ith token from each sentence in the batch, and convert it to 
                # softmax
                decoder_out, decoder_hidden, decoder_attention = self.decoder(
                    decoder_hidden, input_, encoder_outputs,
                    target_lengths, target_mask, di) # di as time_step
                
                # decoder out should be size (32, 1, vocab_size)
                
                loss += loss_function(decoder_out, target_tensor[:,di,:]) # slicing (whole batch, 
                                                                          #          token_index, vocab_size)
                decoder_input = target_sentence[:,di] # Teacher forcing
            
        else:
            # Without teacher forcing: use its own predictions as the next input
            # just like we did in the RNN encoder-decoder above
            for di in range(target_length):
                # target tensor -> (batch_size, vocab_size) of t-th time step tokens 
                # from each sentence, converted to softmax (binary)
                target_tensor = convert_to_softmax(target_sentence[:,di],32)
#                 print ("target_tensor = "+str(target_tensor))
                print ("target tensor size = "+str(target_tensor.size()))
                
                decoder_out, decoder_hidden, decoder_attention = self.decoder(
                    decoder_hidden, input_, encoder_outputs, target_lengths,
                    target_mask, di)
                
                token_out = torch.max(decoder_out.view(BATCH_SIZE,self.decoder.vocab_size),1)[1]
                input_ = token_out.view(-1,1)
                print ("decoder input size = "+str(input_.size()))
                
                print (breaks)

                loss += loss_function(decoder_out, target_tensor[di])
            
        for t in range(0, target_sentence.size(1)):
            
            decoder_out, decoder_hidden = self.decoder(decoder_hidden, # = gru_out_source - instead of encoded_source[0]
                                                 input_, # instead of target sentence up to t 
                                                 target_lengths,  # target lengths
                                                 target_mask,
                                                 t)
            
#             print ("decoder out size = "+str(decoder_out.size()))
            for s in range(batch_size):
                y_outputs[s,t] = decoder_out[s,0]



            
        return y_outputs

# chinese -> english
enc = attnRNNencoder(vocab_size=len(zhen_zh_train_token2id), # for chinese
                 embedding_size=300,
                 percent_dropout=0.3, 
                 hidden_size=256,
                 num_gru_layers=1)

dec = AttnDecoderRNN(vocab_size=len(zhen_en_train_token2id), # for chinese-english's english
                 embedding_size=300,
                 percent_dropout=0.3, 
                 hidden_size=256,
                 num_gru_layers=1)

BATCH_SIZE = 32
def train(encoder, decoder, loader=None, 
          criterion=torch.nn.NLLLoss(),
          optimizer = torch.optim.Adam([*enc.parameters()] + [*dec.parameters()], lr=lr),
          epoch=None, teacher_forcing=True):
    
    
    epoch_loss = 0
    
    for batch_idx, (source_sentence, source_mask, source_lengths, 
                    target_sentence, target_mask, target_lengths)\
    in enumerate(loader):
        
        source_sentence, source_mask = source_sentence.to(device), source_mask.to(device),  
        target_sentence, target_mask = target_sentence.to(device), target_mask.to(device),
        
        optimizer.zero_grad()
        
        # output softmax as generated by decoder 
        encoder_outputs = torch.zeros(BATCH_SIZE,
                                      encoder.max_sentence_len, 
                                      encoder.hidden_size, 
                                      device=device)
        
        max_length = torch.max(torch.from_numpy(source_lengths))
        
        for i in range(max_length):
            #last hidden state of the encoder is the context
            encoder_hidden, encoder_output, source_lengths = encoder(source_sentence[:,i],
                                                                          source_mask,
                                                                          source_lengths,
                                                                          i) # i as time_step
            # doing what we want, uncomment the prints below to check
            # i-th time_step token of each sentence in batch is filled with the corresponding
            # encoder output
            encoder_outputs[:,i,:] = encoder_output.unsqueeze(1)[:,0,0]
            
            # print ("encoder_outputs[:,i,:] size = "+str(encoder_outputs[:,i,:].size()))
            # print ("encoder outputs size = "+str(encoder_outputs.size()))
            
            # print ("encoder outputs = "+str(encoder_outputs))

        # encoder hidden also used as the initial hidden state of the decoder
        decoder_hidden = encoder_hidden

        # decoder should start with SOS tokens 
        # ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        input_ = SOS_token*torch.ones(BATCH_SIZE,1).view(-1,1)
        
        target_length = target_sentence.size(1)

        if teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_outputs = torch.zeros(BATCH_SIZE, torch.max(torch.from_numpy(target_lengths)), decoder.vocab_size)

            for di in range(target_length):
                # target tensor -> (batch_size, vocab_size) of t-th time step tokens 
                # from each sentence, converted to softmax (binary)
#                 target_tensor = convert_to_softmax(target_sentence[:,di],32)
                # print ("target_tensor = "+str(target_tensor))
                # print ("target tensor size = "+str(target_tensor.size()))
                
                # take ith token from each sentence in the batch, and convert it to 
                # softmax
                decoder_out, decoder_hidden, decoder_attention = decoder(
                    decoder_hidden, input_, encoder_outputs,
                    target_lengths, target_mask, di) # di as time_step
                
                decoder_input = target_sentence[:,di].view(-1,1) # Teacher forcing
                
                # decoder out should be size (32, 1, vocab_size)
                decoder_outputs[:,di,:] = decoder_out.view(BATCH_SIZE, decoder.vocab_size)
               
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
            # Without teacher forcing: use its own predictions as the next input
            # just like we did in the RNN encoder-decoder above

            decoder_outputs = torch.zeros(BATCH_SIZE, torch.max(torch.from_numpy(target_lengths)), decoder.vocab_size)
            
            for di in range(target_length):
                # target tensor -> (batch_size, vocab_size) of t-th time step tokens 
                # from each sentence, converted to softmax (binary)
                target_tensor = convert_to_softmax(target_sentence[:,di],32)
                # print ("target_tensor = "+str(target_tensor))
                
                decoder_out, decoder_hidden, decoder_attention = self.decoder(
                    decoder_hidden, input_, encoder_outputs, target_lengths,
                    target_mask, di)
                
                token_out = torch.max(decoder_out.view(BATCH_SIZE, decoder.vocab_size),1)[1]
                input_ = token_out.view(-1,1)
                
                decoder_outputs[:,di,:] = decoder_out.view(BATCH_SIZE, decoder.vocab_size)
               
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
        
    return loss

num_epochs = 150
lr = 1e-3
# batch_

loss_train = []

for epoch in range(num_epochs):
    print ("epoch = "+str(epoch))

    loss = train(enc, dec,
                 loader = zhen_train_loader,
                 optimizer = torch.optim.Adam([*enc.parameters()] + [*dec.parameters()], lr=lr, weight_decay=1e-6),
                 epoch = epoch, teacher_forcing=True)
    
#     loss_train.append(loss)
    
#     print (loss_train)