# from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# same as 1st model's RNN encoder
# the different part is the attention decoder in model 2

class RNNencoder(nn.Module):
    def __init__(self,
                 input_size
                 hidden_size=256,
                 num_gru_layers=1):
        
        super(RNNencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_gru_layers
        
        self.vocab_size = vocab_size
        self.embed_size = embedding_size
        self.dropout = percent_dropout
        self.embed_source = nn.Embedding(self.input_size,
                                         self.embed_size,
                                         padding_idx=0
                                        )
        
        self.max_sentence_len = max_sentence_len
        
        self.GRU = nn.GRU(self.hidden_size, # instead of embed_size, using only the hidden
                          self.hidden_size, 
                          self.num_layers, 
                          batch_first=True, # always keep True
                          bidirectional=False)
        
        self.drop_out_function = nn.Dropout(0.1)
        
    def init_hidden(self, batch_size):
        
        hidden_ = torch.zeros(self.num_layers*self.num_directions, 
                             batch_size, 
                             self.hidden_size).to(device)
        return hidden_

    def forward(self, source_sentence, source_lengths, hidden):
        
        sort_original_source = sorted(range(len(source_lengths)), 
                             key=lambda sentence: -source_lengths[sentence])
        unsort_to_original_source = sorted(range(len(source_lengths)), 
                             key=lambda sentence: sort_original_source[sentence])
        
        source_sentence = source_sentence[sort_original_source]
        _source_mask = source_mask[sort_original_source]
        source_lengths = source_lengths[sort_original_source]
        batch_size, seq_len_source = source_sentence.size()
        
        # init hidden
        if self.GRU.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        # not initializing hidden inside the encoder, instead we passed this
        # into the translator encode_decode_rnn
        # self.hidden_source = self.init_hidden(batch_size)
        
        # If batch_first == True, then the input and output tensors are provided as 
        # (batch_size, seq_len, feature)
        # https://pytorch.org/docs/stable/nn.html
        embeds_source = self.embed_source(source_sentence)

        # not using pretrained, no need to mask for unknowns
        # embeds_source = source_mask*embeds_source + (1-_source_mask)*embeds_source.clone().detach()
        
        
        embeds_source = torch.nn.utils.rnn.pack_padded_sequence(embeds_source, 
                                                                source_lengths, 
                                                                batch_first=True)

        output = embeds_source
        
        output, hidden = self.GRU(output, hidden)

        # ref: pytorch doc
        # hidden source : h_n of shape 
        # (num_layers * num_directions, batch_size, hidden_size)
        # print ("hidden source size = "+str(self.hidden_source.size()))
        
        if self.GRU.bidirectional:
            hidden = torch.cat([hidden[:,i,:] for i in range(self.num_directions)], dim=1)
            output = output
        else:
            hidden = hidden
            output = output
            
        # view before unsort
        hidden = hidden.view(batch_size, self.hidden_size)
        
        # the following should print (32, 256)
        # print("hidden source size before unsort = "+str(hidden_source.size()))
        # UNSORT HIDDEN
        hidden = hidden[unsort_to_original_source] ## back to original indices
        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # UNSORT GRU OUT
        output = output[unsort_to_original_source]

        
        # here we return both hidden and out since we will pass both to
        # the attention decoder later
        return hidden, output


class RNNdecoder(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 vocab_size=None,
                 percent_dropout=0.1) # for chinese-english's english):
        
        super(RNNdecoder, self).__init__()
        self.vocab_size = vocab_size
        # self.embed_size = embedding_size
        self.dropout = percent_dropout
        self.drop_out_function = nn.Dropout(self.dropout)

        self.hidden_size = hidden_size
        
        self.GRU = nn.GRU(self.hidden_size, 
                          self.hidden_size, 
                          1, 
                          batch_first=True, 
                          bidirectional=False)
        
        # self.GRUcell = nn.GRUCell(self.embed_size, 
        #                   self.hidden_size)
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        
        self.sigmoid = nn.Sigmoid()
        
        self.linear_layer = nn.Linear(self.hidden_size, self.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=0)
        
    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return hidden

    def forward(self,
                input_, # input
                decoder_hidden): ## decoder_hidden = encoder_hidden at first time_step
        
        batch_size = input_.size(0)
        output = self.embedding(input_)
        output = self.drop_out_function(output)
        # print ("input size = "+str(input.size()))
        
        # sort_original_target = sorted(range(len(target_lengths)), 
        #                      key=lambda sentence: -target_lengths[sentence])
        # unsort_to_original_target = sorted(range(len(target_lengths)), 
        #                      key=lambda sentence: sort_original_target[sentence])
        
        # seq_len_target is always 1 in the decoder since we are 
        # passing the tokens for only 1 time_step at a time
        batch_size, seq_len_target = input_.size()
        
        if self.GRU.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        # hidden => initial hidden will be the same as the context
        # vector, which is the hidden_source tensor
        # then as we update the hidden state at each time step, this will be 
        # updated as well

        # cat_out = torch.cat((output, decoder_hidden), 2)
        
        # the following should print (1, 32, 256) for this config
        # print ("self.hidden size = "+str(self.hidden.size()))
        
        output, decoder_hidden = self.GRU(output, decoder_hidden)

        output = self.linear_layer(output.squeeze(dim=1))

        output = self.log_softmax(output)

        return output, decoder_hidden


# model = Translate(enc, dec).to(device)


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


def encode_decode_rnn(encoder,
                      decoder,
                      data_source,
                      data_target,
                      source_lengths):
    
    use_teacher_forcing = True if random.random() < 0.6 else False
    
    batch_size = data_source.size(0)
    encoder_hidden = encoder.init_hidden(batch_size)
    
    encoder_hidden, encoder_output = encoder(data_source,
                                          source_lengths,
                                          encoder_hidden)
    
    decoder_hidden = encoder_hidden
    
    decoder_input = torch.tensor([[SOS_token]]*batch_size).to(device)

    if use_teacher_forcing:
        
        d_out = []
         
        for i in range(MAX_SENTENCE_LENGTH):
            
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            d_out.append(decoder_output.unsqueeze(-1))
            decoder_input = data_target[:,i].view(-1,1)
            
        d_hidden = decoder_hidden
        d_out = torch.cat(d_out,dim=-1)
    else:
        d_out = []
        for i in range(MAX_SENTENCE_LENGTH):
            
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            d_out.append(decoder_output.unsqueeze(-1))
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(-1,1)
            
        d_hidden = decoder_hidden
        d_out = torch.cat(d_out,dim=-1)
        
    return d_out, d_hidden



from sacreBLEU.sacreBLEU import corpus_bleu

def validate_model(encoder, decoder, 
                   dataloader, 
                   loss_fun, 
                   vien_en_): # if source lang == Chinese -> zhen_en_
    # validation pass - no teacher forcing
    encoder.train(False)
    decoder.train(False)
    pred_corpus = []
    true_corpus = []
    running_loss = 0
    running_total = 0

    for data in dataloader:

        encoder_i = data[0].to(device) # encoder input - source sentence
        decoder_i = data[1].to(device) # decoder input - target sentence
        source_lengths = data[2].to(device)

        bs,sl = encoder_i.size()[:2]
        out, hidden = encode_decode_rnn(encoder,decoder,
                                        encoder_i,decoder_i, 
                                        source_lengths)

        loss = loss_fun(out.float(), decoder_i.long())
        running_loss += loss.item()*bs
        running_total += bs
        pred = torch.max(out,dim = 1)[1]

        for t,p in zip(data[1],pred):
            t,p = out_token_2_string(t,lang_en), out_token_2_string(p,lang_en)
            true_corpus.append(t)
            pred_corpus.append(p)

    score = corpus_bleu(pred_corpus,[true_corpus],lowercase=True)[0]
    return running_loss/running_total, score

def train_model(encoder_optimizer,
                decoder_optimizer, 
                encoder, decoder, 
                dataloader,
                loss_function, 
                num_epochs=60):
    
    best_score = 0
    best_au = 0
    loss_hist = {"train": [], "val": []}
    acc_hist = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print ("epoch", epoch)

        for ex, phase in enumerate(["train"]):

            start = time.time()
            total = 0
            top1_correct = 0
            running_loss = 0
            running_total = 0

            if phase == "train":
                encoder.train(True)
                decoder.train(True)

            else:
                encoder.train(False)
                decoder.train(False)
                
            for data in dataloader[phase]:
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_i = data[0].to(device)
                decoder_i = data[1].to(device)
                source_lengths = data[2].to(device)
                                
                out, hidden = encode_decode_rnn(encoder, decoder, 
                                                encoder_i, decoder_i, 
                                                source_lengths)
                
                loss = loss_function(out.float(), decoder_i.long())

                N = decoder_i.size(0)

                running_loss += loss.item() * N
                
                total += N

                if phase == "train":
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    
            loss, score = validate_model(encoder, decoder, 
                                         dataloader["val"], # vien_loader if lang == vien_en_
                                         loss_function, # nll
                                         vien_en_) # target_language

            print("Validation Loss = ", loss)
            print("Validation BLEU Score= ", score)

            loss, score = validate_model(encoder, decoder, 
                                         dataloader["train"], # vien_loader if lang == vien_en_
                                         loss_function, 
                                         vien_en_) # target_language
            
            print("Training Loss = ", loss)
            print("Traning BLEU Score= ", score)

            epoch_loss = running_loss/total
            epoch_acc = 0
            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)
            print("Epoch {} {} Loss = {}, Accurancy = {} Time = {}".format(epoch, phase, epoch_loss, epoch_acc,
                                                                           time.time() - start))
        if phase == "val" and epoch_acc > best_score:
            best_score = epoch_acc

    print("Training completed. Best accuracy is {}".format(best_score))
    return encoder, decoder