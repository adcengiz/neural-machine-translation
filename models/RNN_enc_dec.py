# from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# same as 1st model's RNN encoder
# the different part is the attention decoder in model 2

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
            `   source_sentence, 
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


# model = rnn_translate(enc, dec).to(device)


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



from sacreBLEU.sacreBLEU import corpus_bleu

def validate_model(encoder_model, 
                   decoder_model, 
                   dataloader, 
                   loss_function, 
                   en_): # if source lang == Chinese -> zhen_en_, Vietnamese -> vien_en_
    # validation pass - no teacher forcing
    encoder.train(False)
    decoder.train(False)

    model_corpus = []
    reference_corpus = []

    loss_total = 0
    total = 0

    for sentence in dataloader:

        encoder_input = sentence[0].to(device) # encoder input - source sentence
        decoder_input = sentence[1].to(device) # decoder input - target sentence
        source_lengths = sentence[2].to(device)

        batch_size, seq_len = encoder_input.size()[:2]

        out, hidden = rnn_translate(encoder_model,
                                    decoder_model,
                                    encoder_input,
                                    decoder_input,
                                    source_lengths)

        loss = loss_function(out.float(), decoder_input.long())
        loss_total += loss.item()*batch_size
        total = total + batch_size
        preds = torch.max(out, dim = 1)[1]

        for true, preds in zip(sentence[1], preds):

            true, preds = out_token_2_string(true, en_), out_token_2_string(preds, en_)
            # model-translated tokens
            model_corpus.append(preds)
            # ground truth translation
            reference_corpus.append(true)
            

    bleu_score = corpus_bleu((" ").join(pred_corpus),
                             (" ").join(true_corpus))

    loss = loss_total/total

    return loss, bleu_score

def train_model(encoder_optimizer,
                decoder_optimizer, 
                encoder_model, 
                decoder_model, 
                dataloader,
                loss_function, 
                num_epochs=10):
    
    loss_hist_dict = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print ("epoch", epoch)

        for i, mode in enumerate(["train"]):

            start = time.time()
            total = 0
            top1_correct = 0
            loss_total = 0
            running_total = 0

            if mode == "train":
                encoder_model.train()
                decoder_model.train()

            else:
                encoder_model.train(False)
                decoder_model.train(False)
                
            for sentence in dataloader[mode]:
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_input = sentence[0].to(device)
                decoder_input = sentence[1].to(device)
                source_lengths = sentence[2].to(device)
                                
                out, hidden = encode_decode_rnn(encoder_model, 
                                                decoder_model, 
                                                encoder_input, 
                                                decoder_input, 
                                                source_lengths)
                
                loss = loss_function(out.float(), decoder_input.long())

                N = decoder_i.size(0)

                loss_total += loss.item() * N
                
                total += N

                if mode == "train":
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    
            loss, score = validate_model(encoder_model, 
                                         decoder_model, 
                                         dataloader["val"], # vien_loader if lang == vien_en_
                                         loss_function, # nll
                                         vien_en_) # target_language

            print("Validation NLL Loss: ", loss)
            print("BLEU score on validation set is ", score)

            loss, score = validate_model(encoder_model, 
                                         decoder_model, 
                                         dataloader["train"], # vien_loader if lang == vien_en_
                                         loss_function, 
                                         vien_en_) # target_language

            epoch_loss = loss_total/total
            epoch_acc = 0
            loss_hist_dict[mode].append(epoch_loss)

            print("Epoch {} {}, Time = {}".format(epoch, mode, 
                time.time() - start))

    return encoder_model, decoder_model