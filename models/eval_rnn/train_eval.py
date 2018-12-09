# chinese -> english

# encoder_ = RNNencoder(vocab_size=len(zhen_zh_train_token2id),
#                       embedding_size=300,
#                       percent_dropout=0.3, 
#                       hidden_size=256,
#                       num_gru_layers=16).to(device)

# decoder_ = RNNdecoder(vocab_size=len(zhen_en_train_token2id),
#                       embedding_size=300,
#                       percent_dropout=0.3,
#                       hidden_size=256,
#                       num_gru_layers=4).to(device)

# vietnamese -> english
BATCH_SIZE = 32
# hyperparameters
emb_size = 300
emb_dropout = 0.3
val_hidden_size = 256
lr = 1e-3

# model param
enc_num_layers = 16
dec_num_layers = 4

encoder_ = RNNencoder(vocab_size=len(vien_vi_train_token2id),
                      embedding_size=emb_size,
                      percent_dropout=emb_dropout, 
                      hidden_size=val_hidden_size,
                      num_gru_layers=16).to(device)

decoder_ = RNNdecoder(vocab_size=len(vien_en_train_token2id),
                      embedding_size=emb_size,
                      percent_dropout=emb_dropout,
                      hidden_size=256,
                      num_gru_layers=4).to(device)

def train_eval_rnn(encoder_model, decoder_model, source_lang = "vi", 
                   train_loader=None, train_criterion=torch.nn.NLLLoss(),
                   optimizer=torch.optim.Adam([*encoder_.parameters()] + [*decoder_.parameters()], lr=lr),
                   val_loader=None, val_criterion=corpus_bleu, epoch=None, 
                   search="greedy", val_interval = 10, lr = 1e-3, num_epochs=100,
                   teacher_forcing = True):
    
    """Trains and evaluates the seq2seq model
    
    No teacher forcing in validation!"""
    
    if source_lang == "vi" and train_loader != vien_train_loader:
        raise ValueError("Train loader and source language should be compatible!")
    elif source_lang == "zh" and train_loader != zhen_train_loader:
        raise ValueError("Train loader and source language should be compatible!")
    elif source_lang == "vi" and val_loader != vien_dev_loader:
        raise ValueError("Val loader and source language should be compatible!")
    elif source_lang == "zh" and val_loader != zhen_dev_loader:
        raise ValueError("Val loader and source language should be compatible!")
    
    for epoch in range(num_epochs):
        print ("epoch = "+str(epoch))
        
        if epoch != val_interval:
            # train
            encoder_model.train()
            decoder_model.train()
            optimizer.zero_grad()

            loss = 0

            for batch_idx, (source_sentence, source_mask, source_lengths, 
                            target_sentence, target_mask, target_lengths)\
                            in enumerate(train_loader):

                source_sentence, source_mask, source_lengths = source_sentence.to(device), source_mask.to(device), torch.from_numpy(source_lengths).to(device)
                target_sentence, target_mask, target_lengths = target_sentence.to(device), target_mask.to(device), torch.from_numpy(target_lengths).to(device)

                encoder_hidden, source_lengths = encoder_model(source_sentence, source_mask, source_lengths)

                decoder_hidden = encoder_hidden.to(device)
                input_ = SOS_token*torch.ones(BATCH_SIZE,1).view(-1,1).to(device)

                if teacher_forcing:

                    decoder_outputs = torch.zeros(BATCH_SIZE, decoder_model.max_sentence_len, decoder_model.vocab_size).to(device)

                    for t in range(0, target_sentence.size(1)):
                        decoder_out, decoder_hidden = decoder_model(decoder_hidden, 
                                                             input_, 
                                                             target_lengths,  
                                                             target_mask,
                                                             t)
                        decoder_outputs[:,t,:] = decoder_out.view(BATCH_SIZE, decoder_model.vocab_size)
                        input_ = target_sentence[:,t].view(-1,1).to(device)

                    loss_tensor = torch.zeros(BATCH_SIZE, 1)

                    for i in range(BATCH_SIZE):
                        loss_tensor[i] = torch.sum(train_criterion(decoder_outputs[i], 
                                                             target_sentence[i]))/target_lengths.float()[i]

                    loss = torch.mean(loss_tensor).to(device)
                    loss.backward(retain_graph = True)

                    print ("loss = "+str('{0:.16f}'.format(loss)))

                    optimizer.step()

#                     if epoch >= 10:
#                         print ("deco out sentence 0: " + str([*decoder_outputs[0].topk(1)[1].detach().cpu().numpy()]))
#                         print ("target sentence 0: "+ str([*target_sentence[0].detach().cpu().numpy()]))

                else:

                    decoder_outputs = torch.zeros(BATCH_SIZE, decoder_model.max_sentence_len, decoder_model.vocab_size).to(device)

                    for t in range(0, target_sentence.size(1)):
                        decoder_out, decoder_hidden = decoder_model(decoder_hidden, 
                                                             input_, 
                                                             target_lengths,
                                                             target_mask,
                                                             t)

                        decoder_outputs[:,t,:] = decoder_out.view(BATCH_SIZE, decoder_model.vocab_size)
                        input_ = decoder_out.topk(1)[1].view(BATCH_SIZE, 1).to(device)

                    loss_tensor = torch.zeros(BATCH_SIZE, 1)
                    decoder_outputs.detach()
                    
                    for i in range(BATCH_SIZE):
                        loss_tensor[i] = torch.sum(criterion(decoder_outputs[i], 
                                                             target_sentence[i]))/target_lengths.float()[i]

                    loss = (torch.sum(loss_tensor)/BATCH_SIZE).to(device)
                    loss.backward(retain_graph = True)

                    print ("loss = "+str('{0:.16f}'.format(loss)))
                    optimizer.step()

#                     if epoch == 50:
#                         print ("deco out : " + str(decoder_outputs[0].topk(1)[1]))
#                         print ("target_sentence 1 :  "+str(target_sentence))
            
        else:
            
            f_target = open("rnn_validation_target.txt","a")
            f_model = open("rnn_validation_model.txt","a")
            
    
            # validate in "val_interval" epochs
            for batch_idx, (source_sentence, source_mask, source_lengths, 
                            target_sentence, target_mask, target_lengths)\
                            in enumerate(val_loader):
                
                source_sentence, source_mask, source_lengths = source_sentence.to(device), source_mask.to(device), torch.from_numpy(source_lengths).to(device)
                target_sentence, target_mask, target_lengths = target_sentence.to(device), target_mask.to(device), torch.from_numpy(target_lengths).to(device)
                
                encoder_hidden, source_lengths = encoder_model(source_sentence, source_mask, source_lengths)

                decoder_hidden = encoder_hidden.to(device)
                input_ = SOS_token*torch.ones(BATCH_SIZE,1).view(-1,1).to(device)
                
                decoder_outputs = torch.zeros(BATCH_SIZE, decoder_model.max_sentence_len, decoder_model.vocab_size).to(device)

                for t in range(0, target_sentence.size(1)):
                    decoder_out, decoder_hidden = decoder_model(decoder_hidden, 
                                                         input_, 
                                                         target_lengths,
                                                         target_mask,
                                                         t)

                    decoder_outputs[:,t,:] = decoder_out.view(BATCH_SIZE, decoder_model.vocab_size)
                    input_ = decoder_out.topk(1)[1].view(BATCH_SIZE, 1).to(device)

                decoder_out_indices_val = decoder_outputs.topk(1, 2)[1].view(BATCH_SIZE, decoder_model.max_sentence_len).detach().cpu().numpy()
                target_indices_val = target_sentence.detach().cpu().numpy()
                
                for bi in range(BATCH_SIZE):
                    output, reference = out_token_to_string(decoder_out_indices_val[bi],
                                                           target_indices_val[bi], source_lang="vi")
                    f_target.write(reference)
                    f_model.write(output)
                    
            f_target.close()
            f_model.close()
                    