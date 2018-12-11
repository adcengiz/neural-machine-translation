# chinese -> english
enc = RNNencoder(vocab_size=len(vien_vi_train_token2id), # for chinese
                 embedding_size=300,
                 percent_dropout=0.3, 
                 hidden_size=256,
                 num_gru_layers=20).to(device)

dec = RNNdecoder(vocab_size=len(vien_en_train_token2id), # for chinese-english's english
                 embedding_size=300,
                 percent_dropout=0.3, 
                 hidden_size=256,
                 num_gru_layers=20).to(device)

# model = Translate(enc, dec).to(device)

loss_hist = []
# train

BATCH_SIZE = 32
def train(encoder, decoder, loader=vien_train_loader,
          optimizer = torch.optim.Adam([*enc.parameters()] + [*dec.parameters()], lr=1e-4),
          epoch=None, teacher_forcing=False, criterion = torch.nn.BCEWithLogitsLoss()):

    encoder.train()
    decoder.train()
    optimizer.zero_grad()
    
    loss = 0
    
    for batch_idx, (source_sentence, source_mask, source_lengths, 
                    target_sentence, target_mask, target_lengths)\
                    in enumerate(loader):
        
        sigmoid = nn.Sigmoid()
        
        source_sentence, source_mask, source_lengths = source_sentence.to(device), source_mask.to(device), torch.from_numpy(source_lengths).to(device)
        target_sentence, target_mask, target_lengths = target_sentence.to(device), target_mask.to(device), torch.from_numpy(target_lengths).to(device)
        
        encoder_hidden, source_lengths = encoder(source_sentence,
                                               source_mask,
                                               source_lengths)
        
        decoder_hidden = encoder_hidden.to(device)
        
        # decoder should start with SOS tokens 
        # ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        input_ = SOS_token*torch.ones(BATCH_SIZE,1).view(-1,1).to(device)
        
        
        if teacher_forcing:
            
            decoder_outputs = torch.zeros(BATCH_SIZE, decoder.max_sentence_len, decoder.vocab_size).to(device)
            
            for t in range(0, target_sentence.size(1)):

                decoder_out, decoder_hidden = decoder(decoder_hidden, # = gru_out_source - instead of encoded_source[0]
                                                     input_, # instead of target sentence up to t 
                                                     target_lengths,  # target lengths
                                                     target_mask,
                                                     t)
                
#                 decoder_outputs[:,t,:] = decoder_out.view(BATCH_SIZE, decoder.vocab_size)
            
                input_ = target_sentence[:,t].view(-1,1).to(device)
#                 print ("deco out =" +str(decoder_out.view(BATCH_SIZE, decoder.vocab_size).topk(1)[1]))
                print ("deco out = "+str(decoder_out))
                loss += criterion(decoder_out.view(BATCH_SIZE, decoder.vocab_size), convert_to_softmax(target_sentence[:,t], batch_size=32))
                print ("loss = "+str(loss))
                
                
#             loss = torch.mean(loss_tensor).to(device)
            loss.cuda().backward()
#             loss += criterion(F.sigmoid(decoder_out), target_tokens)
            

            print ("loss = "+str('{0:.16f}'.format(loss)))
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

            optimizer.step()
            
            if epoch >= 10:
                print ("deco out sentence 0: " + str([*decoder_outputs[0].topk(1)[1].detach().cpu().numpy()]))
                print ("target sentence 0: "+ str([*target_sentence[0].detach().cpu().numpy()]))
                                                                           
            
        
        else:
            
            decoder_outputs = torch.zeros(BATCH_SIZE, decoder.max_sentence_len, decoder.vocab_size).to(device)
            
            for t in range(0, target_sentence.size(1)):

                decoder_out, decoder_hidden = decoder(decoder_hidden, # = gru_out_source - instead of encoded_source[0]
                                                     input_, # instead of target sentence up to t 
                                                     target_lengths,  # target lengths
                                                     target_mask,
                                                     t)

                decoder_outputs[:,t,:] = decoder_out.view(BATCH_SIZE, decoder.vocab_size)
                input_ = decoder_out.topk(1)[1].view(BATCH_SIZE, 1).to(device)
#                 print ("input_ = "+ str(input_))
#                 print ("input_ size = "+str(input_.size()))

#                 loss += criterion(F.sigmoid(decoder_out), target_tokens)

            loss_tensor = torch.zeros(BATCH_SIZE, 1)
            decoder_outputs.detach()
            for i in range(BATCH_SIZE):
                loss_tensor[i] = torch.sum(criterion(decoder_outputs[i], target_sentence[i]))/target_lengths.float()[i]
                
            loss = (torch.sum(loss_tensor)/BATCH_SIZE).to(device)
            loss.backward(retain_graph = True)
#             loss += criterion(F.sigmoid(decoder_out), target_tokens)
            

            print ("loss = "+str('{0:.16f}'.format(loss)))
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

            optimizer.step()
            
            if epoch == 50:
                print ("deco out : " + str(decoder_outputs[0].topk(1)[1]))
                print ("target_sentence 1 :  "+str(target_sentence))
        
    torch.save(encoder.state_dict(), "rnn_encoder_state_dict")
    torch.save(decoder.state_dict(), "rnn_decoder_state_dict")
            
    return loss
        

    