def out_token_2_string(index_tensor, 
                       language):
    sentence = []
    for i in index_tensor:
        if i.item() not in [0, 1, 3]: # <PAD>, <SOS>, <EOS>
            sentence.append(language.index2word[i.item()])
    return (' ').join(sentence)


def token_to_string_(output_sentence, target_sentence, 
                        source_lang = "vi"):
    
    """Takes as input model output and reference token tensors, and 
    converts into plain string.
    
    :output_sentence: a list of integers, can be obtained by [*batch_tensor.numpy[i]] for an
                      output tensor of (batch_size, seq_len)
    :target_setence: a list of integers, can be obtained by [*batch_tensor.numpy[i]] for an
                      output tensor of (batch_size, seq_len)
                      
    :source_lang: 'vi' or 'zh', used to determine the target reference vocab.
    
    """
    
    if source_lang == "vi":
        target_vocab = vien_en_train.index2word
    else:
        target_vocab = zhen_en_train.index2word
    
    output_string = (" ").join([target_vocab[x] for x in [*output_sentence]]) + " "
    reference_string = (" ").join([target_vocab[x] for x in [*target_sentence]]) + " "
    
    return output_string, reference_string



 def convert_to_softmax(tensor_of_indices,
                       batch_size,
                       vocab_size = len(zhen_en_train_token2id)):
    """
    - takes as input a time_step vector of the batch (t-th token of each sentence in the batch)
      size: (batch_size, 1)
    - converts it to softmax of (batch_size, vocab_size)
    """
    index_tensor_ = tensor_of_indices.view(-1,1).long()
        
    one_hot = torch.FloatTensor(batch_size, vocab_size).zero_()
    one_hot.scatter_(1, index_tensor_.detach().cpu(), 1)
    
    return one_hot


# Convert a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    """About "NFC" and "NFD": 
    
    For each character, there are two normal forms: normal form C 
    and normal form D. Normal form D (NFD) is also known as canonical 
    decomposition, and translates each character into its decomposed form. 
    Normal form C (NFC) first applies a canonical decomposition, then composes 
    pre-combined characters again.
    
    About unicodedata.category: 
    
    Returns the general category assigned to the Unicode character 
    unichr as string."""
    
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def out_token_to_string(output_sentence, target_sentence, 
                        source_lang = "vi"):
    
    """Takes as input model output and reference token tensors, and 
    converts into plain string.
    
    :output_sentencco   output tensor of (batch_size, seq_len)
    :target_setence: a list of integers, can be obtained by [*batch_tensor.numpy[i]] for an
                      output tensor of (batch_size, seq_len)
                      
    :source_lang: 'vi' or 'zh', used to determine the target reference vocab.
    
    """
    
    if source_lang == "vi":
        target_vocab = vien_en_train.index2word
    else:
        target_vocab = zhen_en_train.index2word
    
    output_string = (" ").join([target_vocab[x] for x in [*output_sentence]]) + " "
    reference_string = (" ").join([target_vocab[x] for x in [*target_sentence]]) + " "
    
    return output_string, reference_string


def train_eval_rnn(encoder_model, decoder_model, source_lang = "zh", 
                   train_loader=None, train_criterion=torch.nn.NLLLoss(ignore_index=0, reduction="sum"),
                   optimizer=torch.optim.Adam([*encoder_.parameters()] + [*decoder_.parameters()], lr=lr),
                   val_loader=None, val_criterion=corpus_bleu, epoch=None, 
                   search="greedy", val_interval = [10,20,30], lr = 1e-3, num_epochs=100,
                   teacher_forcing = False):
    
    """Trains and evaluates the rnn encoder-decoder-based seq2seq model
    
    No teacher forcing in validation!"""
    
    if source_lang == "vi" and train_loader != vien_train_loader:
        raise ValueError("Train loader and source language should be compatible!")
    elif source_lang == "zh" and train_loader != zhen_train_loader:
        raise ValueError("Train loader and source language should be compatible!")
    elif source_lang == "vi" and val_loader != vien_dev_loader:
        raise ValueError("Val loader and source language should be compatible!")
    elif source_lang == "zh" and val_loader != zhen_dev_loader:
        raise ValueError("Val loader and source language should be compatible!")
        
    training_losses = []
    
    encoder_model.train()
    decoder_model.train()
    
    for epoch in range(num_epochs):
        print ("epoch = "+str(epoch))
        # train
        
        
        for batch_idx, (source_sentence, source_mask, source_lengths, 
                target_sentence, target_mask, target_lengths)\
                in enumerate(train_loader):
            
            loss = 0
            
            optimizer.zero_grad()

            source_lengths = torch.from_numpy(source_lengths)
            target_lengths = torch.from_numpy(target_lengths)

            source_sentence, source_lengths = source_sentence.to(device), source_lengths.to(device)
            target_sentence, target_lengths = target_sentence.to(device), target_lengths.to(device)

            encoder_hidden, encoder_outs = encoder_model(source_sentence, source_lengths)

            decoder_hidden = encoder_hidden
            #input_ = SOS_token*torch.ones(BATCH_SIZE,1).to(device)
            input_ = torch.tensor([1]*BATCH_SIZE).unsqueeze(1).to(device)
            
            
            
            if teacher_forcing:

                for t in range(0, target_sentence.size(1)):
                    decoder_out, decoder_hidden = decoder_model(decoder_hidden, 
                                                         input_, 
                                                         target_lengths,
                                                         t)

                    input_ = target_sentence[:,t].unsqueeze(1)
                    loss += train_criterion(decoder_out,
                              target_sentence[:,t]) 

                loss /= torch.sum(target_lengths.float())
                training_losses.append(loss.item())
                print("loss = "+str(loss.item()))
                loss.backward()
                optimizer.step()
                

            else:

                loss = 0

                for t in range(0, target_sentence.size(1)):
                    decoder_out, decoder_hidden = decoder_model(decoder_hidden, 
                                                         input_, 
                                                         target_lengths,
                                                         t)

                    input_ = decoder_out.topk(1)[1].view(BATCH_SIZE, 1).to(device)
                    loss += train_criterion(decoder_out.view(BATCH_SIZE, decoder_model.vocab_size),
                              target_sentence[:,t])

                loss /= torch.sum(target_lengths.float())
                loss.backward()

                print ("loss = "+str('{0:.16f}'.format(loss)))
                training_losses.append(loss.item())
                optimizer.step()

#                     if epoch == 50:
#                         print ("deco out : " + str(decoder_outputs[0].topk(1)[1]))
#                         print ("target_sentence 1 :  "+str(target_sentence))

            if epoch in val_interval:
            	
            	# validation pass
                f_target = open("rnn_validation_target_"+str(epoch)+".txt","a")
                f_model = open("rnn_validation_model_"+str(epoch)+".txt","a")


                # validate in "val_interval" epochs
                for batch_idx, (source_sentence, source_mask, source_lengths, 
                        target_sentence, target_mask, target_lengths)\
                        in enumerate(val_loader):

                    source_lengths = torch.from_numpy(source_lengths)
                    target_lengths = torch.from_numpy(target_lengths)

                    source_sentence, source_lengths = source_sentence.to(device), source_lengths.to(device)
                    target_sentence, target_lengths = target_sentence.to(device), target_lengths.to(device)

                    encoder_hidden, source_lengths = encoder_model(source_sentence, source_lengths)

                    decoder_hidden = encoder_hidden.to(device)
                    input_ = SOS_token*torch.ones(BATCH_SIZE,1).view(-1,1).to(device)

                    decoder_outputs = torch.zeros(BATCH_SIZE, decoder_model.max_sentence_len, decoder_model.vocab_size).to(device)

                    for t in range(0, target_sentence.size(1)):
                        decoder_out, decoder_hidden = decoder_model(decoder_hidden, 
                                                             input_, 
                                                             target_lengths,
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
                    
                
        return training_losses