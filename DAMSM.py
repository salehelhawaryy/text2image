# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Date        :2021/04/14 16:05
@Author      :Wentong Liao, Kai Hu
@Email       :liao@tnt.uni-hannover.de
@Version     :0.1
@Description : Implementation of SSA-GAN
'''
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from datasets import TextDatasetDAMSM_Text
from datasets import prepare_data_text


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


# ############## Text2Image Encoder-Decoder #######

class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        #model = models.inception_v3()
        #url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # model.load_state_dict(model_zoo.load_url(url))
        model = models.inception_v3(pretrained=True, transform_input=False)
        for param in model.parameters():
            param.requires_grad = False
        #print('Load pretrained model from ', url)
        # print(model)

        print('Load pretrained inception v3 model')
        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb
    
if __name__ == '__main__':
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    
    dataset = TextDatasetDAMSM_Text(cfg.DATA_DIR, 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    # ixtoword = dataset.ixtoword
    # print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    dataset_val = TextDatasetDAMSM_Text(cfg.DATA_DIR, 'validation',
                                    base_size=cfg.TREE.BASE_SIZE,
                                    transform=image_transform)
    # ixtoword = dataset.ixtoword
    # print(dataset.n_words, dataset.embeddings_num)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder_ar = RNN_ENCODER(dataset.n_words_ar, nhidden=cfg.TEXT.EMBEDDING_DIM)
    text_encoder_ar.cuda()
    text_encoder_en = RNN_ENCODER(27297, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder_en.load_state_dict(state_dict)
    text_encoder_en.cuda()
    for p in text_encoder_en.parameters():
        p.requires_grad = False
    text_encoder_en.eval()
    #fetch one batch of data from dataloader and print resutls

    # for param in text_encoder_ar.parameters(): print(param.requires_grad)
    # print('')
    # for param in text_encoder_en.parameters(): print(param.requires_grad)


    no_epochs = 5
    best_val_loss = float('inf')
    # Define the loss function and the optimizer
    criterion = nn.MSELoss()  # Use Mean Squared Error loss for comparing embeddings
    optimizer = torch.optim.Adam(text_encoder_ar.parameters(), lr=0.001)

    for epoch in range(no_epochs):
        data_iter = iter(dataloader)
        total_train_loss = 0
        print(f'Epoch: {epoch + 1}')
        for step in tqdm(range(len(data_iter))):
            data = next(data_iter)
            captions_ar, sorted_cap_len_ar, captions_en, sorted_cap_len_en, class_ids_ar, class_ids_en, keys_ar, keys_en = prepare_data_text(
                data)

            # Initialize hidden states
            hidden_ar = text_encoder_ar.init_hidden(cfg.TRAIN.BATCH_SIZE)
            hidden_en = text_encoder_en.init_hidden(cfg.TRAIN.BATCH_SIZE)

            # Get embeddings from Arabic text encoder
            words_embs_ar, sent_emb_ar = text_encoder_ar(captions_ar, sorted_cap_len_ar, hidden_ar)
            words_embs_ar, sent_emb_ar = words_embs_ar, sent_emb_ar

            # Get embeddings from English text encoder
            words_embs_en, sent_emb_en = text_encoder_en(captions_en, sorted_cap_len_en, hidden_en)
            words_embs_en, sent_emb_en = words_embs_en.detach(), sent_emb_en.detach()

            # Compute the loss
            loss = criterion(sent_emb_ar, sent_emb_en)
            total_train_loss += loss.item()

            # Perform backpropagation
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Zero the gradients
            optimizer.zero_grad()

        print(f'Training Loss: {total_train_loss / len(data_iter)}')
        data_iter_val = iter(dataloader_val)
        total_val_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for step in tqdm(range(len(data_iter_val))):
                data = next(data_iter_val)
                captions_ar, sorted_cap_len_ar, captions_en, sorted_cap_len_en, class_ids_ar, class_ids_en, keys_ar, keys_en = prepare_data_text(
                    data)

                # Initialize hidden states
                hidden_ar = text_encoder_ar.init_hidden(cfg.TRAIN.BATCH_SIZE)
                hidden_en = text_encoder_en.init_hidden(cfg.TRAIN.BATCH_SIZE)

                # Get embeddings from Arabic text encoder
                words_embs_ar, sent_emb_ar = text_encoder_ar(captions_ar, sorted_cap_len_ar, hidden_ar)
                words_embs_ar, sent_emb_ar = words_embs_ar, sent_emb_ar

                # Get embeddings from English text encoder
                words_embs_en, sent_emb_en = text_encoder_en(captions_en, sorted_cap_len_en, hidden_en)
                words_embs_en, sent_emb_en = words_embs_en.detach(), sent_emb_en.detach()

                # Compute the validation loss
                val_loss = criterion(sent_emb_ar, sent_emb_en)
                total_val_loss += val_loss.item()

        # Print the average validation loss after each epoch
        avg_val_loss = total_val_loss / len(data_iter_val)
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(text_encoder_ar.state_dict(), 'DAMSMencoders/text_encoder_ar.pth')
            print(f"New best validation loss: {best_val_loss}. Saving model.")
    #Write training loop for text_encoder
