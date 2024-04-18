# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Date        :2021/04/14 16:05
@Author      :Wentong Liao, Kai Hu
@Email       :liao@tnt.uni-hannover.de
@Version     :0.1
@Description : Implementation of SSA-GAN
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]

def prepare_data_text(data):
    captions_ar, captions_len_ar, captions_en, captions_len_en, class_ids_ar, class_ids_en, keys_en, keys_ar = data

    # sort data by the length in a decreasing order
    sorted_cap_len_ar, sorted_cap_indices_ar = \
        torch.sort(captions_len_ar, 0, True)

    sorted_cap_len_en, sorted_cap_indices_en = \
        torch.sort(captions_len_en, 0, True)

    # real_imgs = []
    # for i in range(len(imgs)):
    #     imgs[i] = imgs[i][sorted_cap_indices]
    #     if cfg.CUDA:
    #         real_imgs.append(Variable(imgs[i]).cuda())
    #     else:
    #         real_imgs.append(Variable(imgs[i]))

    captions_ar = captions_ar[sorted_cap_indices_ar].squeeze()
    captions_en = captions_en[sorted_cap_indices_en].squeeze()

    class_ids_ar = class_ids_ar[sorted_cap_indices_ar].numpy()
    class_ids_en = class_ids_en[sorted_cap_indices_en].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys_ar = [keys_ar[i] for i in sorted_cap_indices_ar.numpy()]
    key_en =  [keys_en[i] for i in sorted_cap_indices_en.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions_ar = Variable(captions_ar).cuda()
        sorted_cap_len_ar = Variable(sorted_cap_len_ar).cuda()

        captions_en = Variable(captions_en).cuda()
        sorted_cap_len_en = Variable(sorted_cap_len_en).cuda()
    else:
        captions_ar = Variable(captions_ar)
        sorted_cap_len_ar = Variable(sorted_cap_len_ar)

        captions_en = Variable(captions_en)
        sorted_cap_len_en = Variable(sorted_cap_len_en)

    return [captions_ar, sorted_cap_len_ar, captions_en, sorted_cap_len_en,
            class_ids_ar, class_ids_en, keys_ar, keys_en]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    ret.append(normalize(img))
    # if cfg.GAN.B_DCGAN:
    '''
    for i in range(cfg.TREE.BRANCH_NUM):
        # print(imsize[i])
        re_img = transforms.Resize(imsize[i])(img)
        ret.append(normalize(re_img))
    '''

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None, is_arabic=True):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.split_name = split

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split, is_arabic)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames, is_arabic):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                #captions = f.read().decode('utf-8').split('\n')
                captions = f.read().split('\n')
                print(captions)
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    if is_arabic:
                        tokenizer = RegexpTokenizer(r'[\u0621-\u064A]+')
                    else:
                        tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue
                    tokens_new = []
                    for t in tokens:
                        t = t.encode('utf-8', 'ignore').decode('utf-8')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split, is_arabic):
        filepath = os.path.join(data_dir, 'captions.pickle')
        if is_arabic:
            train_names = self.load_filenames(data_dir, 'train_ar')
            test_names = self.load_filenames(data_dir, 'test_ar')
        else:
            train_names = self.load_filenames(data_dir, 'train_en')
            test_names = self.load_filenames(data_dir, 'test_en')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names, is_arabic)
            test_captions = self.load_captions(data_dir, test_names, is_arabic)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011/images' % self.data_dir
        else:
            bbox = None
            #data_dir = self.data_dir
            # if self.split_name == 'train':
            #     data_dir = '/data/scene_understanding/coco2014/train2014'
            # else:
            #     data_dir = '/data/scene_understanding/coco2014/val2014'
            data_dir = '/kaggle/input/coco2014/val2014/val2014'
        #
        img_name = '%s/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key

    def __len__(self):
        return len(self.filenames)
    

class TextDatasetDAMSM_Text(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.split_name = split

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames_ar, self.captions_ar, self.ixtoword_ar, \
            self.wordtoix_ar, self.n_words_ar = self.load_text_data(os.path.join(data_dir, 'ar_coco'), split, True)
        
        self.filenames_en, self.captions_en, self.ixtoword_en, \
            self.wordtoix_en, self.n_words_en = self.load_text_data(os.path.join(data_dir, 'en_coco'), split, False)

        self.class_id = self.load_class_id(split_dir, len(self.filenames_ar))
        self.number_example = len(self.filenames_ar)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames, is_arabic):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "rb") as f:
                try:
                    captions = f.read().decode('utf-8', 'ignore').split('\n')
                except UnicodeDecodeError:
                    f.seek(0)  # move file cursor back to beginning
                    print(f.read())
                    print(f"Error decoding file: {cap_path}")
                    sys.exit(1)
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    if is_arabic:
                        tokenizer = RegexpTokenizer(r'[\u0621-\u064A]+')
                    else:
                        tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if i == 123 and cnt == 3:
                        print(cap)
                    if len(tokens) == 0:
                        print('cap token == 0', i, cnt, cap_path)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('utf-8', 'ignore').decode('utf-8')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split, is_arabic):
        if is_arabic:
            filepath = os.path.join(data_dir, 'captions.pickle')
            train_names = self.load_filenames(data_dir, 'train_ar')
            test_names = self.load_filenames(data_dir, 'test_ar')
            if not os.path.isfile(filepath):
                train_captions = self.load_captions(data_dir, train_names, is_arabic)
                test_captions = self.load_captions(data_dir, test_names, is_arabic)

                train_captions, test_captions, ixtoword, wordtoix, n_words = \
                    self.build_dictionary(train_captions, test_captions)
                with open(filepath, 'wb') as f:
                    pickle.dump([train_captions, test_captions,
                                ixtoword, wordtoix], f, protocol=2)
                    print('Save to: ', filepath)
            else:
                with open(filepath, 'rb') as f:
                    x = pickle.load(f)
                    train_captions, test_captions = x[0], x[1]
                    ixtoword, wordtoix = x[2], x[3]
                    del x
                    n_words = len(ixtoword)
                    print('Load from: ', filepath)
            if split == 'train':
                # a list of list: each list contains
                # the indices of words in a sentence
                captions = train_captions
                filenames = train_names
            else:  # split=='test'
                captions = test_captions
                filenames = test_names
            return filenames, captions, ixtoword, wordtoix, n_words
        else:
            filepath = os.path.join(data_dir, 'captions.pickle')
            train_names = self.load_filenames(data_dir, 'train_en')
            test_names = self.load_filenames(data_dir, 'test_en')
            if not os.path.isfile(filepath):
                train_captions = self.load_captions(data_dir, train_names, is_arabic)
                test_captions = self.load_captions(data_dir, test_names, is_arabic)

                train_captions, test_captions, ixtoword, wordtoix, n_words = \
                    self.build_dictionary(train_captions, test_captions)
                with open(filepath, 'wb') as f:
                    pickle.dump([train_captions, test_captions,
                                ixtoword, wordtoix], f, protocol=2)
                    print('Save to: ', filepath)
            else:
                with open(filepath, 'rb') as f:
                    x = pickle.load(f)
                    train_captions, test_captions = x[0], x[1]
                    ixtoword, wordtoix = x[2], x[3]
                    del x
                    n_words = len(ixtoword)
                    print('Load from: ', filepath)
            if split == 'train':
                # a list of list: each list contains
                # the indices of words in a sentence
                captions = train_captions
                filenames = train_names
            else:  # split=='test'
                captions = test_captions
                filenames = test_names
            return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix, is_arabic):
        # a list of indices for a sentence
        if is_arabic:
            sent_caption = np.asarray(self.captions_ar[sent_ix]).astype('int64')
        else:
            sent_caption = np.asarray(self.captions_en[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames_ar[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011/images' % self.data_dir
        else:
            bbox = None
            #data_dir = self.data_dir
            if self.split_name == 'train':
                data_dir = '/data/scene_understanding/coco2014/train2014'
            else:
                data_dir = '/data/scene_understanding/coco2014/val2014'
        #
        # img_name = '%s/%s.jpg' % (data_dir, key)
        # imgs = get_imgs(img_name, self.imsize,
        #                 bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps_ar, cap_len_ar = self.get_caption(new_sent_ix, True)
        caps_en, cap_len_en = self.get_caption(new_sent_ix, False)

        return caps_ar, cap_len_ar, caps_en, cap_len_en, cls_id, cls_id, key, key

    def __len__(self):
        return len(self.filenames_ar)
