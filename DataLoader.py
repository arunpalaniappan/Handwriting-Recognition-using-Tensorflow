#!/usr/bin/env python3

import os, random, cv2
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers

import logging

class DataLoader:

    def get_file_path(self, file_name):
#        path = '/root/Data/IAM/Words'
        path = 'Data/Words'
        destination = file_name[0]
        path = os.path.join(path, destination)
        for i in range(1, len(file_name)):
            destination = destination + "-" + file_name[i]
            path = os.path.join(path, destination)

        path = path + '.png'
        return (path)


    def get_image_paths_labels(self):
        img_paths = []
        img_labels = []
#        data_path = '/root/Data/IAM'
        data_path = 'Data'

        words = os.path.join(data_path, 'words.txt')

        with open(words) as labels_file:
            for line in labels_file:
                if line[0] == "#":
                    continue
                else:
                    label = line.split(' ')[-1].strip('\n')
                    file_name = line.split(' ')[0].split('-')
                    file_name[2] = file_name[2] + '-' + file_name[3]
                    file_name.pop()

                    path = self.get_file_path(file_name)
                    img_paths.append(path)
                    img_labels.append(label)

        return (img_paths, img_labels)


    def make_train_test(self):

        img_paths, img_labels = self.get_image_paths_labels()

        word_count = defaultdict(int)

        for word in img_labels:
            word_count[word] += 1

        common_words_counts = Counter(word_count).most_common(20)

        common_words = list(list(zip(*common_words_counts))[0])

        not_common_paths_labels = []

        for i, label in enumerate(img_labels):
            if label not in common_words:
                not_common_paths_labels.append((img_paths[i], label))


        random.shuffle(not_common_paths_labels)

        train_len = int(0.7 * len(not_common_paths_labels))
        test_len = len(not_common_paths_labels) - train_len

        train_paths_labels = not_common_paths_labels[0:train_len]
        test_paths_labels = not_common_paths_labels[train_len:]

        print ('Length of training data is {} test data is {}'.format(len(train_paths_labels), len(test_paths_labels)))
        train_imgs_ts = list()
        train_imgs_bs = list()
        train_labels = list()
        train_n = list()

        count = 0
        width, height = 128, 32

        for path_label in train_paths_labels:

            n = len(path_label[1])
            img = cv2.imread(path_label[0])
            if (img is not None) and (not np.isnan(img).any()):   #It is a valid image and the input does not contain empty values
                img = (cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)) / 255.0
                img_ts = cv2.resize(img, (width, height)).astype(np.float32).reshape(1, 32, 128, 1) 
                train_imgs_ts.append(img_ts)

                img_bs = cv2.resize(img, (16*n, height)).astype(np.float32)
                shape = img_bs.shape
                img_bs = img_bs.reshape((1, shape[0], shape[1], 1))
                train_imgs_bs.append(img_bs)

                train_labels.append(path_label[1])
                train_n.append(n)


        train = {'imgs_ts':train_imgs_ts, 'imgs_bs':train_imgs_bs, 'labels':train_labels, 'n':train_n}

        test_imgs_ts = list()
        test_imgs_bs = list()
        test_labels = list()
        test_n = list()

        count = 0
        width, height = 128, 32

        for path_label in test_paths_labels:
            n = len(path_label[1])
            img = cv2.imread(path_label[0])
            if img is not None:
                img = (cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))/255.0

                img_ts = cv2.resize(img, (width, height)).astype(np.float32).reshape(1, 32, 128, 1)
                test_imgs_ts.append(img_ts)

                img_bs = cv2.resize(img, (16*n, height)).astype(np.float32)
                shape = img_bs.shape
                img_bs = img_bs.reshape((1, shape[0], shape[1], 1))
                test_imgs_bs.append(img_bs)

                test_labels.append(path_label[1])
                test_n.append(n)


        test = {'imgs_ts':test_imgs_ts, 'imgs_bs':test_imgs_bs, 'labels':test_labels, 'n':test_n}

        return (train, test)
