#!/usr/bin/env python3

import os, random, cv2
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers

from DataLoader import DataLoader
from Models import Models

def train(ts_model, ms_model, bs_model, train_data, optimizer):
    f = open('train.LOG', 'a')
    f.write('In training\n')
    f.close()

    save_path = '/root/arun/Handwriting-Recognition/models'

    _, labels_pos_map, pos_labels_map = find_unique_characters()

    epochs = 10
    train_imgs_ts, train_imgs_bs, train_labels, train_n  =\
         train_data['imgs_ts'], train_data['imgs_bs'], train_data['labels'], train_data['n']

    for epoch in range(1, epochs+1):
        f = open('train.LOG', 'a')
        f.write('Epoch is {}\n'.format(epoch))
        f.close()

        epoch_loss = 0
        cer = 0
        indexes = np.arange(0, len(train_imgs_ts), 1)
        np.random.shuffle(indexes)

        for i, idx in enumerate(indexes):
            if (i%1000 == 0):
                f = open('train.LOG', 'a')
                f.write('Epoch {} i {}   '.format(epoch, i))
                f.close()

            img_ts = tf.convert_to_tensor(train_imgs_ts[idx])  #top stream input
            img_bs = tf.convert_to_tensor(train_imgs_bs[idx])  # bottom stream input
            actual_word = train_labels[idx]

            with tf.GradientTape(persistent=True) as tape:
            # ms_model - middle stream model, ts - top stream model, bs_model - bottom stream model
                out = ms_model(ts_model(img_ts) + bs_model(img_bs))
                out = tf.reshape(out, (out.shape[2], out.shape[3]))
                out = out / tf.norm(out, axis=1, keepdims=True)
                predicted_word = list()
                for i in range(1, len(out), 2):
                    predicted_word.append(pos_labels_map[np.argmax(out[i])])
                predicted_word = ''.join(predicted_word)

                target = np.zeros(out.shape)
                for i in range(1, out.shape[0], 2):
                    char = actual_word[(i - 1)//2]  # Shape of out is 2n
                    pos = labels_pos_map[char]
                    target[i][pos] = 1.0
                target = tf.convert_to_tensor(target, dtype=tf.float32)

                loss = models.norm_inf_loss(target, out)

            grads_ms = tape.gradient(loss, ms_model.trainable_weights)
            grads_ts = tape.gradient(loss, ts_model.trainable_weights)
            grads_bs = tape.gradient(loss, bs_model.trainable_weights)

            grads_ms, _ = tf.clip_by_global_norm(grads_ms, 1.0)
            grads_ts, _ = tf.clip_by_global_norm(grads_ts, 1.0)
            grads_bs, _ = tf.clip_by_global_norm(grads_bs, 1.0)

            grads_ms = adjust_gradient(grads_ms, epoch)
            grads_ts = adjust_gradient(grads_ts, epoch)
            grads_bs = adjust_gradient(grads_bs, epoch)

            optimizer.apply_gradients(zip(grads_ms, ms_model.trainable_weights))
            optimizer.apply_gradients(zip(grads_bs, bs_model.trainable_weights))
            optimizer.apply_gradients(zip(grads_ts, ts_model.trainable_weights))

            epoch_loss += loss
            cer += models.get_character_error_rate(actual_word, predicted_word)

    ts_model.save(os.path.join(save_path, 'ts_model.h5'))
    ms_model.save(os.path.join(save_path, 'ms_model.h5'))
    bs_model.save(os.path.join(save_path, 'bs_model.h5'))


    return (ts_model, ms_model, bs_model)

def test(ts_model, ms_model, bs_model, test_data):
    print ('In testing \n\n\n')
    _, labels_pos_map, pos_labels_map = find_unique_characters()

    test_imgs_ts, test_imgs_bs, test_labels, test_n =\
         test_data['imgs_ts'], test_data['imgs_bs'], test_data['labels'], test_data['n']

    total_characters = 0
    correct_prediction = 0

    for idx in range(len(test_imgs_ts)):
        img_ts = tf.convert_to_tensor(test_imgs_ts[idx])  #top stream input
        img_bs = tf.convert_to_tensor(test_imgs_bs[idx])  # bottom stream input
        actual_word = test_labels[idx]

        out = ms_model(ts_model(img_ts) + bs_model(img_bs))
        out = tf.reshape(out, (out.shape[2], out.shape[3]))
        out = out / tf.norm(out, axis=1, keepdims=True)
        predicted_word = list()
        for i in range(1, len(out), 2):
            predicted_word.append(pos_labels_map[np.argmax(out[i])])
        predicted_word = ''.join(predicted_word)

        total_characters += len(actual_word)
        for i in range(len(actual_word)):
            if actual_word[i] == predicted_word[i]:
                correct_prediction += 1

        if (idx < 5):
            print (actual_word, predicted_word, sep=' ', end='   ')

    print ('Total characters {} correct prediction {}'.format(total_characters, correct_prediction))
    print ("Character wise accuracy {}".format(correct_prediction*100 / total_characters))
    return (0)

def find_unique_characters():
#Finding Ns and creating a map of labels and index
    _, labels = dataloader.get_image_paths_labels()
    unique_characters = list()
    for i, label in enumerate(labels):
        for char in label:
            if char not in unique_characters:
                unique_characters.append(char)

    unique_characters.sort()
    Ns = len(unique_characters)

    pos_labels_map = {}
    labels_pos_map = {}
    for i, char in enumerate(unique_characters):
        labels_pos_map[char] = i
        pos_labels_map[i] = char

    return (Ns, labels_pos_map, pos_labels_map)

def adjust_gradient(grads, epoch, eta=0.03, gamma=0.55):
    #Adding gradient noise and clipping the gradients
    var = eta / ((1 + epoch) ** gamma)
    for i, grad in enumerate(grads):
        grads[i] = grad + np.random.normal(0, var, grad.shape)
    return (grads)

def main():

    Ns, _, __ = find_unique_characters()

    start = datetime.now()
    train_data, test_data = dataloader.make_train_test()
    end = datetime.now()

    f = open('main.LOG', 'a')
    f.write('Time taken to load data is {} seconds'.format((end - start).seconds))
    f.close()

    train_count = len(train_data['n'])
    test_count = len(test_data['n'])
    print ('Total samples in training data is {} and in test data is {}'.format(train_count, test_count))

    models_loc = '/root/arun/Handwriting-Recognition/models'

    # files = os.listdir(models_loc)
    # for file in files:
    #     os.remove(os.path.join(models_loc, file))

    if len(os.listdir(models_loc)) >= 3:
        ts_model = tf.keras.models.load_model(os.path.join(models_loc, 'ts_model.h5'))
        bs_model = tf.keras.models.load_model(os.path.join(models_loc, 'bs_model.h5'))
        ms_model = tf.keras.models.load_model(os.path.join(models_loc, 'ms_model.h5'))
    else:
        ts_model = models.get_top_stream()
        bs_model = models.get_bottom_stream()
        ms_model = models.get_middle_stream(Ns)

    optimizer = models.get_optimizer()

    start = datetime.now()
    ts_model, ms_model, bs_model = train(ts_model, ms_model, bs_model, train_data, optimizer)
    end = datetime.now()
    f = open('main.LOG', 'a')
    f.write('Time trained {} seconds'.format((end - start).seconds))
    f.close()
    test(ts_model, ms_model, bs_model, test_data)
#Training model


if __name__ == '__main__':
    models = Models()
    dataloader = DataLoader()
    main()
