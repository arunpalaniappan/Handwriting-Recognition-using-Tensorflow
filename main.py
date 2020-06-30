from collections import defaultdict, Counter
import random, cv2
import numpy as np
import tensorflow as tf

import DataLoader, models
#The model library define all model requirements like nerual network architecture, optimizer function, loss function.

def train(ts_model, ms_model, bs_model, train_data, optimizer):
    _, labels_pos_map, pos_labels_map = find_unique_characters()

    epochs = 100

    train_imgs_ts, train_imgs_bs, train_labels, train_n  =\
         train_data['imgs_ts'], train_data['imgs_bs'], train_data['labels'], train_data['n']

    for epoch in range(epochs):
        epoch_loss = 0
        cer = 0

        for idx in range(len(train_imgs_ts)):
            img_ts = tf.convert_to_tensor(train_imgs_ts[idx])  #top stream input
            img_bs = tf.convert_to_tensor(train_imgs_bs[idx])  # bottom stream input
            actual_word = train_labels[idx]

            with tf.GradientTape(persistent=True) as tape:

            # ms_model - middle stream model, ts - top stream model, bs_model - bottom stream model
                out = ms_model(ts_model(img_ts) + bs_model(img_bs))
                out = tf.reshape(out, (out.shape[2], out.shape[3]))

                predicted_word = list()
                for i in range(1, len(out), 2):
                    predicted_word.append(pos_labels_map[np.argmax(out[i])])
                predicted_word = ''.join(predicted_word)

            # print (actual_word, predicted_word)
            # loss = get_character_error_rate(actual_word, predicted_word)    ->  Not able to construct gradients using this method

            #Note: Shape of target depends on output shape.
                target = np.zeros(out.shape)
                for i in range(1, out.shape[0], 2):
                    char = actual_word[(i - 1)//2]  # Shape of out is 2n
                    pos = labels_pos_map[char]
                    target[i][pos] = 1.0
                target = tf.convert_to_tensor(target, dtype=tf.float32)

                loss = models.norm_loss(target, out)
           
            grads_ms = tape.gradient(loss, ms_model.trainable_weights)
            grads_ts = tape.gradient(loss, ts_model.trainable_weights)
            grads_bs = tape.gradient(loss, bs_model.trainable_weights)
            
            optimizer.apply_gradients(zip(grads_ms, ms_model.trainable_weights))
            optimizer.apply_gradients(zip(grads_bs, bs_model.trainable_weights))
            optimizer.apply_gradients(zip(grads_ts, ts_model.trainable_weights))
            epoch_loss += loss
            cer += models.get_character_error_rate(actual_word, predicted_word)

        epoch_loss = int(epoch_loss)
        if (epoch % 10 == 0):
            print ('Epoch {} Loss {}  CER {}'.format(epoch, epoch_loss, cer))

    return (ts_model, ms_model, bs_model)


def test(ts_model, ms_model, bs_model, test_data):
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

        predicted_word = list()
        for i in range(1, len(out), 2):
            predicted_word.append(pos_labels_map[np.argmax(out[i])])
        predicted_word = ''.join(predicted_word)

        total_characters += len(actual_word)
        for i in range(len(actual_word)):
            if actual_word[i] == predicted_word[i]:
                correct_prediction += 1
    
    print ("Character wise accuracy {}".format(correct_prediction*100 / total_characters))
    return (0)

def find_unique_characters():
#Finding Ns and creating a map of labels and index
    _, labels = DataLoader.get_image_paths_labels()
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

def main():
    Ns, _, __ = find_unique_characters()
    train_data, test_data = DataLoader.make_train_test()

    train_count = len(train_data['n'])
    test_count = len(test_data['n'])
    print ('Total samples in training data is {} and in test data is {}'.format(train_count, test_count))
    

    ts_model = models.get_top_stream()
    bs_model = models.get_bottom_stream()
    ms_model = models.get_middle_stream(Ns)

    optimizer = models.get_optimizer()
    print (type(optimizer))

    ts_model, ms_model, bs_model = train(ts_model, ms_model, bs_model, train_data, optimizer)
    test(ts_model, ms_model, bs_model, test_data)
#Training model


if __name__ == '__main__':
    main()
