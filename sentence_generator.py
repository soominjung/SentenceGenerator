import tensorflow as tf

import numpy as np
import os
import argparse

import reader

#tf.reset_default_graph()

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--mode", type=str, help="train/generate")
parser.add_argument("--model", type=str, default="model", help="model/pretrained")
arg = parser.parse_args()

raw = reader.read_data("data/ptb.train.txt")
word2idx, idx2word = reader.build_vocab(raw)

# hyperparameters
hidden_layer = 128
num_classes = len(word2idx)
seq_length = []
learning_rate = 0.01
batch_size = 512

datasetX, datasetY, seq_len_set, max_seq_len, n_data = reader.create_dataset(raw, word2idx, idx2word, batch_size)

# build a model
X = tf.placeholder(tf.int32, [None, max_seq_len])
Y = tf.placeholder(tf.int32, [None, max_seq_len])
seq_len = tf.placeholder(tf.int32, [None])

n_seq = tf.shape(X)[0]
X_one_hot = tf.one_hot(X, num_classes)

cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_layer, state_is_tuple=True)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_layer, state_is_tuple=True)

multi_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(multi_cell, X_one_hot, sequence_length=seq_len, dtype=tf.float32)

W = tf.Variable(tf.random_normal([hidden_layer, num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))

outputs = tf.reshape(outputs, [-1, hidden_layer])
outputs = tf.matmul(outputs, W)+b
model = tf.reshape(outputs, [n_seq, -1, num_classes])

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(model, axis=2)

if arg.mode == 'train':
    print("!!Start learning!!")
    saver = tf.train.Saver(tf.global_variables(), reshape=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for i in range(20):
        for j in range(n_data//batch_size):
            batch_X, batch_Y, batch_seq = sess.run([datasetX, datasetY, seq_len_set])
            _, l = sess.run([optimizer, cost], feed_dict={X:batch_X, Y:batch_Y, seq_len:batch_seq})
            print(i, j, "cost: ", l)
        saver.save(sess, 'model/train', global_step=i, max_to_keep=5)

if arg.mode == 'generate':
    # use pretrained model to predict word sequences
    print("!!Start prediction!!")
    saver = tf.train.Saver(tf.global_variables(), reshape=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(arg.model))

    while True:
        input_words = input("Enter words(enter '!' to exit): ")
        if input_words == '!':
            print("Bye!")
            break
        input_words = input_words.lower()
        input_words = input_words.split()
    
        flag = True
        for input_word in input_words:
            if input_word not in word2idx:
                print("word %s not predictable" % input_word)
                flag = False
                break
        if flag is False:
            continue
        
        # insert zeros before the first word
        results = []
        input_idx = [word2idx[input_word] for input_word in input_words]
        results.extend(input_idx)
        len_input = len(input_idx)
        while len(input_idx)<max_seq_len:
            input_idx.append(0)
    
        # predict words till it gets '<eos>'
        result_idx = len_input-1
        while True:
            result = sess.run(prediction, feed_dict={X:[input_idx], seq_len:[len(input_idx)]})
            if result[0][result_idx] == 0:
                break
            results.append(result[0][result_idx])
            if result_idx<max_seq_len-1:
                input_idx.insert(result_idx+1, result[0][result_idx])
                del input_idx[-1]
                result_idx+=1
            else:
                input_idx = results[-max_seq_len:]
    
        predicted_word = [idx2word[idx] for idx in results]
        print('predicted_sentence: ', ' '.join(predicted_word))