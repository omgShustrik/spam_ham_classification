#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:47:44 2017

@author: Rustem
"""
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Start a graph
sess = tf.Session()

# Set RNN parameters
epochs = 70
batch_size = 250
max_sequence_length = 25
rnn_size = 50
embedding_size = 50
learning_rate = 0.0003
dropout_keep_prob = tf.placeholder(tf.float32)

def get_messages_from_folder(root_data_dir, data_dir):
    text_data = []
    list_of_txt = filter(lambda x: x.endswith('.txt'),
                         os.listdir(os.path.join(root_data_dir, data_dir)))
    for txt in list_of_txt:
        text_data.append(os.path.join(root_data_dir, data_dir, txt))
    return(text_data)

def read_messages_from_txt(data, text_class):
    text_data = []
    for d in data:
        try:
            text = open(d, 'r').read()
            text = text.encode('ascii',errors='ignore').strip()
            text = text.decode()#.strip('\n')
            text = re.sub('\s+', ' ', text)
            text_data.append(text_class+'\t'+text)
        except Exception:
            pass
    return(text_data)

def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)

def get_vector(text_string):
    text_vector = [dictionary.get(x) for x in text_string.split(' ')
                                              if dictionary.get(x) is not None]
    return(text_vector)

def get_ids(text_string, vocab_dict):
    ids_vector = [vocab_dict.get(x) for x in text_string.split(' ')]
    return(ids_vector)
    
# Open dictionary
dict_dir = 'temp'
dict_file = 'result_256.json'
json_file = json.load(open(os.path.join(dict_dir, dict_file)))
dictionary = json_file['vocabulary']
dictionary = dict([(key, np.array(value, dtype=np.float32)) for key, value in dictionary.items()])

vocabulary_size = json_file['vocabularySize']
vector_size = json_file['vectorSize']
# Open and join data
root_data_dir = 'enron_all_'
spam_data_dir = 'spam'
ham_data_dir = 'ham'
all_text_data = []
spam_data = get_messages_from_folder(root_data_dir, spam_data_dir)
ham_data = get_messages_from_folder(root_data_dir, ham_data_dir)
all_text_data.extend(read_messages_from_txt(spam_data, 'spam'))
all_text_data.extend(read_messages_from_txt(ham_data, 'ham'))

#median_lenght = int(np.median(list(map(lambda x: len(x.split(' ')), all_text_data))))
#reduced_text_data = [x if len(x.split(' ')) < median_lenght else x[:median_lenght] for x in all_text_data]
# Separate to target and text
text_data = [x.split('\t') for x in all_text_data if len(x)>=1]

[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

# Clean texts
text_data_train = [clean_text(x) for x in text_data_train]

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))


# Shuffle and split data
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])
np.random.seed([111])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
print(shuffled_ix)
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled)*0.9)
_x, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
_y, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("90-10 Train Test split: {:d} -- {:d}".format(len(_y), len(y_test)))

# Create placeholders
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
embeddings = np.random.uniform(-1,1,(len(vocab_processor.vocabulary_),vector_size))
for key, value  in vocab_dict.items():
    if key in dictionary:
        embeddings[value] = dictionary[key]
embeddings = np.array(embeddings, dtype = np.float32)
text_processed = [get_ids(x, vocab_dict) for x in text_data_train]


# Create embedding
#embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
#embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
embedding_output = tf.nn.embedding_lookup(embeddings, x_data)
#embedding_output_expanded = tf.expand_dims(embedding_output, -1)

# Define the RNN cell

cell=tf.contrib.rnn.BasicRNNCell(num_units = rnn_size)
#cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)


weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.matmul(last, weight) + bias

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, 
                                                        labels=y_output) # logits=float32, labels=int32
loss = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), 
                                           tf.cast(y_output, tf.int64)),
                                           tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
val_loss = []
test_loss = []
train_accuracy = []
val_accuracy = []
test_accuracy = []
# Start training
for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(_x)))
    _x = _x[shuffled_ix]
    _y = _y[shuffled_ix]
    
    ix_cutoff = int(len(_y)*0.9)
    x_train, x_val = _x[:ix_cutoff], _x[ix_cutoff:]
    y_train, y_val = _y[:ix_cutoff], _y[ix_cutoff:]

    num_batches = int(len(x_train)/batch_size) + 1

    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        
        # Run train step
        train_dict = {x_data: x_train_batch,
                      y_output: y_train_batch, 
                      dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
        
    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy],
                                               feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    
     # Run Validation Step
    val_dict = {x_data: x_val, 
                y_output: y_val, 
                dropout_keep_prob:1.0}
    temp_val_loss, temp_val_acc = sess.run([loss, accuracy], 
                                           feed_dict=val_dict)
    val_loss.append(temp_val_loss)
    val_accuracy.append(temp_val_acc)
    
    # Run Eval Step
    test_dict = {x_data: x_test, 
                 y_output: y_test, 
                 dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], 
                                             feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Val Acc: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_val_acc ,temp_test_acc))
    
# Plot loss over time
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, val_loss, 'r-.', label='Validation Set')

plt.plot(epoch_seq, test_loss, 'g-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, val_accuracy, 'r-.', label='Validation Set')
plt.plot(epoch_seq, test_accuracy, 'g-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()