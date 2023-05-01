import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def load_dataset():
    df = pd.read_csv('arrival_from_australia.csv', index_col='Date')
    return df

dataset = load_dataset()

num_input = 1
num_output = 1
context_unit = 3

# Untuk nentuin suatu data, kita perlu berapa data sebelumnya
time_seq = 3

test_size = 30

train_data = dataset[:round(len(dataset) * 0.7)]
test_data = dataset[len(train_data):]

minMaxScaler = MinMaxScaler()
train_data = minMaxScaler.fit_transform(train_data)
test_data = minMaxScaler.fit_transform(test_data)

cell = tf.nn.rnn_cell.BasicRNNCell(context_unit, activation=tf.nn.relu)
cell = tf.estimator.rnn.OutputProjectionWrapper(cell, output_size=num_output, activation=tf.nn.relu)

feature_placeholder = tf.placeholder(tf.float32, [None, time_seq, num_input])
target_placeholder = tf.placeholder(tf.float32, [None, time_seq, num_output])

output, _ = tf.nn.dynamic_rnn(cell, feature_placeholder, dtype=tf.float32)

error = tf.reduce_mean(0.5 * (target_placeholder - output) ** 2)

learning_rate = 0.1
optimizer = tf.train.AdamOptimizer(learning_rate)

epoch = 1000
batch_size = 3

def next_batch(dataset, batch_size):
    x_batch = np.zeros([batch_size, time_seq, num_input])
    y_batch = np.zeros([batch_size, time_seq, num_input])

    for i in range(len(batch_size)):
        start = np.random.randint(0, len(dataset) - time_seq)
        x_batch[i] = dataset[start:start+time_seq]
        y_batch[i] = dataset[start+1:start+1+time_seq]

    # x_batch = input
    # y_batch = output
    return x_batch, y_batch

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        x_batch, y_batch = next_batch(dataset, batch_size)
        train_data = {
            feature_placeholder: x_batch,
            target_placeholder: y_batch
        }

        sess.run(optimizer, feed_dict=train_data)

        if(i % 50 == 0):
            loss = sess.run(error, feed_dict=train_data)
            print(f"Loss = {loss * 100}%")