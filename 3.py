import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.compat.v1.disable_eager_execution()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
# OneHotEncoder -> buat output, ini panjang, jadi binary
# OrdinalEncoder -> buat input, ini pendek
from sklearn.model_selection import train_test_split

def load_dataset():
    df = pd.read_csv("foodDiet.csv")
    feature = df[['gender', 'age', 'height', 'pre-weight', 'weight6weeks']]
    target = df[['diet']]

    return feature, target

feature, target = load_dataset()

# Normalization dataset and target
minmaxscaler = MinMaxScaler()
feature = minmaxscaler.fit_transform(feature)

ordinalencoder = OrdinalEncoder()
feature = ordinalencoder.fit_transform(feature)

onehotencoder = OneHotEncoder(sparse=False)
target = onehotencoder.fit_transform(target)

#
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)

layers = {
    # Number of input node
    'input': 5,
    # Hidden layer node
    'hidden': 5,
    # Kemungkinan ada berapa unique outputnya
    'output': 3
}

weight = {
    'input2hidden' : tf.Variable(tf.random.normal([layers['input'], layers['hidden']])),
    'hidden2output' : tf.Variable(tf.random.normal([layers['hidden'], layers['output']]))
}

bias = {
    'hidden' : tf.Variable(tf.random.normal([layers['hidden']])),
    'output' : tf.Variable(tf.random.normal([layers['output']]))
}

feature_placeholder = tf.placeholder(tf.float32, [None, layers['input']])
target_placeholder = tf.placeholder(tf.float32, [None, layers['output']])

def feed_forward():
    y1 = tf.matmul(feature_placeholder, weight['input2hidden']) + bias['hidden']
    y1act = tf.nn.sigmoid(y1)

    print(y1, y1act)

    y2 = tf.matmul(y1act, weight['hidden2output']) + bias['output']
    y2act = tf.nn.sigmoid(y2)

    return y2act

learning_rate = 0.1
epoch = 1000

output = feed_forward()
error = tf.reduce_mean((0.5 * target_placeholder - output) ** 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error) # Back propagationnya

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        train_dictionary = {
            feature_placeholder: x_train,
            target_placeholder: y_train
        }

        sess.run(optimizer, feed_dict=train_dictionary)
        loss = sess.run(error, feed_dict = train_dictionary)
        if(i % 50 == 0):
            print(f"Loss: {loss}")

    matches = tf.equal(tf.argmax(target_placeholder), tf.argmax(output))
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    test_dict = {
        feature_placeholder: x_test,
        target_placeholder: y_test
    }
    print(f"Accuracy = {sess.run(accuracy, feed_dict=test_dict) * 100}")