import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

def load_dataset():
    df = pd.read_csv('foodDiet.csv')
    feature = df[['gender','age','height','pre-weight','diet','weight6weeks']]
    target = df[['diet']]
    return feature,target

feature, target = load_dataset()

minMaxScaler = MinMaxScaler()
feature = minMaxScaler.fit_transform(feature)

ordinalEncoder = OrdinalEncoder()
feature = ordinalEncoder.fit_transform(feature)

oneHotEncoder = OneHotEncoder()
target = oneHotEncoder.fit_transform(target)

xtrain, ytrain, xtest,ytest = train_test_split(feature,target,test_size=0.3)
    
layers = {
    'input': 5,
    'hidden': 5,
    'output' : 3
}

weight = {
    'input2hidden' : tf.Variable(tf.random_normal([layers['input'],layers['hidden']])),
    'hidden2output' : tf.Variable(tf.random_normal([layers['hidden'],layers['output']])),
}

bias = {
    'hidden' : tf.Variable(tf.random_normal([layers['hidden']])),
    'output' : tf.Variable(tf.random_normal([layers['output']]))
}
# y1active = 

feature_placeHolder = tf.placeholder(tf.float32, [None, layers['input']])
target_placeHolder = tf.placeholder(tf.float32, [None, layers['output']])

def feed_forward():
    y1 = tf.matmul(feature_placeHolder,weight['input2hidden']) + bias['hidden']
    y1active = tf.nn.sigmoid(y1)

    y2 = tf.matmul(y1active,weight['hidden2output']) + bias['output']
    y2active = tf.nn.sigmoid(y2)

    return y2active
learningRate = 0.1
epoch = 1000


output = feed_forward()
error = tf.reduce_mean((0.5 * target_placeHolder - output)**2)
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(error)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1,epoch+1):
        train_dict = {
            feature_placeHolder:xtrain,
            target_placeHolder:ytrain
        }
        sess.run(optimizer,feed_dict = train_dict)
        loss = sess.run(error,feed_dict=train_dict)

        if(i%50==0):
            print(f'loss= {loss} ')
        matches = tf.equal(tf.argmax(target_placeHolder), tf.arg_max(output))
        
        test_dict = {
            feature_placeHolder:xtest,
            target_placeHolder:ytest
        }

        accuracy = tf.reduce_mean(tf.cast(matches),tf.float32)
        print(f'accuracy = {sess.run(accuracy,feed_dict=test_dict)}')