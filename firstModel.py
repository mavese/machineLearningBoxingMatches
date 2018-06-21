import pandas as pd
import numpy as np
import tensorflow as tf

numFeatures = 18
numClasses = 2
batchSize = 50

def main():
    df = pd.read_csv('./BoxingData.csv')
    df = df.iloc[:,0:19].dropna().reindex()

    def labelEncode(r):
        if r == 'win_A':
            return 0
        elif r == 'win_B':
            return 1
        else:
            return 2

    df['label'] = df['result'].apply(labelEncode)
    print df


def trainNetwork():
    X = tf.placeholder(tf.float32, [batchSize,numFeatures])
    y = tf.placeholder(tf.float32, [batchSize, 1])

    W = tf.Variable(tf.random_normal([numFeatures, 1], stddev=.1))
    b = tf.Variable(tf.random_normal([1,1]))

    a = tf.add(tf.matmul(X,W), b)
    

if __name__ == '__main__':
    main()