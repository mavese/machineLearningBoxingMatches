import pandas as pd
import numpy as np
import tensorflow as tf

numFeatures = 18
numClasses = 2
batchSize = 50

def main():
    # reading data files and selecting features and labels
    df = pd.read_csv('./BoxingData.csv')
    df = df.iloc[:,0:19].dropna().reindex()

    # declaring encoding functions to turn strings to integers
    def labelOneHotEncoder(r,column):
        if r == 'win_A':
            switcher = {1:1, 2:0, 3:0}
            return switcher.get(column, 0)
        elif r == 'win_B':
            switcher = {1:0, 2:1, 3:0}
            return switcher.get(column, 0)
        else:
            switcher = {1:0, 2:0, 3:1}
            return switcher.get(column, 0)

    def stanceEncoder(r):
        if r == 'orthodox':
            return 0
        else:
            return 1

    # encoding strings in the data, the label and the stance columns
    df['label_1'] = df['result'].apply(lambda r: labelOneHotEncoder(r,1))
    df['label_2'] = df['result'].apply(lambda r: labelOneHotEncoder(r,2))
    df['label_3'] = df['result'].apply(lambda r: labelOneHotEncoder(r,3))
    df['stance_A'] = df['stance_A'].apply(stanceEncoder)
    df['stance_B'] = df['stance_B'].apply(stanceEncoder)

    # seperate features and labels
    data = df.iloc[:,0:18]
    labels = df.iloc[:,19:22]

    # convert pandas dataframe's to numpy nDarray's
    npData = data.values
    npLabels = labels.values

    # randomly select 80% of the data to train on
    randomArr = np.random.random_integers(0, high=npData.shape[0]-1, size=int(round(.8*npData.shape[0])))
    xTrain = npData[randomArr,:]
    yTrain = npLabels[randomArr,:]
    
    # call function to train the network on the training data, this returns the trained model
    model = trainNetwork(xTrain, yTrain)
        


def trainNetwork(xTrain, yTrain):
    # setting up layers
    inputs = tf.keras.layers.Input(shape=[18])
    x = tf.keras.layers.Dense(units=10, activation='relu')(inputs)
    x = tf.keras.layers.Dense(units=10, activation='relu')(x)
    output = tf.keras.layers.Dense(units=3, activation='softmax')(x)

    # setting up the model, compiling and fitting to the given data
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=xTrain, y=yTrain, batch_size=32, epochs=10)

    return model
    

if __name__ == '__main__':
    main()