import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras


if __name__ == '__main__':
    pickle_inX = open("X.pickle", "rb")
    X_train = pickle.load(pickle_inX)
    #print(len(X_train))
    #print(X_train[0])
    pickle_inX.close()

    pickle_iny = open("y.pickle", "rb")
    y_train = pickle.load(pickle_iny)
    #print(len(y_train))
    pickle_iny.close()

    #plt.imshow(X_train[0])
    #plt.show()

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    #print(X_train[0])

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #loss_fn = keras.losses.sparse_categorical_crossentropy
    #model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    #mnist = tf.keras.datasets.mnist
    #(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    #xtrain = tf.keras.utils.normalize(xtrain, axis=1)
    #xtest = tf.keras.utils.normalize(xtest, axis=1)

    model.fit(X_train, y_train, epochs=3)

    #val_loss, val_acc = model.evaluate()


