import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

img_rows = 50
img_cols = 50
num_classes = 3

input_shape = (100, 100, 1)

def get_Model1():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model

def get_Model2():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))

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

    pickle_inXtest = open("XTest.pickle", "rb")
    X_test = pickle.load(pickle_inXtest)
    pickle_inXtest.close()

    pickle_inytest = open("yTest.pickle", "rb")
    y_test = pickle.load(pickle_inytest)
    pickle_inytest.close()

    #plt.imshow(X_train[0])
    #plt.show()

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    #print(X_train[0])

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        x_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        x_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    model = get_Model1()

    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #loss_fn = keras.losses.sparse_categorical_crossentropy
    #model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    #mnist = tf.keras.datasets.mnist
    #(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    #xtrain = tf.keras.utils.normalize(xtrain, axis=1)
    #xtest = tf.keras.utils.normalize(xtest, axis=1)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])


    # treniranje i evaluacija







    #print(X_test[0][0].size)
    #print(y_test.size)
    print()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #model.fit(X_train, y_train, batch_size=128, epochs=3, verbose=1, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, epochs=3)


    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss, val_acc)


