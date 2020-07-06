import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D
from keras import backend as K
import numpy as np
import random
from XrayProjekat.training_data import CATEGORIES

img_rows = 70
img_cols = 70
num_classes = 3


def Model1(input_shape):
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

def Model2(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model


def Model3(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def Model4(input_shape):
    model = Sequential()
    model.add(Conv2D(20, kernel_size=(9,9), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    pickle_inX = open("X.pickle", "rb")
    X_train = pickle.load(pickle_inX)
    pickle_inX.close()

    pickle_iny = open("y.pickle", "rb")
    y_train = pickle.load(pickle_iny)
    pickle_iny.close()

    pickle_inXtest = open("XTest.pickle", "rb")
    X_test = pickle.load(pickle_inXtest)
    pickle_inXtest.close()

    pickle_inytest = open("yTest.pickle", "rb")
    y_test = pickle.load(pickle_inytest)
    pickle_inytest.close()


    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    input_shape = X_train.shape[1:]

    #model = Model3(input_shape)

    #model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    #model.fit(X_train, y_train, batch_size=32, validation_split=0.1, epochs=10)

    #val_loss1, val_acc1 = model.evaluate(X_test, y_test)
    #print(val_loss1, val_acc1)

    #model.save('model1.model')


    new_model = tf.keras.models.load_model('model1.model')

    val_loss, val_acc = new_model.evaluate(X_test, y_test)
    print("Loss: ", val_loss)
    print("Accuracy: ", val_acc)
    predictions = new_model.predict([X_test])

    n = random.randint(0, len(y_test))
    i = np.argmax(predictions[n])
    print(CATEGORIES[i])

    X_test = np.reshape(X_test, (-1, img_cols, img_rows))
    plt.imshow(X_test[n], cmap='gray')
    plt.show()




