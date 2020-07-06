import pandas as pd
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle

data_folder = Path("chest_xray_data_set/metadata")
file = data_folder / "chest_xray_metadata.csv"
f = open(file)

# print(f.read())

d = pd.read_csv(f, index_col=0,
                header=0)  # , names=["", "X_ray_image_name", "Label", "Label_2_Virus_category", "Label_1_Virus_category"], header=0)
# print(d)
# d.info()

CATEGORIES = ['Normal', 'Virus', 'bacteria']
NAMES = d['X_ray_image_name']
LABELS = d['Label']
LABEL1 = d['Label_1_Virus_category']

training_data = []

IMG_SIZE = 50


def get_class(img):
    for name in NAMES:
        if name == img:
            ind = NAMES.tolist().index(name)
            l = LABELS.tolist()[ind]
            #print(l)
            if(l == 'Pnemonia'):
                l = LABEL1.tolist()[ind]
            for c in CATEGORIES:
                if l == c:
                    return CATEGORIES.index(c)
            return None



def create_training_data():
    data_folder2 = Path("chest_xray_data_set")
    ol = os.listdir(data_folder2)
    ol.remove("metadata")
    print(len(ol))


    for img in ol:
        class_num = get_class(img)
        if class_num == None:
            #print("none")
            continue
        else:
            img_array = cv2.imread(os.path.join(data_folder2, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            #plt.imshow(new_array, cmap='gray')
            #plt.show()
            training_data.append([new_array, class_num])



if __name__ == '__main__':

    #myset = set(LABELS)
    #CATEGORIES = list(myset)
    print(CATEGORIES)
    print(len(NAMES))

    create_training_data()
    print(len(training_data))
    #print(training_data[0])
    random.shuffle(training_data)
    for sample in training_data[:20]:
        print(sample[1])

    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)
    arr = np.array(X)
    X = np.reshape(arr, (-1, IMG_SIZE, IMG_SIZE, 1))
    #X = np.array(X).reshape(-1, (IMG_SIZE, IMG_SIZE, 1))
    print(len(X))

    y = np.array(y)

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()



