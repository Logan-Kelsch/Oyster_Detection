import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN (optional)

import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import glob
from glob import glob
import os
import re
import math

import PIL
from PIL import Image

print("---------------------------------")
print("Running TensorFlow version: " + tf.__version__)
print("---------------------------------")

def search_dir(parent):
    p = Path(parent)
    files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        files.extend(p.rglob(ext))
    return [str(f) for f in files]

fish1_path = "species_1/sample"
fish2_path = "species_2/sample"
fish3_path = "species_3/sample"
fish4_path = "species_4/sample"
fish5_path = "species_5/sample"

fish1_images = search_dir(fish1_path)
fish2_images = search_dir(fish2_path)
fish3_images = search_dir(fish3_path)
fish4_images = search_dir(fish4_path)
fish5_images = search_dir(fish5_path)

def combine_data(full_image_list, full_label_list, new_img_list, new_label):
    for file in new_img_list:
        img = Image.open(file).resize((128,128))
        img_arr = np.asarray(img)
        full_image_list.append(img_arr)
        full_label_list.append(new_label)
    return full_image_list, full_label_list

def unison_shuffled_copies(I, L):
    assert len(I) == len(L)
    p = np.random.permutation(len(I))
    return I[p], L[p]

images = []
labels = []
images_1, labels_1 = combine_data(images, labels, fish1_images, 0)
images_2, labels_2 = combine_data(images_1, labels_1, fish2_images, 1)
images_3, labels_3 = combine_data(images_2, labels_2, fish3_images, 2)
images_4, labels_4 = combine_data(images_3, labels_3, fish4_images, 3)
images_5, labels_5 = combine_data(images_4, labels_4, fish5_images, 4)

CATEGORIES = ['Dascyllus reticulatus ', 'Myripristis kuntee ', 'Hemigymnus fasciatus ', 'Neoniphon sammara ', 'Lutjanus fulvus ']
category_to_index = dict((name,index) for index,name in enumerate(CATEGORIES))
category_to_index

fish_dataset = np.array(images_5)
label_dataset = np.array(labels_5)

fish_dataset, label_dataset = unison_shuffled_copies(fish_dataset, label_dataset)
split = int(len(fish_dataset)*.1)

train_img = np.array(fish_dataset)[split:]
train_lbl = np.array(label_dataset)[split:]
test_img  = np.array(fish_dataset)[0:split]
test_lbl  = np.array(label_dataset)[0:split]

def display_images(images, labels):
    plt.figure(figsize=(15,15))
    grid_size = min(16, len(images))
    for i in range(grid_size):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(CATEGORIES[labels[i]])

#display_images(train_img, train_lbl)
#plt.show()

subtitle = 112
plt.figure()
plt.imshow(images[subtitle], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.xlabel(CATEGORIES[labels[subtitle]])
CNN_model = keras.Sequential([

    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
    activation='relu', input_shape=(128,128,3)),
    (keras.layers.MaxPooling2D(pool_size=(2,2))),
    (tf.keras.layers.Dropout(0.3)),

    (keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same',
    activation=tf.nn.relu)),
    (keras.layers.MaxPooling2D(pool_size=(2,2))),
    (keras.layers.Dropout(0.5)),

    (keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same',
    activation=tf.nn.relu)), 
    (keras.layers.MaxPooling2D(pool_size=(2,2))),
    (keras.layers.Dropout(0.5)),

    (keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same',
    activation=tf.nn.relu)),
    (keras.layers.MaxPooling2D(pool_size=(2,2))),
    (keras.layers.Dropout(0.5)),

    (keras.layers.Flatten()),
    (keras.layers.Dense(128,activation=tf.nn.relu)),
    (tf.keras.layers.Dropout(0.5)),
    keras.layers.Dense(64, activation='relu'),
    (tf.keras.layers.Dropout(0.25)),

    (keras.layers.Dense(5, activation=tf.nn.softmax))])

CNN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = CNN_model.fit(train_img,train_lbl, validation_split=0.10, shuffle=True, epochs=10)
plt.style.use('dark_background')

#Plot training & validation accuracy values

plt.plot(history.history['accuracy'],'r--',history.history['val_accuracy'],'b--')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.grid()
plt.show()

#Plot training & validation loss values

plt.plot(history.history['loss'], 'r--', history.history['val_loss'], 'b--')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid()
#plt.show()
CNN_model.evaluate(test_img, test_lbl, batch_size = 1, verbose = 1)

def make_labels( new_img_list, new_label):
    images = []
    labels = []
    for file in new_img_list:
        img = Image.open(file).resize((128,128))
        img_arr = np.asarray(img)
        images.append(img_arr)
        labels.append(new_label)
    return images, labels

test1_path = "species_1/test"
test2_path = "species_2/test"
test3_path = "species_3/test"
test4_path = "species_4/test"
test5_path = "species_5/test"

test1_images = search_dir(test1_path)
test2_images = search_dir(test2_path)
test3_images = search_dir(test3_path)
test4_images = search_dir(test4_path)
test5_images = search_dir(test5_path)

images = []
labels = []
images_test1, labels_test1 = make_labels(test1_images, 0)
images_test2, labels_test2 = make_labels(test2_images, 1)
images_test3, labels_test3 = make_labels(test3_images, 2)
images_test4, labels_test4 = make_labels(test4_images, 3)
images_test5, labels_test5 = make_labels(test5_images, 4)

import statistics
from statistics import mode
def disp_full_img (path):
    full_img = Image.open(path)
    img_arr=np.asarray(full_img)
    plt.figure(figsize=(10,10))
    grid_size = 25

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img_arr, cmap=plt.cm.binary)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}%\n ({})".format(CATEGORIES[predicted_label],
        100*np.max(predictions_array),
        CATEGORIES[true_label]), color=color)
    
def species_list(predictions_array, answers):
    print("Predictions:      Actual:")
    picture=0
    for picture in range (0, len(predictions_array)):
        img_guess_num = np.argmax(predictions_array[picture])
        img_guess_name = CATEGORIES[img_guess_num]
        print(img_guess_name, "     ", answers)
        picture + 1

def species_guess(predictions_array):
    results = []
    img_guess_num = np.argmax(predictions_array)
    picture = 0
    for picture in range (0, len(predictions_array)):
        img_guess_num = np.argmax(predictions_array[picture])
        results.append(img_guess_num)
        picture + 1
    final_guess = mode(results)
    return CATEGORIES[final_guess]


#plt.style.use('classic')
test_set = np.array(images_test5)
test_labels = np.array(labels_test5)
predictions = CNN_model.predict(test_set)

num_rows = 5
num_cols = 6
num_images = num_rows*num_cols
#plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#for i in range(num_images):
#    plt.subplot(num_rows, 2*num_cols, 2*i+1)
#    plot_image(i, predictions, test_labels, test_set)
#plt.show()




