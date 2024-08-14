import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib
import numpy as np
import pandas as pd

people = fetch_lfw_people(min_faces_per_person = 20, resize = 0.7)
image_shape = people.images[0].shape

print("people.images.shape : {}".format(people.images.shape))
print("Class : {}".format(people.target_names.size))

fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.show()

counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print('{0:30} {1:3}'.format(name, count))

min_count = np.argmin(counts)
print(people.target_names[min_count] + " " + str(counts[min_count]))

idx = np.zeros(people.target.shape, np.bool_)
for target in np.unique(people.target):
    idx[np.where(people.target == target)[0][:min_count]] = 1

x_people = people.data[idx]
y_people = people.target[idx]

pca = PCA(n_components=100, whiten=True, random_state=0) 
pca.fit_transform(x_people)
x_pca = pca.transform(x_people)

print('pca.components_.shape : {}'.format(pca.components_.shape)) 

fig, axes = plt.subplots(3, 5, subplot_kw={'xticks': (), 'yticks': ()})

for i, (comp, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(comp.reshape(image_shape))
    ax.set_title('pricipal component {}'.format(i+1))
plt.show()

scaler = MinMaxScaler()
x_people_scaled = scaler.fit_transform(x_people)

x_train, x_test, y_train, y_test = train_test_split(x_people_scaled, y_people, stratify=y_people, random_state=0) 

def pca_faces(X_train, X_test):
    reduced_images = []
    for n_components in [10, 25, 50, 100]:
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_test_pca = pca.transform(X_test)
        X_test_back = pca.inverse_transform(X_test_pca)
        reduced_images.append(X_test_back)
    return reduced_images

def plot_pca_faces(X_train, X_test, image_shape):
    reduced_images = pca_faces(X_train, X_test)
    fix, axes = plt.subplots(3, 5, figsize=(15, 12),
                             subplot_kw={'xticks': (), 'yticks': ()})
    for i, ax in enumerate(axes):
        ax[0].imshow(X_test[i].reshape(image_shape),
                     vmin=0, vmax=1)
        for a, X_test_back in zip(ax[1:], reduced_images):
            a.imshow(X_test_back[i].reshape(image_shape), vmin=0, vmax=1)

    axes[0, 0].set_title("original image")
    for ax, n_components in zip(axes[0, 1:], [10, 25, 50, 100]):
        ax.set_title("%d components" % n_components)

pca = PCA(n_components=100, whiten=False, random_state=0)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train_pca, y_train)

print('knn.score(x_test, y_test) : {:.3f}'.format(knn.score(x_test_pca, y_test))) 

pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train_pca, y_train)

print('knn.score(x_test, y_test) : {:.3f}'.format(knn.score(x_test_pca, y_test))) 

X_people = people.images
Y_people = people.target

X_people = X_people.reshape(X_people.shape[0], image_shape[0], image_shape[1], 1)
X_people.shape

face_labels = to_categorical(Y_people)

X_train, X_test, Y_train, Y_test = train_test_split(X_people, face_labels, stratify=face_labels, random_state=0) 

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_shape[0], image_shape[1], 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(people.target_names.size, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

print("Test loss = {}".format(score[0]))
print("Test accuracy = {}".format(score[1]))