import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import time
import os


df = pd.read_csv("clean_train.csv")


X, y = df.iloc[:, 2:], df.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



def train_model(X_train, y_train, X_test, y_test, iterations=100, epochs = 100, batch_size=32, model_arg=None):
    if model_arg == None:
        model = Sequential()
        model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model = model_arg
    accuracy_array = []
    for i in tqdm(range(iterations)):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
        accuracy_array.append(model.evaluate(X_test, y_test, verbose=0)[1])
    return model, accuracy_array

def plot_accuracy(accuracy_array):
    plt.plot(accuracy_array)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    
    #add a trendline
    plt.plot(np.poly1d(np.polyfit(range(len(accuracy_array)), accuracy_array, 1))(range(len(accuracy_array))))
    print("poly1d:", np.poly1d(np.polyfit(range(len(accuracy_array)), accuracy_array, 1)))
    
    #set the y axis to start at 0 and end at 1
    plt.ylim(0, 1)
    
    plt.show()

if os.path.exists("model.h5"):
    model = load_model("model.h5")
    print("model loaded")
else:
    model = None
    print("model not loaded")

t1 = time.time()

iterations = 40
epochs = 50
batch_size = 16


model, accuracy_array = train_model(X_train, y_train, X_test, y_test, iterations=iterations, epochs=epochs, batch_size=batch_size, model_arg = model)

_, accuracy = model.evaluate(X_test, y_test)

print("time:", str(int((time.time() - t1)*1000)) + "ms")

print("Accuracy: ", accuracy)

accuracy_anomalys = [i for i in accuracy_array if i < accuracy * 0.8]



print(accuracy_anomalys)

plot_accuracy(accuracy_array)

print("iterations:", iterations, "epochs:", epochs, "batch_size:", batch_size)

print("total:", iterations * epochs)