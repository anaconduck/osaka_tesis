import os
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Input


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def resnet_block(x, filters, stride=1):
    shortcut = x
    x = Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def main():
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
    with open(os.path.join(data_dir, "img_train.pkl"), "rb") as fh:
        data = pickle.load(fh)
    X_train_ = pd.DataFrame(data)["img_array"] 
    
    with open(os.path.join(data_dir, "img_test.pkl"), "rb") as fh:
        data = pickle.load(fh)
    X_test_ = pd.DataFrame(data)["img_array"]
    
    with open(os.path.join(data_dir, "img_y_train.pkl"), "rb") as fh:
        data = pickle.load(fh)
    y_train = np.array(pd.DataFrame(data)["label"].values.astype(np.float32)).flatten()
    
    with open(os.path.join(data_dir, "img_y_test.pkl"), "rb") as fh:
        data = pickle.load(fh)
    y_test = np.array(pd.DataFrame(data)["label"].values.astype(np.float32)).flatten()
    

    y_test[y_test == 2] = -1
    y_test[y_test == 1] = 2
    y_test[y_test == -1] = 1
    
    y_train[y_train == 2] = -1
    y_train[y_train == 1] = 2
    y_train[y_train == -1] = 1
    

    X_train = []
    X_test = []
    
    for i in range(len(X_train_)):
        X_train.append(X_train_.values[i])
        
    for i in range(len(X_test_)):
        X_test.append(X_test_.values[i])
    
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    
    acc = []
    f1 = []
    precision = []
    recall = []
    seeds = random.sample(range(1, 200), 5)
    for seed in seeds:
        reset_random_seeds(seed)
        img_input = Input(shape=(72, 72, 3))
        x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        x = resnet_block(x, 64)
        x = resnet_block(x, 64)
        x = resnet_block(x, 128, stride=2)
        x = resnet_block(x, 128)
        x = resnet_block(x, 256, stride=2)
        x = resnet_block(x, 256)
        x = resnet_block(x, 512, stride=2)
        x = resnet_block(x, 512)
        
        x = GlobalAveragePooling2D()(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(img_input, output)
        
        
        model.compile(Adam(learning_rate = 0.001), 'binary_crossentropy', metrics = ['binary_accuracy'])
        
        model.summary()
        
    
        history = model.fit(X_train, y_train, epochs=50, batch_size=32,validation_split=0.1, verbose=1) 
        
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        acc.append(score[1])
        
        test_predictions = model.predict(X_test)
        test_label = y_test

        true_label = test_label

        predicted_label = (test_predictions > 0.5).astype(int)
        
        cr = classification_report(true_label, predicted_label, output_dict=True)
        precision.append(cr["macro avg"]["precision"])
        recall.append(cr["macro avg"]["recall"])
        f1.append(cr["macro avg"]["f1-score"])
    
    print("Avg accuracy: " + str(np.array(acc).mean()))
    print("Avg precision: " + str(np.array(precision).mean()))
    print("Avg recall: " + str(np.array(recall).mean()))
    print("Avg f1: " + str(np.array(f1).mean()))
    print("Std accuracy: " + str(np.array(acc).std()))
    print("Std precision: " + str(np.array(precision).std()))
    print("Std recall: " + str(np.array(recall).std()))
    print("Std f1: " + str(np.array(f1).std()))
    print(acc)
    print(precision)
    print(recall)
    print(f1)
    

    
if __name__ == '__main__':
    main()
    
    
