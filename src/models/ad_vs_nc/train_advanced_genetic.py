
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, GlobalAveragePooling1D, LayerNormalization, Reshape, Add, MultiHeadAttention
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def main():
    
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
        #this is created in the genetic preprocess jupyter notebook
        X_train = pd.read_pickle(os.path.join(data_dir, "X_train_vcf.pkl"))
        y_train = pd.read_pickle(os.path.join(data_dir, "y_train_vcf.pkl"))

        X_test = pd.read_pickle(os.path.join(data_dir, "X_test_vcf.pkl"))
        y_test = pd.read_pickle(os.path.join(data_dir, "y_test_vcf.pkl"))


        acc = []
        f1 = []
        precision = []
        recall = []
        seeds = random.sample(range(1, 200), 5)

        for seed in seeds:
            reset_random_seeds(seed)
            snp_input = Input(shape=(15965,))
            x = Reshape((15965, 1))(snp_input)
            x = Conv1D(filters=64, kernel_size=32, strides=32, activation='relu')(x)
            
            attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
            x = Add()([x, attn_output])
            x = LayerNormalization()(x)
            
            ffn_output = Dense(128, activation='relu')(x)
            ffn_output = Dense(64)(ffn_output)
            x = Add()([x, ffn_output])
            x = LayerNormalization()(x)
            
            x = GlobalAveragePooling1D()(x)
            output = Dense(1, activation='sigmoid')(x)
            
            model = Model(snp_input, output)

            model.compile(Adam(learning_rate = 0.001), 'binary_crossentropy', metrics = ['binary_accuracy'])


            history = model.fit(X_train, y_train,epochs=50,batch_size=32,validation_split = 0.1, verbose=1) 

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
    
