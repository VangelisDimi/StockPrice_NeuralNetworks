#Supress unrelated warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import pandas as pd
import math
from tensorflow import keras
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import LSTM,RepeatVector,TimeDistributed,Input,Conv1D,UpSampling1D,MaxPooling1D
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class multiLayer_LSTM():
    def __init__(self, dataset, batch_size=3, num_epochs=10, num_layers=3, num_units=50, dropout_rate=0.2, window=60, train_size = 0.8):
        #Initialize network
        self.dataset = dataset
        self.num_units = num_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.window = window
        self.checkpoint = "models/multilayer_lstm.pkl"

        self.data_size=self.dataset.shape[1]-1
        self.train_size=math.floor(self.data_size*train_size)
        self.test_size=self.data_size-self.train_size

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        #Format data
        #Create list of training sets
        self.X_train=[]
        self.y_train=[]
        self.dataset_train=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, 1:self.train_size+1].values
            _dataset= np.array([_dataset]).T
            _dataset = self.scaler.fit_transform(_dataset)
            self.dataset_train.append(_dataset)

            X_t = []
            y_t = []
            for j in range(self.window,self.train_size):
                X_t.append(_dataset[j-self.window:j, 0])
                y_t.append(_dataset[j, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            
            self.X_train.append(X_t)
            self.y_train.append(y_t)
        
        #Create list of testing sets
        self.X_test=[]
        self.y_test=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, self.train_size+1:].values
            _dataset = np.array([_dataset]).T
            _dataset = self.scaler.transform(_dataset)

            X_t = []
            y_t = []
            for j in range(self.window,self.test_size):
                X_t.append(_dataset[j-self.window:j, 0])
                y_t.append(_dataset[j, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            
            self.X_test.append(X_t)
            self.y_test.append(y_t)
        

        #Create list of testing sets
        self.dataset_test=[]
        for i in range(len(self.dataset)):
            _dataset = self.dataset.iloc[i, self.train_size+1:].values
            _dataset = np.array([_dataset]).T
            self.dataset_test.append(_dataset)


        #Create network
        self.model = Sequential()
        #First Layer
        self.model.add(LSTM(units = self.num_units, return_sequences = True, input_shape = (self.X_train[0].shape[1], 1)))
        self.model.add(Dropout(self.dropout_rate))
        #Add layers
        for i in range(self.num_layers):
            self.model.add(LSTM(units = self.num_units, return_sequences = True))
            self.model.add(Dropout(self.dropout_rate))
        self.model.add(LSTM(units = self.num_units))
        self.model.add(Dropout(self.dropout_rate))
        #Output Layer
        self.model.add(Dense(units = 1))
        #Compile
        self.model.compile(optimizer = 'adam', loss = 'mae')

    def fit(self):
        #Fit all datasets to model
        for i in range(len(self.X_train)):
            print("Fitting: ",i+1,"/",len(self.X_train))
            self.model.fit(self.X_train[i], self.y_train[i],epochs=self.num_epochs,batch_size=self.batch_size)

    def predict(self,i):
        predicted_stock_price = self.model.predict(self.X_test[i])
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)

        predict_df = pd.DataFrame(index=self.dataset.columns[self.train_size+1+self.window:])
        predict_df['predicted_close'] = predicted_stock_price

        _dataset = self.dataset.iloc[i, self.train_size+1:].values
        _dataset = np.array([_dataset]).T
        predict_df['close'] = _dataset[self.window:]

        return predict_df
    
    def score(self):
        X_pred = self.model.predict(self.X_train)
        return accuracy_score(self.y_train, X_pred)

    def save(self,path=None):
        if path is None:
            path=self.checkpoint
        self.model.save(path)
    
    def load(self,path=None):
        if path is None:
            path=self.checkpoint
        self.model = keras.models.load_model(path)



class LSTM_autoencoder():
    def __init__(self, dataset, mae=0.65, batch_size=3, num_epochs=10, num_layers=2, num_units=50, dropout_rate=0.2, window=60, train_size = 0.8):
        #Initialize network
        self.dataset = dataset
        self.num_units = num_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.window = window
        self.mae = mae
        self.checkpoint = "models/lstm_autoencoder.pkl"

        self.data_size=self.dataset.shape[1]-1
        self.train_size=math.floor(self.data_size*train_size)
        self.test_size=self.data_size-self.train_size

        #Create and fit scaler
        self.scaler = StandardScaler()
        for i in range(len(self.dataset)):
            _dataset = self.dataset.iloc[i, 1:self.train_size+1].values
            _dataset= np.array([_dataset]).T
            self.scaler = self.scaler.fit(_dataset)

        #Format data
        #Create list of training sets
        self.X_train=[]
        self.y_train=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, 1:self.train_size+1].values
            _dataset= np.array([_dataset]).T
            _dataset = self.scaler.transform(_dataset)
            
            X_t = []
            y_t = []
            for j in range(self.window,self.train_size):
                X_t.append(_dataset[j-self.window:j, 0])
                y_t.append(_dataset[j, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            
            self.X_train.append(X_t)
            self.y_train.append(y_t)

        #Create list of testing sets
        self.X_test=[]
        self.y_test=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, self.train_size+1:].values
            _dataset = np.array([_dataset]).T
            _dataset = self.scaler.transform(_dataset)

            X_t = []
            y_t = []
            for j in range(self.window,self.test_size):
                X_t.append(_dataset[j-self.window:j, 0])
                y_t.append(_dataset[j, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            
            self.X_test.append(X_t)
            self.y_test.append(y_t)

        #Create network
        self.model = Sequential()
        #First Layer
        self.model.add(LSTM(units=self.num_units,input_shape=(self.X_train[0].shape[1],1)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(RepeatVector(n=self.X_train[0].shape[1]))
        #Add layers
        for i in range(self.num_layers):
            self.model.add(LSTM(units = self.num_units, return_sequences = True))
            self.model.add(Dropout(self.dropout_rate))
        #Output Layer
        self.model.add(TimeDistributed(Dense(units=1)))
        #Compile
        self.model.compile(loss='mae', optimizer='adam')

    def fit(self):
        #Fit all datasets to model
        for i in range(len(self.X_train)):
            print("Fitting: ",i+1,"/",len(self.X_train))
            self.model.fit(self.X_train[i], self.y_train[i],epochs=self.num_epochs,batch_size=self.batch_size)

    def predict(self,i):
        X_pred = self.model.predict(self.X_test[i])
        test_mae_loss = np.mean(np.abs(X_pred - self.X_test[i]), axis=1)

        test_score_df = pd.DataFrame(index=self.dataset.columns[self.train_size+1+self.window:])
        test_score_df['loss'] = test_mae_loss
        test_score_df['threshold'] = self.mae
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold

        _dataset = self.dataset.iloc[i, self.train_size+1:].values
        _dataset = np.array([_dataset]).T
        test_score_df['close'] = _dataset[self.window:]

        return test_score_df

    def score(self):
        X_pred = self.model.predict(self.X_train)
        return accuracy_score(self.y_train, X_pred)

    def save(self,path=None):
        if path is None:
            path=self.checkpoint
        self.model.save(path)
    
    def load(self,path=None):
        if path is None:
            path=self.checkpoint
        self.model = keras.models.load_model(path)


class CNN_autoencoder():
    def __init__(self, dataset, latent_dim=3, batch_size=3, num_epochs=10, window=10, train_size=0.8):
        #Initialize network
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.window = window
        self.latent_dim = latent_dim
        self.checkpoint = "models/cnn_autoencoder.pkl"

        self.data_size=self.dataset.shape[1]-1
        self.train_size=math.floor(self.data_size*train_size)
        self.test_size=self.data_size-self.train_size

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        #Format data
        #Create list of training sets
        self.X_train=[]
        self.y_train=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, 1:self.train_size+1].values
            _dataset= np.array([_dataset]).T
            _dataset = self.scaler.fit_transform(_dataset)
            
            X_t = []
            y_t = []
            for j in range(self.window,self.train_size,self.window):
                X_t.append(_dataset[j-self.window:j, 0])
                y_t.append(_dataset[j, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            y_t = np.reshape(y_t, (y_t.shape[0], 1))

            self.X_train.append(X_t.astype('float32'))
            self.y_train.append(y_t.astype('float32'))

        #Create list of testing sets
        self.X_test=[]
        self.y_test=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, self.train_size+1:].values
            _dataset = np.array([_dataset]).T
            _dataset = self.scaler.fit_transform(_dataset)

            X_t = []
            y_t = []
            for j in range(self.window,self.test_size,self.window):
                X_t.append(_dataset[j-self.window:j, 0])
                y_t.append(_dataset[j, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            y_t = np.reshape(y_t, (y_t.shape[0], 1))

            self.X_test.append(X_t.astype('float32'))
            self.y_test.append(y_t.astype('float32'))

        #Create network
        #Input
        input = Input(shape=(self.window,1))
        # Encoder
        conv1_1 = Conv1D(16, 3, activation="relu", padding="same")(input)
        pool1 = MaxPooling1D(strides=3, padding="same")(conv1_1)
        conv1_2 = Conv1D(1, 3, activation="relu", padding="same")(pool1)
        encoded = MaxPooling1D(strides=3, padding="same")(conv1_2)
        self.encoder = Model(input, encoded)
        self.encoder.compile(optimizer='adam', loss='mae')
        # Decoder
        conv2_1 = Conv1D(1, 3, activation="relu", padding="same")(encoded)
        up1 = UpSampling1D(2)(conv2_1)
        conv2_2 = Conv1D(16, 2, activation='relu')(up1)
        up2 = UpSampling1D(2)(conv2_2)
        #Output
        output = Conv1D(1, 3, activation='sigmoid', padding='same')(up2)
        #Compile
        self.autoencoder = Model(input, output)
        self.autoencoder.compile(optimizer='adam', loss='mae')

    def fit_autoencoder(self):
        #Fit all datasets to model
        for i in range(len(self.X_train)):
            print("Fitting: ",i+1,"/",len(self.X_train))
            self.autoencoder.fit(self.X_train[i], self.y_train[i],epochs=self.num_epochs,batch_size=self.batch_size,validation_data=(self.X_test[i], self.y_test[i]))
    
    def fit_encoder(self):
        #Fit all datasets to model
        for i in range(len(self.X_train)):
            print("Fitting: ",i+1,"/",len(self.X_train))
            self.encoder.fit(self.X_train[i], self.y_train[i],epochs=self.num_epochs,batch_size=self.batch_size,validation_data=(self.X_test[i], self.y_test[i]))
    
    def predict(self,i):
        X_pred = self.autoencoder.predict(self.X_test[i])
        
    def encode_dataset(self,dataset):
        return dataset
    
    def save(self,path=None):
        if path is None:
            path=self.checkpoint
        self.model.save(path)
    
    def load(self,path=None):
        if path is None:
            path=self.checkpoint
        self.model = keras.models.load_model(path)