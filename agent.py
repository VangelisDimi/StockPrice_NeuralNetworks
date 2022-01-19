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
from keras.layers import LSTM,RepeatVector,TimeDistributed,Input,Conv1D,UpSampling1D,MaxPooling1D,BatchNormalization
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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

        self.scalers = [] 

        #Format data
        #Create list of training sets
        self.X_train=[]
        self.y_train=[]
        for i in range(len(self.dataset)):
            self.scalers.append(MinMaxScaler(feature_range=(0, 1)))
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, 1:self.train_size+1].values
            _dataset= np.array([_dataset]).T
            _dataset = self.scalers[i].fit_transform(_dataset)

            X_t = []
            y_t = []
            for j in range(self.window,self.train_size):
                X_t.append(_dataset[j-self.window:j, 0])
                y_t.append(_dataset[j, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            
            self.X_train.append(X_t)
            self.y_train.append(y_t)
        self.X_train_concat=self.X_train[0]
        self.y_train_concat=self.y_train[0]
        for i in range(1,len(self.X_train)):
            self.X_train_concat=np.append(self.X_train_concat,self.X_train[i],axis=0)
            self.y_train_concat=np.append(self.y_train_concat,self.y_train[i],axis=0)

        #Create list of testing sets
        self.X_test=[]
        self.y_test=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, self.train_size+1:].values
            _dataset = np.array([_dataset]).T
            _dataset = self.scalers[i].transform(_dataset)

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
        self.model.fit(self.X_train_concat, self.y_train_concat,epochs=self.num_epochs,batch_size=self.batch_size, validation_split=0.1)

    def predict(self,i):
        predicted_stock_price = self.model.predict(self.X_test[i])
        predicted_stock_price = self.scalers[i].inverse_transform(predicted_stock_price)

        predict_df = pd.DataFrame(index=self.dataset.columns[self.train_size+1+self.window:])
        predict_df['predicted_close'] = predicted_stock_price

        _dataset = self.dataset.iloc[i, self.train_size+1:].values
        _dataset = np.array([_dataset]).T
        predict_df['close'] = _dataset[self.window:]

        print("Mean absolute error of prediction : %.5f" % mean_absolute_error(predict_df['close'],predict_df['predicted_close']))
        return predict_df

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
        
        self.scalers = []

        #Format data
        #Create list of training sets
        self.X_train=[]
        self.y_train=[]
        for i in range(len(self.dataset)):
            self.scalers.append(StandardScaler())
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, 1:self.train_size+1].values
            _dataset= np.array([_dataset]).T
            _dataset = self.scalers[i].fit_transform(_dataset)
            
            X_t = []
            y_t = []
            for j in range(self.window,self.train_size):
                X_t.append(_dataset[j-self.window:j, 0])
                y_t.append(_dataset[j, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            
            self.X_train.append(X_t)
            self.y_train.append(y_t)
        self.X_train_concat=self.X_train[0]
        self.y_train_concat=self.y_train[0]
        for i in range(1,len(self.X_train)):
            self.X_train_concat=np.append(self.X_train_concat,self.X_train[i],axis=0)
            self.y_train_concat=np.append(self.y_train_concat,self.y_train[i],axis=0)

        #Create list of testing sets
        self.X_test=[]
        self.y_test=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, self.train_size+1:].values
            _dataset = np.array([_dataset]).T
            _dataset = self.scalers[i].transform(_dataset)

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
        self.model.compile(optimizer='adam',loss='mae')

    def fit(self):
        #Fit all datasets to model
        self.model.fit(self.X_train_concat, self.y_train_concat,epochs=self.num_epochs,batch_size=self.batch_size, validation_split=0.1)

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

        print("Mean absolute error of autoencoding: %.5f" % test_score_df['loss'].mean())

        return test_score_df

    def save(self,path=None):
        if path is None:
            path=self.checkpoint
        self.model.save(path)
    
    def load(self,path=None):
        if path is None:
            path=self.checkpoint
        self.model = keras.models.load_model(path)


class CNN_autoencoder():
    def __init__(self, dataset, batch_size=3, num_epochs=10, train_size=0.8):
        #Initialize network
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.window = 10
        self.checkpoint = "models/cnn_autoencoder.pkl"

        self.data_size=self.dataset.shape[1]-1
        self.train_size=math.floor(self.data_size*train_size)
        self.test_size=self.data_size-self.train_size

        self.scalers = []

        #Format data
        #Create list of training sets
        self.X_train=[]
        self.y_train=[]
        for i in range(len(self.dataset)):
            self.scalers.append(MinMaxScaler(feature_range=(0, 1)))
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, 1:self.train_size+1].values
            _dataset= np.array([_dataset]).T
            _dataset = self.scalers[i].fit_transform(_dataset)
            
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
        self.X_train_concat=self.X_train[0]
        self.y_train_concat=self.y_train[0]
        for i in range(1,len(self.X_train)):
            self.X_train_concat=np.append(self.X_train_concat,self.X_train[i],axis=0)
            self.y_train_concat=np.append(self.y_train_concat,self.y_train[i],axis=0)

        #Create list of testing sets
        self.X_test=[]
        self.y_test=[]
        for i in range(len(self.dataset)):
            # Normalize the dataset
            _dataset = self.dataset.iloc[i, self.train_size+1:].values
            _dataset = np.array([_dataset]).T
            _dataset = self.scalers[i].transform(_dataset)

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
        conv1_1 = BatchNormalization()(conv1_1)
        pool1 = MaxPooling1D(strides=2, padding="same")(conv1_1)
        conv1_2 = Conv1D(1, 3, activation="relu", padding="same")(pool1)
        conv1_2 = BatchNormalization()(conv1_2)
        encoded = MaxPooling1D(strides=2, padding="same",name='encoded')(conv1_2)
        self.encoder = Model(input, encoded)
        self.encoder.compile(optimizer='adam', loss='mae')
        # Decoder
        conv2_1 = Conv1D(1, 3, activation="relu", padding="same")(encoded)
        conv2_1 = BatchNormalization()(conv2_1)
        up1 = UpSampling1D(2)(conv2_1)
        conv2_2 = Conv1D(16, 2, activation='relu')(up1)
        conv2_2 = BatchNormalization()(conv2_2)
        up2 = UpSampling1D(2)(conv2_2)
        decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(up2)
        #Compile
        self.autoencoder = Model(input, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mae')

        self.latent_dim=self.encoder.get_layer('encoded').output_shape[1]

    def fit_autoencoder(self):
        #Fit all datasets to model
        self.autoencoder.fit(self.X_train_concat, self.y_train_concat,epochs=self.num_epochs,batch_size=self.batch_size, validation_split=0.1)
    
    def fit_encoder(self):
        #Fit all datasets to model
        self.encoder.fit(self.X_train_concat, self.y_train_concat,epochs=self.num_epochs,batch_size=self.batch_size, validation_split=0.1)
    
    def predict(self,i):
        X_pred = self.autoencoder.predict(self.X_test[i])
        print("Mean absolute error of autoencoding: %.5f" % 
                mean_absolute_error( self.scalers[i].inverse_transform(self.format_timeseries(X_pred)),
                                    self.scalers[i].inverse_transform(self.format_timeseries(self.X_test[i]))) )
        
    def encode_dataset(self,dataset):
        df=pd.DataFrame()

        for i in range(len(dataset)):
            scaler=MinMaxScaler(feature_range=(0, 1))
            _dataset = self.dataset.iloc[i,1:].values
            _dataset = np.array([_dataset]).T
            _dataset = scaler.fit_transform(_dataset)

            X_t = []
            for j in range(self.window,self.data_size,self.window):
                X_t.append(_dataset[j-self.window:j, 0])
            X_t = np.array(X_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))

            X_pred = self.encoder.predict(X_t)
            X_pred=scaler.inverse_transform(self.format_timeseries(X_pred))

            row=[]
            row.append(dataset['id'][i])
            for j in range(len(X_pred)):
                row.append(X_pred[j,0])
            df = df.append([row])
        return df
    
    def save(self,path=None):
        if path is None:
            path=self.checkpoint
        self.encoder.save(path+"/encode")
        self.autoencoder.save(path+"/autoencode")
    
    def load(self,path=None):
        if path is None:
            path=self.checkpoint
        self.encoder = keras.models.load_model(path+"/encode")
        self.autoencoder = keras.models.load_model(path+"/autoencode")

    def format_timeseries(self,timeseries):
        #Concat splitted timeseries
        X=[]
        for i in range(len(timeseries)):
            for j in range(len(timeseries[i])):
                X.append(timeseries[i,j])
        return X