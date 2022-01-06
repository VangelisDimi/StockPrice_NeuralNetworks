#Supress cuda warnings on non-nvidia computers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random

np.random.seed(21)

class multiLayer_LSTM():
    def __init__(self, dataset, batch_size=3, num_epochs=10, num_layers=3, num_units=50, dropout_rate=0.2, look_back=60, train_size = 0.8):
        #Initialize network
        self.dataset = dataset
        self.num_units = num_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.look_back = look_back
        self.dropout_rate = dropout_rate
        
        self.data_size=self.dataset.shape[1]-1
        self.train_size=math.floor(self.data_size*train_size)
        self.test_size=self.data_size-self.train_size-1

        self.scaler = MinMaxScaler(feature_range=(0, 1))

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

            # Creating a data structure with self.look_back time-steps and 1 output
            X_t = []
            y_t = []
            for i in range(self.look_back,self.train_size):
                X_t.append(_dataset[i-self.look_back:i, 0])
                y_t.append(_dataset[i, 0])
            X_t, y_t = np.array(X_t), np.array(y_t)
            X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], 1))
            
            self.X_train.append(X_t)
            self.y_train.append(y_t)
        
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

        #Compiling the RNN
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    def fit(self):
        #Fit all datasets to model
        for i in range(len(self.X_train)):
            print("Fitting: ",i+1,"/",len(self.X_train))
            self.model.fit(self.X_train[i], self.y_train[i], epochs=self.num_epochs, batch_size=self.batch_size)

    def predict(self,i):
        #Predict results for stock
        dataset_total = self.dataset.iloc[i,1:].values
        dataset_total= np.array([dataset_total]).T
        inputs = dataset_total[self.train_size - self.test_size - self.look_back:]

        inputs = inputs.reshape(-1,1)
        inputs = self.scaler.transform(inputs)

        X_test = []
        for y in range(self.look_back, self.test_size + self.look_back+1):
            X_test.append(inputs[y-self.look_back:y, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = self.model.predict(X_test)
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)
        return predicted_stock_price




class LSTM_encoder_decoder():
    def __init__(self, dataset, batch_size, num_epochs, num_layers, num_units, layers_size, dropout_rate=0.2, look_back=1):
        
        self.dataset = dataset
        self.num_units = num_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.layers_size = layers_size
        self.look_back = look_back
        self.dropout_rate = dropout_rate

        # normalize the dataset
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.scaler.fit_transform(self.dataset)

        self.train_X, self.train_Y, self.test_X, self.test_Y = train_test_split(self.dataset)

        self.model = Sequential()
        for i in range(self.num_layers):
            self.model.add(LSTM(units = self.num_units, return_sequences = True))
            self.model.add(Dropout(self.dropout_rate))
        #Output Layer
        self.model.add(Dense(units = 1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self):
        self.model.fit(self.train_X, self.train_Y, epochs=self.num_epochs, batch_size=self.batch_size, verbose=2)

    def score(self):
        # make predictions
        self.pred_train_X = self.model.predict(self.train_X)
        self.pred_test_X = self.model.predict(self.test_X)

        # invert predictions
        self.pred_train_X = self.scaler.inverse_transform(self.pred_train_X)
        train_Y = self.scaler.inverse_transform([self.train_Y])
        self.pred_test_X = self.scaler.inverse_transform(self.pred_test_X)
        test_Y = self.scaler.inverse_transform([self.test_Y])

        # calculate root mean squared error
        train_score = math.sqrt(mean_squared_error(train_Y[0], self.pred_train_X[:,0]))
        print('Train Score: %.2f RMSE' % (train_score))
        test_score = math.sqrt(mean_squared_error(test_Y[0], self.pred_test_X[:,0]))
        print('Test Score: %.2f RMSE' % (test_score))

    def plot(self):
        # shift train predictions for plotting
        train_plot = numpy.empty_like(self.dataset)
        train_plot[:, :] = numpy.nan
        train_plot[self.look_back:len(self.pred_train_X)+self.look_back, :] = self.pred_train_X

        # shift test predictions for plotting
        test_plot = numpy.empty_like(self.dataset)
        test_plot[:, :] = numpy.nan
        test_plot[len(self.pred_train_X)+(self.look_back*2)+1:len(self.dataset)-1, :] = self.pred_test_X

        # plot baseline and predictions
        plt.plot(self.scaler.inverse_transform(self.dataset))
        plt.plot(train_plot)
        plt.plot(test_plot)
        plt.show()


class CNN_autoencoder():
    def __init__(self, dataset, batch_size, num_epochs, num_layers, num_units, layers_size, dropout_rate=0.2, look_back=1):

        self.dataset = dataset
        self.num_units = num_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.layers_size = layers_size
        self.look_back = look_back

        # normalize the dataset
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.scaler.fit_transform(self.dataset)

        self.train_X, self.train_Y, self.test_X, self.test_Y = train_test_split(self.dataset)

        # create and fit the LSTM network
        self.model = Sequential()
        self.model.add(LSTM(num_layers, input_shape=layers_size))
        self.model.add(Dense(1))
        self.model.add(Dropout(rate=dropout_rate, input_shape=layers_size))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self):
        self.model.fit(self.train_X, self.train_Y, epochs=self.num_epochs, batch_size=self.batch_size, verbose=2)

    def score(self):
        # make predictions
        self.pred_train_X = self.model.predict(self.train_X)
        self.pred_test_X = self.model.predict(self.test_X)

        # invert predictions
        self.pred_train_X = self.scaler.inverse_transform(self.pred_train_X)
        train_Y = self.scaler.inverse_transform([self.train_Y])
        self.pred_test_X = self.scaler.inverse_transform(self.pred_test_X)
        test_Y = self.scaler.inverse_transform([self.test_Y])

        # calculate root mean squared error
        train_score = math.sqrt(mean_squared_error(train_Y[0], self.pred_train_X[:,0]))
        print('Train Score: %.2f RMSE' % (train_score))
        test_score = math.sqrt(mean_squared_error(test_Y[0], self.pred_test_X[:,0]))
        print('Test Score: %.2f RMSE' % (test_score))

    def plot(self):
        # shift train predictions for plotting
        train_plot = numpy.empty_like(self.dataset)
        train_plot[:, :] = numpy.nan
        train_plot[self.look_back:len(self.pred_train_X)+self.look_back, :] = self.pred_train_X

        # shift test predictions for plotting
        test_plot = numpy.empty_like(self.dataset)
        test_plot[:, :] = numpy.nan
        test_plot[len(self.pred_train_X)+(self.look_back*2)+1:len(self.dataset)-1, :] = self.pred_test_X

        # plot baseline and predictions
        plt.plot(self.scaler.inverse_transform(self.dataset))
        plt.plot(train_plot)
        plt.plot(test_plot)
        plt.show()