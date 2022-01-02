#Supress cuda warnings on non-nvidia computers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

numpy.random.seed(21)

class multiLayer_LSTM():
    def __init__(self, dataset, batch_size, num_epochs, num_layers, layers_size, dropout_rate=0.2, look_back=1):

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.layers_size = layers_size
        self.look_back = look_back

        self._dataset = self.dataset.iloc[:, 1:self.dataset.shape[1]].values

        # normalize the dataset
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._dataset = self.scaler.fit_transform(self._dataset)

        y=numpy.array(list(range(len(self._dataset))))

        self.train_X, self.test_Y, self.train_Y, self.test_Y = train_test_split(self._dataset,y,test_size=0.75,train_size=0.25,shuffle=True)

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


class LSTM_encoder_decoder():
    def __init__(self, dataset, batch_size, num_epochs, num_layers, layers_size, dropout_rate=0.2, look_back=1):
        
        self.dataset = dataset

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


class CNN_autoencoder():
    def __init__(self, dataset, batch_size, num_epochs, num_layers, layers_size, dropout_rate=0.2, look_back=1):

        self.dataset = dataset

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