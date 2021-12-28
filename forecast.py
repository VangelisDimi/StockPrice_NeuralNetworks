import argparse
import numpy
import matplotlib.pyplot as plt
import pandas
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
numpy.random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument('-d')
parser.add_argument('-n')
parser.add_argument('-batch_size',type=int, default=1)
parser.add_argument('-num_layers',type=int, default=4)
parser.add_argument('-layers_size',type=tuple, default=(1,1))
parser.add_argument('-num_epochs',type=int, default=100)

args = parser.parse_args()
dataset = args.d
number_of_time_series_selected = args.n
print("-d =",dataset)
print("-n =",number_of_time_series_selected)

look_back=1
num_layers = args.num_layers
batch_size = args.batch_size
layers_size = args.layers_size
num_epochs = args.num_epochs

dataframe = pandas.read_csv(dataset)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_X, train_Y, test_X, test_Y = train_test_split(dataset)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(num_layers, input_shape=layers_size))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_Y, epochs=100, batch_size=batch_size, verbose=2)

# make predictions
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
train_Y = scaler.inverse_transform([train_Y])
testPredict = scaler.inverse_transform(testPredict)
test_Y = scaler.inverse_transform([test_Y])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_Y[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_Y[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
