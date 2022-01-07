from agent import LSTM_autoencoder
from utils import ArgumentParser, create_dataset
import os
import random
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    if not os.path.exists('output'):
             os.makedirs('output')

    parser = ArgumentParser()
    dataset=create_dataset(parser.dataset)
    agent = LSTM_autoencoder(dataset=dataset, mae=parser.error_value_as_double, batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
                num_layers=parser.num_layers, num_units=parser.num_units, dropout_rate=parser.dropout_rate, window=parser.window,
                train_size=parser.train_size)
    agent.fit()


    num_predictions=parser.number_of_time_series_selected
    for i in random.sample(range(len(dataset)),num_predictions):
        #With dataset training
        test_score_df,X_pred=agent.predict(i)

        #Plot
        anomalies = test_score_df[test_score_df.anomaly == True]

        plt.plot(test_score_df.index,agent.dataset_test[i][agent.window:], color = 'green', label = 'stock price')
        plt.plot(test_score_df.index,X_pred[:,0], color = 'blue', label = 'predicted stock price')
        plt.scatter(anomalies.index,anomalies['anomaly'], color = 'red', label = 'anomaly')

        plt.title('Anomaly detection: '+agent.dataset['id'][i])
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig('output/LSTM_autoencoder_%s.png' % dataset['id'][i])