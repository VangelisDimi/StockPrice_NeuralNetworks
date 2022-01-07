from scipy.sparse import data
from agent import LSTM_autoencoder
from utils import ArgumentParser, create_dataset
import os
import random
import matplotlib.pyplot as plt

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
        test_score_df=agent.predict(i)

        #Plot
        plt.plot(test_score_df.index,test_score_df['loss'], color = 'blue', label = 'loss')
        plt.plot(test_score_df.index,test_score_df['threshold'], color = 'orange', label = 'threshold')
        plt.title('Anomaly detection: '+agent.dataset['id'][i])
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('output/LSTM_autoencoder_%s.png' % dataset['id'][i])