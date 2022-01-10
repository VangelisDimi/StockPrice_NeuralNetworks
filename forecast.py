from agent import multiLayer_LSTM
from utils import ArgumentParser, create_dataset
import random
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if not os.path.exists('output'):
             os.makedirs('output')
    if not os.path.exists('models'):
            os.makedirs('models')

    parser = ArgumentParser()
    
    dataset=create_dataset(parser.dataset)
    agent = multiLayer_LSTM(dataset=dataset, batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
                num_layers=parser.num_layers, num_units=parser.num_units, dropout_rate=parser.dropout_rate, window=parser.window,
                train_size=parser.train_size)
    if parser.train:
        agent.fit()
        agent.save('models/lstm_multilayer_'+parser.model_name)
    else:
        agent.load('models/lstm_multilayer_'+parser.model_name)

    num_predictions=parser.number_of_time_series_selected
    for i in random.sample(range(len(dataset)),num_predictions):
        #With dataset training
        predicted_stock_price=agent.predict(i)

        #With single time-series training
        s_dataset=dataset.iloc[[i]]
        s_dataset.index=[0]
        agent_single = multiLayer_LSTM(dataset=s_dataset, batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
                    num_layers=parser.num_layers, num_units=parser.num_units, window=parser.window)
        agent_single.fit()
        predicted_stock_price_s=agent_single.predict(0)

        #Plot
        plt.plot(predicted_stock_price.index,predicted_stock_price['close'], color = 'red', label = 'Real Stock Price')
        plt.plot(predicted_stock_price.index,predicted_stock_price['predicted_close'], color = 'blue', label = 'Predicted Stock Price')
        plt.plot(predicted_stock_price_s.index,predicted_stock_price_s['predicted_close'], color = 'green', label = 'Predicted Stock Price (Single)')
        plt.title('Stock Price Prediction: '+agent.dataset['id'][i])
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig('output/LSTM_multilayer_%s.png' % dataset['id'][i])
        plt.clf()