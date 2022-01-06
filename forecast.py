from agent import multiLayer_LSTM
from utils import ArgumentParser, create_dataset
import random
import os

if __name__ == "__main__":
    if not os.path.exists('output'):
             os.makedirs('output')

    parser = ArgumentParser()

    dataset=create_dataset(parser.dataset)
    agent = multiLayer_LSTM(dataset=dataset, batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
                num_layers=parser.num_layers, num_units=parser.num_units, layers_size=parser.layers_size, look_back=parser.look_back)
    agent.fit()

    num_predictions=1
    for i in random.sample(range(len(dataset)),num_predictions):
        predicted_stock_price=agent.predict(i)
        agent.plot(i,predicted_stock_price,'output/LSTM_%s.png' % dataset['id'][i])