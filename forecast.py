from agent import multiLayer_LSTM
from utils import ArgumentParser

parser = ArgumentParser()
agent = multiLayer_LSTM(dataset=parser.dataset, batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
            num_layers=parser.num_layers, layers_size=parser.layers_size, look_back=parser.look_back)
agent.fit()
agent.score()
agent.plot()