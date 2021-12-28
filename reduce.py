import argparse
from utils import ArgumentParser
from agent import CNN_autoencoder

parser = ArgumentParser()
agent = CNN_autoencoder(dataset=parser.dataset, batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
            num_layers=parser.num_layers, layers_size=parser.layers_size, look_back=parser.look_back)
agent.fit()
agent.score()
agent.plot()
