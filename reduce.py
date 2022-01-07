from utils import ArgumentParser, create_dataset
from agent import CNN_autoencoder
import os

if __name__ == "__main__":
    if not os.path.exists('output'):
             os.makedirs('output')

    parser = ArgumentParser()
    
    dataset=create_dataset(parser.dataset)
    agent = CNN_autoencoder(dataset=create_dataset(parser.dataset), batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
                num_layers=parser.num_layers, num_units=parser.num_units, layers_size=parser.layers_size, look_back=parser.look_back)
    agent.fit()