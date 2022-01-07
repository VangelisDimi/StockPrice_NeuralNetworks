from utils import ArgumentParser, create_dataset
from agent import CNN_autoencoder
import os

if __name__ == "__main__":
    if not os.path.exists('output'):
             os.makedirs('output')

    parser = ArgumentParser()
    
    dataset_i=create_dataset(parser.dataset)
    dataset_q=create_dataset(parser.queryset)

    agent = CNN_autoencoder(dataset=dataset_i,batch_size=parser.batch_size, num_epochs=parser.num_epochs, window=parser.window)
    agent.fit()