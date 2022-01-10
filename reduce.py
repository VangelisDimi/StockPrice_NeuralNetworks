from scipy.sparse import data
from utils import ArgumentParser, create_dataset
from agent import CNN_autoencoder
import pandas as pd
import os

if __name__ == "__main__":
    if not os.path.exists('output'):
             os.makedirs('output')
    if not os.path.exists('models'):
            os.makedirs('models')

    parser = ArgumentParser()
    
    dataset_i=create_dataset(parser.dataset)
    dataset_q=create_dataset(parser.queryset)
    dataset_joined=pd.concat([dataset_i, dataset_q], ignore_index=True)
    
    agent = CNN_autoencoder(dataset=dataset_joined, latent_dim=parser.latent_dim, batch_size=parser.batch_size, num_epochs=parser.num_epochs, window=parser.window)
    if parser.train:
        agent.fit()
        agent.save('models/cnn_autoencoder_'+parser.model_name)
    else:
        agent.open('models/cnn_autoencoder_'+parser.model_name)

    agent.predict(1)