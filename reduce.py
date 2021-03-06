from utils import ArgumentParser, create_dataset
from agent import CNN_autoencoder
import pandas as pd
import os
import random

if __name__ == "__main__":
    if not os.path.exists('output'):
             os.makedirs('output')
    if not os.path.exists('models'):
            os.makedirs('models')

    parser = ArgumentParser()
    
    dataset_i=create_dataset(parser.dataset)
    dataset_q=create_dataset(parser.queryset)
    dataset_joined=pd.concat([dataset_i, dataset_q], ignore_index=True)
    
    agent = CNN_autoencoder(dataset=dataset_joined, batch_size=parser.batch_size, num_epochs=parser.num_epochs)
    if parser.train:
        agent.fit_encoder()
        agent.fit_autoencoder()
        agent.save('models/cnn_autoencoder_'+parser.model_name)
    else:
        agent.load('models/cnn_autoencoder_'+parser.model_name)

    #Testing
    num_predictions=parser.number_of_time_series_selected
    for i in random.sample(range(len(dataset_joined)),num_predictions):
        agent.predict(i)

    #Encode given datasets
    dataset_reduced= agent.encode_dataset()
    dataset_reduced.iloc[:len(dataset_i.index),:].to_csv('output/'+parser.output_dataset_file,'\t', index=False, header=False, float_format='%.4f')
    dataset_reduced.iloc[len(dataset_i.index):,:].to_csv('output/'+parser.output_query_file,'\t', index=False, header=False, float_format='%.4f')
