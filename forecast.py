from agent import multiLayer_LSTM
from utils import ArgumentParser, create_dataset

if __name__ == "__main__":
    parser = ArgumentParser()

    agent = multiLayer_LSTM(dataset=create_dataset(parser.dataset), batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
                num_layers=parser.num_layers, num_units=parser.num_units, layers_size=parser.layers_size, look_back=parser.look_back)
    agent.fit()
    agent.test_predict()