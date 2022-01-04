from utils import ArgumentParser, create_dataset
from agent import CNN_autoencoder


if __name__ == "__main__":
    parser = ArgumentParser()
    agent = CNN_autoencoder(dataset=create_dataset(parser.dataset), batch_size=parser.batch_size, num_epochs=parser.num_epochs, 
                num_layers=parser.num_layers, layers_size=parser.layers_size, look_back=parser.look_back)
    agent.fit()
    agent.score()
    agent.plot()
