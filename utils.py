import argparse
import pandas

class ArgumentParser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', type=str, default="Examples/nasd_input.csv")
        parser.add_argument('-n', type=int, default=10)
        parser.add_argument('-q', type=str, default="Examples/nasd_query.csv")
        parser.add_argument('-od', type=str, default="output_dataset.csv")
        parser.add_argument('-oq', type=str, default="output_query_file.csv")
        parser.add_argument('-mae', type=float, default=0.65)
        parser.add_argument('-batch_size', type=int, default=30)
        parser.add_argument('-num_layers', type=int, default=1)
        parser.add_argument('-num_units', type=int, default=50)
        parser.add_argument('-num_epochs', type=int, default=10)
        parser.add_argument('-window', type=int, default=10)
        parser.add_argument('-dropout_rate', type=float, default=0.2)
        parser.add_argument('-train_size', type=float, default=0.8)
        parser.add_argument('-latent_dim', type=int, default=3)
        parser.add_argument('--train', action='store_true')
        parser.add_argument('-model_name', type=str, default='model')

        args = parser.parse_args()
        self.dataset = args.d
        self.queryset = args.q
        self.output_dataset_file = args.od
        self.output_query_file = args.oq
        self.error_value_as_double = args.mae
        self.number_of_time_series_selected = args.n
        self.window=args.window
        self.num_layers = args.num_layers
        self.num_units = args.num_units
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.dropout_rate = args.dropout_rate
        self.train_size = args.train_size
        self.latent_dim = args.latent_dim
        self.train = args.train
        self.model_name = args.model_name

def create_dataset(dataset):
    #Columns: stock_id,day1,......,dayN
    df = pandas.read_csv(dataset,'\t', header=None)
    column_names=['id']
    for i in range(1,df.shape[1]): column_names.append(i)
    df=df.set_axis(column_names, axis='columns', inplace=False)
    return df