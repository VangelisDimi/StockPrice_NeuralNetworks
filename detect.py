import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d')
parser.add_argument('-n')
parser.add_argument('-mae')

args = parser.parse_args()
dataset = args.d
number_of_time_series_selected = args.n
error_value_as_double = args.mae
