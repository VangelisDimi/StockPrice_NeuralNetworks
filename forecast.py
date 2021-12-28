import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d')
parser.add_argument('-n')
args = parser.parse_args()
dataset = args.d
number_of_time_series_selected = args.n
print("-d =",dataset)
print("-n =",number_of_time_series_selected)
