import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d')
parser.add_argument('-q')
parser.add_argument('-od')
parser.add_argument('-oq')

args = parser.parse_args()
dataset = args.d
queryset = args.q
output_dataset_file = args.od
output_query_file = args.oq