import sys
sys.path.append("./")
import argparse
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

def print_lasotext_results(res_path):
    trackers = []
    dataset_name = 'lasot_extension_subset'
    method_name = res_path.split('/')[-1]
    trackers.extend(trackerlist(name=method_name, parameter_name="LaSOT_ext", dataset_name=dataset_name, run_ids="", display_name=method_name))
    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

def print_otb100_results(res_path):
    trackers = []
    dataset_name = 'otb'
    method_name = res_path.split('/')[-1]
    trackers.extend(trackerlist(name=method_name, parameter_name="OTB100", dataset_name=dataset_name, run_ids="", display_name=method_name))
    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", default="output/samosa/", help="path to test video path")
    args = parser.parse_args()

    print_lasotext_results(args.res_path)
    print_otb100_results(args.res_path)