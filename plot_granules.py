import os
from data_entry import DataEntry
import argparse

# Constants
folderPath = "img/"

fullData = {}
names = []

num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}


def plot_results(specific_name=None, save_plot=True, linkage="single"):
    if specific_name is None:
        for name in names:
            fullData[name].fit_plot_fuzzy_labels('t', linkage=linkage)
            fullData[name].fit_plot_fuzzy_labels('e', linkage=linkage)
            fullData[name].fit_plot_fuzzy_labels('g', linkage=linkage)
    else:
        data_to_plot = fullData.get(specific_name)
        if data_to_plot is not None:
            data_to_plot.fit_plot_fuzzy_labels('t', linkage=linkage, save_to_file=save_plot)
            data_to_plot.fit_plot_fuzzy_labels('e', linkage=linkage, save_to_file=save_plot)
            data_to_plot.fit_plot_fuzzy_labels('g', linkage=linkage, save_to_file=save_plot)


def main():
    parser = argparse.ArgumentParser(prog="Plot Granule Clusterization",
                                     description="Plot results of fuzzy hierarchical clusterization for 3 relation types, granule numbers and xi values")
    parser.add_argument("-f", "--filename", help="specific file to plot results of, all files if not given")
    parser.add_argument("-l", "--linkage", choices=["single", "complete"], default="single",
                        help="linkage of the hierachical clustering: single or complete, single by default")
    parser.add_argument("-p", "--save_plot", action="store_true", help="save plot to .pdf instead of showing (always saves for all files plotting)")
    args = parser.parse_args()
    for folder in os.scandir("dane_labelled"):
        for file in os.scandir(folder.path):
            with open(file.path) as f:
                # Assumption: all folders contain only files with .data extension
                names.append(file.name[:-5])
                if "spheres" in file.name:
                    fullData[file.name[:-5]] = DataEntry(f.read(), file.name[:-5], 3)
                else:
                    fullData[file.name[:-5]] = DataEntry(f.read(), file.name[:-5])
                for key, value in num_of_clusters.items():
                    if key in file.name:
                        fullData[file.name[:-5]].clusters_number = value
                        break
    plot_results(args.filename, args.save_plot, args.linkage)


if __name__ == "__main__":
    main()
