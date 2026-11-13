import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
from data_entry import DataEntry
import argparse

# Constants
resultsFolderPath = "wyniki/"
imageFolderPath = "img/"

fullData = {}
names = []

# Values picked by visually deciding the best number of clusters for a given shape
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]


def compare_time_complexity(repeats, save_results, save_plot, use_sklearn_lib, name="corners"):
    measured_data = {}
    if use_sklearn_lib:
        suffix = "_sklearn"
        plot_title = "Grupowanie hierarchiczne danych niezgranulowanych algorytmem bibliotecznym"
        data_size_presets_for_time_measurement = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
    else:
        suffix = ""
        plot_title = "Grupowanie hierarchiczne danych niezgranulowanych"
        data_size_presets_for_time_measurement = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for i in data_size_presets_for_time_measurement:
        if use_sklearn_lib:
            measured_data[name + str(i)] = fullData[name + str(i)]
        else:
            if "spheres" in name:
                measured_data[name + str(i)] = DataEntry(fullData[name + "5000"].data[:i], name + str(i), 3, has_labels=False)
            else:
                measured_data[name + str(i)] = DataEntry(fullData[name + "5000"].data[:i], name + str(i), 2, has_labels=False)
        print("Measure ", name, str(i))
        measured_data[name + str(i)].measure_times(repeats, use_sklearn_lib)

    s = len(data_size_presets_for_time_measurement)
    regular_avg = np.empty(shape=s)
    granule50_avg = np.empty(shape=s)
    granule100_avg = np.empty(shape=s)
    granule200_avg = np.empty(shape=s)

    for i in range(s):
        regular_avg[i] = np.mean(measured_data[name + str(data_size_presets_for_time_measurement[i])].strict_number_times)
        granule50_avg[i] = np.mean(
            measured_data[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[50])
        granule100_avg[i] = np.mean(
            measured_data[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[100])
        granule200_avg[i] = np.mean(
            measured_data[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[200])

    results = [regular_avg, granule50_avg, granule100_avg, granule200_avg]
    results = [list(i) for i in zip(*results)]
    results = pandas.DataFrame(results, columns=["strict", "50 granules", "100 granules", "200 granules"])
    if save_results:
        if not os.path.exists(resultsFolderPath):
            os.makedirs(resultsFolderPath)
        results.to_csv(resultsFolderPath + "times" + suffix + ".csv")
    else:
        print(results.to_string())

    plt.figure()
    plt.plot(data_size_presets_for_time_measurement, regular_avg)
    plt.plot(data_size_presets_for_time_measurement, granule50_avg)
    plt.plot(data_size_presets_for_time_measurement, granule100_avg)
    plt.plot(data_size_presets_for_time_measurement, granule200_avg)
    plt.xlabel("Liczba danych")
    plt.ylabel("Czas wykonania [ms]")
    plt.title("Średni czas wykonania algorytmów w zależności od liczby danych")
    legend = plt.legend([plot_title, "Grupowanie hierarchiczne 50 granul",
                         "Grupowanie hierarchiczne 100 granul", "Grupowanie hierarchiczne 200 granul"],
                        bbox_to_anchor=(0.5, -0.38), loc='lower center')
    if save_plot:
        if not os.path.exists(imageFolderPath):
            os.makedirs(imageFolderPath)
        plt.savefig(imageFolderPath + "comparing_times" + suffix + ".pdf", bbox_extra_artists=[legend], bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(prog="Measure Time",
                                     description="Measure time of fuzzy hierarchical clusterization in comparison with strict clusterization")
    parser.add_argument("-r", "--repeats", type=int, default=4, help="measurement repeats for averaging, 4 by default")
    parser.add_argument("-l", "--library_function", action="store_true", help="use scikit-learn's AgglomerativeClustering as the strict implementation instead of simple O(n^3) algorithm, measure larger data")
    parser.add_argument("-t", "--save_times", action="store_true", help="save times to .csv instead of printing")
    parser.add_argument("-p", "--save_plot", action="store_true", help="save plot to .pdf instead of showing")
    parser.add_argument("-f", "--filename", choices=names_roots, default="corners", help="specific file to measure, 'corners' by default")
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
                        fullData[file.name[:-5]].name_root = key
                        fullData[file.name[:-5]].clusters_number = value
                        break
    compare_time_complexity(args.repeats, args.save_times, args.save_plot, args.library_function, args.filename)


if __name__ == "__main__":
    main()
