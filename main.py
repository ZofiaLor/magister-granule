import matplotlib.pyplot as plt
import sklearn
import numpy as np
import math
from hierarchy import HierarchicalClustering, Granule, calculate_fcm_variance
from fuzzy_cmeans import FuzzyCMeans
import os
import time
import skfuzzy as fuzz
import pandas
from scipy.cluster.hierarchy import dendrogram
from data_entry import DataEntry

# Constants
seed = 42
n_granules = 50
n_iterations = 80
repeats = 4
s = 10
folderPath = "img/iter80/"

fullData = {}
names = []

# Values picked by visually deciding the best number of clusters for a given shape
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
# blobs10000 -> 7 minutes of clustering
data_size_presets = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
data_size_presets_for_time_measurement = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]


def measure_accuracy_recall_precision(linkage, specific_file=None, print_to_console=False):
    linkages = {"single": "sl", "complete": "cl"}
    if linkages.get(linkage) is None:
        linkage = "single"
    if specific_file is not None:
        data_to_measure = fullData.get(specific_file)
        if data_to_measure is not None:
            print(data_to_measure.measure_accuracy(linkage))
    else:
        for root in names_roots:
            accuracy_results = pandas.DataFrame()
            for data_size in data_size_presets:
                print(root + str(data_size))
                result = fullData[root + str(data_size)].measure_accuracy(linkage)
                accuracy_results = pandas.concat([accuracy_results, result], ignore_index=True)
            if print_to_console:
                print(root)
                print(accuracy_results)
            else:
                accuracy_results.to_csv("wyniki/" + root + "_" + linkages[linkage] + "_accuracy.csv")


def compare_time_complexity(name="circles"):
    data10000 = {}
    for i in range(500, 5500, 500):
        if "spheres" in name:
            data10000[name + str(i)] = DataEntry(fullData[name + "5000"].data[:i], name + str(i), 3)
        else:
            data10000[name + str(i)] = DataEntry(fullData[name + "5000"].data[:i], name + str(i), 2)
        print("Measure ", name, str(i))
        data10000[name + str(i)].measure_times()

    regular_avg = np.empty(shape=s)
    granule50_avg = np.empty(shape=s)
    granule100_avg = np.empty(shape=s)
    granule200_avg = np.empty(shape=s)

    for i in range(len(data_size_presets_for_time_measurement)):
        regular_avg[i] = np.mean(data10000[name + str(data_size_presets_for_time_measurement[i])].strict_number_times)
        granule50_avg[i] = np.mean(data10000[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[50])
        granule100_avg[i] = np.mean(data10000[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[100])
        granule200_avg[i] = np.mean(data10000[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[200])

    plt.figure()
    plt.plot(data_size_presets, regular_avg)
    plt.plot(data_size_presets, granule50_avg)
    plt.plot(data_size_presets, granule100_avg)
    plt.plot(data_size_presets, granule200_avg)
    plt.xlabel("Liczba danych")
    plt.ylabel("Czas wykonania [ms]")
    plt.title("Średni czas wykonania algorytmów w zależności od liczby danych")
    legend = plt.legend(["Grupowanie hierarchiczne danych niezgranulowanych", "Grupowanie hierarchiczne 50 granul",
                         "Grupowanie hierarchiczne 100 granul", "Grupowanie hierarchiczne 200 granul"],
                        bbox_to_anchor=(0.5, -0.38), loc='lower center')
    plt.savefig(folderPath + "comparing_times.pdf", bbox_extra_artists=[legend], bbox_inches='tight')


def plot_results(specific_name=None, save_plot=True):
    if specific_name is None:
        for name in names:
            fullData[name].fit_plot_fuzzy_labels('t')
            fullData[name].fit_plot_fuzzy_labels('e')
            fullData[name].fit_plot_fuzzy_labels('g')
    else:
        data_to_plot = fullData.get(specific_name)
        if data_to_plot is not None:
            data_to_plot.fit_plot_fuzzy_labels('t', save_to_file=save_plot)
            data_to_plot.fit_plot_fuzzy_labels('e', save_to_file=save_plot)
            data_to_plot.fit_plot_fuzzy_labels('g', save_to_file=save_plot)


for folder in os.scandir("dane"):
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

user_input = ""
while True:
    user_input = ""
    while user_input not in ["1", "2", "3", "4"]:
        user_input = input("Select an action:\n1. Measure accuracy, recall and precision\n2. Compare time of fuzzy "
                           "and non-fuzzy algorithms\n3. Plot\n4. Exit\n")
    if user_input == "1":
        user_input = ""
        while user_input not in ["1", "2", "3", "4"]:
            user_input = input("Select an action:\n1. Measure all files, save to csv\n2. Measure all files, "
                               "print result\n3. Measure specific file, print result\n4. Exit\n")
        if user_input == "1":
            user_input = ""
            while user_input not in ["1", "2"]:
                user_input = input("Select linkage:\n1. Single\n2. Complete\n")
            if user_input == "1":
                measure_accuracy_recall_precision("single")
            elif user_input == "2":
                measure_accuracy_recall_precision("complete")
        elif user_input == "2":
            user_input = ""
            while user_input not in ["1", "2"]:
                user_input = input("Select linkage:\n1. Single\n2. Complete\n")
            if user_input == "1":
                measure_accuracy_recall_precision("single", print_to_console=True)
            elif user_input == "2":
                measure_accuracy_recall_precision("complete", print_to_console=True)
        elif user_input == "3":
            user_input = ""
            filename = ""
            while user_input not in ["1", "2"]:
                user_input = input("Select linkage:\n1. Single\n2. Complete\n")
            filename = input("Input file name\n")
            if user_input == "1":
                measure_accuracy_recall_precision("single", specific_file=filename, print_to_console=True)
            elif user_input == "2":
                measure_accuracy_recall_precision("complete", specific_file=filename, print_to_console=True)
    elif user_input == "2":
        user_input = input("Enter data shape name (blobs, circles, corners, crescents, laguna, spheres)\n")
        if user_input in names_roots:
            compare_time_complexity(user_input)
        else:
            compare_time_complexity()
    elif user_input == "3":
        user_input = ""
        while user_input not in ["1", "2", "3", "4"]:
            user_input = input("Select an action:\n1. Plot all data, save to file\n2. Plot specific data, save to "
                               "file\n3. Plot specific data, show plots\n4. Exit\n")
        if user_input == "1":
            plot_results()
        elif user_input == "2":
            user_input = input("Input file name\n")
            plot_results(user_input, True)
        elif user_input == "3":
            user_input = input("Input file name\n")
            plot_results(user_input, False)
    elif user_input == "4":
        break


# # Generate data
# data, labels = sklearn.datasets.make_blobs(n_samples=1000, centers=4, cluster_std=[0.5, 2, 1.5, 1], random_state=seed)
# # data, labels = sklearn.datasets.make_blobs(n_samples=1000, centers=4, random_state=100)
#
# # A small data set for manual verification from https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
# # data_small = np.array([[4, 21], [5, 19], [10, 24], [4, 17], [3, 16], [11, 25], [14, 24], [6, 22], [10, 21], [12, 21]])
# # data_small, labels = sklearn.datasets.make_blobs(n_samples=50, centers=4)
