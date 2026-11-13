# data sources https://snap.stanford.edu/data/loc-gowalla.html https://snap.stanford.edu/data/loc-brightkite.html

from data_entry import DataEntry
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas
import argparse
import os


np.random.seed(42)

def measure_time(data, clusters, step):
    length = data.length
    r = []
    measured_data = dict()
    if (step is None) or (step > length):
        if length >= 1000000:
            r = [50000, 100000, 500000] + list(range(1000000, length, 1000000))
        else:
            r = list(range(length // 10, length, length // 10))
    else:
        r = list(range(step, length, step))
    for size in r:
        data_shortened = DataEntry(data.data[np.random.choice(length, size, replace=False), :],
                             "data_shortened", 2, has_labels=False)
        data_shortened.clusters_number = clusters
        print(size)
        data_shortened.measure_fuzzy_number_times(1)
        measured_data[size] = dict()
        for n in data_shortened.granules_number:
            print(np.mean(data_shortened.fuzzy_number_times[n]))
            measured_data[size][n] = np.mean(data_shortened.fuzzy_number_times[n])

    s = len(r)
    granule50_avg = np.empty(shape=s)
    granule100_avg = np.empty(shape=s)
    granule200_avg = np.empty(shape=s)

    for i in range(s):
        granule50_avg[i] = np.mean(measured_data[r[i]][50]) // 1000
        granule100_avg[i] = np.mean(measured_data[r[i]][100]) // 1000
        granule200_avg[i] = np.mean(measured_data[r[i]][200]) // 1000

    results = [granule50_avg, granule100_avg, granule200_avg]
    results = [list(i) for i in zip(*results)]
    results = pandas.DataFrame(results, columns=["50 granules", "100 granules", "200 granules"])
    print(results.to_string())
    folderPath = "wyniki/"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    if not os.path.isfile(folderPath + "large_data_times.csv"):
        results.to_csv(folderPath + "large_data_times.csv")
    else:
        counter = 1
        while os.path.isfile(folderPath + "large_data_times" + str(counter) + ".csv"):
            counter += 1
        results.to_csv(folderPath + "large_data_times" + str(counter) + ".csv")

    plt.figure()
    plt.plot(r, granule50_avg)
    plt.plot(r, granule100_avg)
    plt.plot(r, granule200_avg)
    plt.xlabel("Liczba danych")
    plt.ylabel("Czas wykonania [s]")
    plt.title("Średni czas wykonania algorytmów w zależności od liczby danych")
    legend = plt.legend(["Czas wykonania rozmytego algorytmu hierarchicznego na dużych liczebnościach danych",
                         "Grupowanie hierarchiczne 50 granul",
                         "Grupowanie hierarchiczne 100 granul", "Grupowanie hierarchiczne 200 granul"],
                        bbox_to_anchor=(0.5, -0.38), loc='lower center')
    plt.tight_layout()
    plt.show()


def plot_clustering(data, clusters, size, flip_coords=True):
    if (size is None) or (size > data.length):
        data_shortened = data
    else:
        data_shortened = DataEntry(data.data[np.random.choice(data.length, 500000, replace=False), :],
                         "data_shortened", 2, has_labels=False)
    ac = sklearn.cluster.AgglomerativeClustering(n_clusters=clusters, linkage='single')
    ac.fit(data_shortened.data)
    plt.figure()
    if flip_coords:
        plt.scatter(data_shortened.data[:, 1], data_shortened.data[:, 0], c=ac.labels_)
    else:
        plt.scatter(data_shortened.data[:, 0], data_shortened.data[:, 1], c=ac.labels_)
    plt.show()
    hc, fcm, granules = data.fuzzy_prepare_fit(100)

    plt.figure(figsize=(12, 10))
    if flip_coords:
        plt.scatter(data.data[:, 1], data.data[:, 0], c='lightgray')
        plt.scatter(fcm.cluster_centers_[:, 1], fcm.cluster_centers_[:, 0], c=hc.labels)
    else:
        plt.scatter(data.data[:, 0], data.data[:, 1], c='lightgray')
        plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=hc.labels)
    plt.show()


def main():
    parser = argparse.ArgumentParser(prog="Analyze Large Dataset",
                                     description="Measure execution time or view results of fuzzy hierarchical clustering on a large 2D dataset")
    parser.add_argument("action", choices=["time", "plot"], help="'time' if the program should measure execution time, 'plot' if the program should plot fuzzy and non-fuzzy clustering results")
    parser.add_argument("filename", help="path to the data file")
    parser.add_argument("-c", "--n_clusters", type=int, default=6, help="number of expected clusters")
    parser.add_argument("-s", "--size", "--step", type=int, help="for 'time', the step size of consecutive data measurements, for 'plot', the size of data for non-fuzzy clustering")
    parser.add_argument("-f", "--flip_coordinates", action="store_true", help="for plotting, flip the data's coordinates")
    args = parser.parse_args()
    path = args.filename
    if os.path.exists(path):
        with open(path) as f:
            long_data = DataEntry(f.read(), "data", 2, has_labels=False)
            if args.action == "time":
                measure_time(long_data, args.n_clusters, args.size)
            elif args.action == "plot":
                plot_clustering(long_data, args.n_clusters, args.size, args.flip_coordinates)
    else:
        print("File not found")


if __name__ == "__main__":
    main()

