import os
import pandas
from data_entry import DataEntry
import argparse

# Constants
folderPath = "wyniki/"

fullData = {}
names = []

# Values picked by visually deciding the best number of clusters for a given shape
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
data_size_presets = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]


def measure_accuracy_recall_precision(linkage, specific_file=None, shape_dependent=False):
    linkages = {"single": "sl", "complete": "cl"}
    if linkages.get(linkage) is None:
        linkage = "single"
    if specific_file is not None:
        print(fullData.keys())
        data_to_measure = fullData.get(specific_file)
        if data_to_measure is not None:
            print(data_to_measure.measure_accuracy(shape_dependent, linkage).to_string())
    else:
        for root in names_roots:
            accuracy_results = pandas.DataFrame()
            for data_size in data_size_presets:
                print(root + str(data_size))
                result = fullData[root + str(data_size)].measure_accuracy(shape_dependent_membership=shape_dependent,
                                                                          linkage=linkage)
                accuracy_results = pandas.concat([accuracy_results, result], ignore_index=True)
            if shape_dependent:
                accuracy_results.to_csv(folderPath + root + "_" + linkages[linkage] + "_shape_dep_accuracy.csv")
            else:
                accuracy_results.to_csv(folderPath + root + "_" + linkages[linkage] + "_accuracy.csv")


def measure_strict_accuracy_recall_precision(linkage, specific_file=None):
    linkages = {"single": "sl", "complete": "cl"}
    if linkages.get(linkage) is None:
        linkage = "single"
    if specific_file is not None:
        data_to_measure = fullData.get(specific_file)
        if data_to_measure is not None:
            print(data_to_measure.measure_strict_accuracy(linkage))
    else:
        for root in names_roots:
            accuracy_results = pandas.DataFrame()
            for data_size in data_size_presets:
                print(root + str(data_size))
                result = fullData[root + str(data_size)].measure_strict_accuracy(linkage=linkage)
                accuracy_results = pandas.concat([accuracy_results, result], ignore_index=True)
            accuracy_results.to_csv(folderPath + root + "_" + linkages[linkage] + "_strict_accuracy.csv")


def main():
    parser = argparse.ArgumentParser(prog="Measure Quality",
                                     description="Measure quality (accuracy, recall, precision) of fuzzy hierarchical clusterization")
    parser.add_argument("-f", "--filename", help="specific file to measure, measures all files if not given; result is printed for one file, saved to .csv for all files")
    parser.add_argument("-l", "--linkage", choices=["sl", "single", "cl", "complete"], default="sl", help="linkage of the hierachical clustering: single(sl) or complete(cl), single by default")
    parser.add_argument("-r", "--relation_dependent", action="store_true", help="calculate granule memberships based on granule shape/relation type")
    parser.add_argument("--strict", action="store_true", help="measure strict (non-fuzzy) clustering results in addition to fuzzy ones")
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
    measure_accuracy_recall_precision(args.linkage, args.filename, args.relation_dependent)
    if args.strict:
        measure_strict_accuracy_recall_precision(args.linkage, args.filename)


if __name__ == "__main__":
    main()
