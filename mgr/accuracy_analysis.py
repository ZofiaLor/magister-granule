import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os

names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
linkages = {"single": "sl", "complete": "cl"}
granules_numbers = [50, 100, 200]
relation_types = ['t', 'e', 'g']
line_type = [':', '-', '--']
cmap = mpl.colormaps['tab10']
colors = cmap(np.linspace(0, 1, 9))
legend_labels = []
data_shapes = dict()
data_shapes_noise = dict()


def read_data(shape_dep, linkage, strict):
    if shape_dep:
        suffix = "_" + linkages[linkage] + "_" + "shape_dep_accuracy.csv"
    else:
        suffix = "_" + linkages[linkage] + "_" + "accuracy.csv"
    for name in names_roots:
        filename = "wyniki/" + name + suffix
        if not os.path.exists(filename):
            print(filename + " not found")
            return None, None
        data_shapes[name] = pandas.read_csv(filename)
        if shape_dep:
            df = data_shapes[name].copy()
            for index, row in df.iterrows():
                df.at[index, 'accuracy'] = row['accuracy'] * (1 - row['noise percentage'])
                df.at[index, 'recall'] = row['recall'] * (1 - row['noise percentage'])
                df.at[index, 'precision'] = row['precision'] * (1 - row['noise percentage'])
            data_shapes_noise[name] = df
        if strict:
            data_strict = pandas.read_csv("wyniki/" + name + "_sl_strict_accuracy.csv")[
                ["granules", "accuracy", "recall", "precision"]].groupby(by=["granules"]).mean()
            print(name, "\n", data_strict)

    if shape_dep:
        return pandas.concat(data_shapes.values()), pandas.concat(data_shapes_noise.values())
    return pandas.concat(data_shapes.values()), None


def analyze_data_shape(shape_dep):
    print("Data shape impact analysis")
    for name in names_roots:
        print("\n" + name)
        print("mean")
        print(data_shapes[name][["accuracy", "recall", "precision"]].mean())
        print("\nmax")
        print(data_shapes[name][["data size", "accuracy", "recall", "precision"]].groupby(by=["data size"]).max())

        if shape_dep:
            print("\nWith noise")
            print("mean")
            print(data_shapes_noise[name][["accuracy", "recall", "precision"]].mean())
            print("\nmax")
            print(data_shapes_noise[name][["data size", "accuracy", "recall", "precision"]].groupby(by=["data size"]).max())


def analyze_relation_type(data, data_noise, shape_dep):
    print("Relation type impact analysis")
    print("mean")
    print(data[["relation type", "accuracy", "recall", "precision"]].groupby(by=["relation type"]).mean().to_string())
    if shape_dep:
        print("\nWith noise")
        print("mean")
        print(data_noise[["relation type", "accuracy", "recall", "precision"]].groupby(
            by=["relation type"]).mean().to_string())


def analyze_granule_number(data, data_noise, shape_dep):
    print("Granule number impact analysis")
    print("mean")
    print(data[["granules number", "accuracy", "recall", "precision"]].groupby(by=["granules number"]).mean().to_string())
    print("noise")
    if shape_dep:
        print("\nWith noise")
        print("mean")
        print(data_noise[["granules number", "accuracy", "recall", "precision"]].groupby(by=["granules number"]).mean().to_string())


def analyze_xi(data, data_noise, shape_dep):
    print("Granule number impact analysis")
    print("mean")
    to_group = data[["ksi", "accuracy", "recall", "precision"]]
    grouped = to_group.groupby(by=["ksi"], as_index=False).mean()
    print(grouped.to_string())
    plt.figure(figsize=(9,9))
    plt.plot(grouped["ksi"], grouped["accuracy"], marker='o')
    plt.plot(grouped["ksi"], grouped["recall"], marker='o')
    plt.plot(grouped["ksi"], grouped["precision"], marker='o')
    plt.legend(["dokładność", "trafność", "precyzja"])
    plt.title(r'Średnia dokładność, trafność i precyzja w zależności od parametru $ \xi $')
    plt.xlabel(r'$ \xi $')
    plt.ylabel('wynik')
    plt.show()

    if shape_dep:
        print("\nWith noise")
        print("mean")
        to_group = data_noise[["ksi", "accuracy", "recall", "precision"]]
        grouped = to_group.groupby(by=["ksi"], as_index=False).mean()
        print(grouped.to_string())
        plt.figure(figsize=(9, 9))
        plt.plot(grouped["ksi"], grouped["accuracy"], marker='o')
        plt.plot(grouped["ksi"], grouped["recall"], marker='o')
        plt.plot(grouped["ksi"], grouped["precision"], marker='o')
        plt.legend(["dokładność", "trafność", "precyzja"])
        plt.title(r'Średnia dokładność, trafność i precyzja w zależności od parametru $ \xi $, przy uwzględnieniu szumu')
        plt.xlabel(r'$ \xi $')
        plt.ylabel('wynik')
        plt.show()


def analyze_combination(linkage):
    metric_names = [["accuracy", "acc", "dokładności", "dokładność"], ["recall", "rec", "trafności", "trafność"], ["precision", "prec", "precyzji", "precyzja"]]
    for name, shape in data_shapes.items():
        folderPath = "img/complex/" + name + "/"
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filtered_size = shape[["granules number", "relation type", "ksi", "accuracy", "recall", "precision"]].groupby(by=["granules number", "relation type", "ksi"]).mean().reset_index()

        for metric in metric_names:
            plt.figure(figsize=(12, 10))
            for i in range(3):
                for j in range(3):
                    d = filtered_size[(filtered_size['granules number'] == granules_numbers[i]) & (filtered_size['relation type'] == relation_types[j])]
                    plt.plot(d['ksi'], d[metric[0]], linestyle=line_type[i])
                    plt.title("Zależność " + metric[2] + " grupowania danych " + name + r" od liczby granul, typu relacji i wartości $ \xi $ przy łączności " + linkage + " linkage")
                    plt.xlabel(r"$ \xi $")
                    plt.ylabel(metric[3])
                    legend_labels.append(str(granules_numbers[i]) + " granul, typ relacji: " + relation_types[j])
            plt.legend(legend_labels)
            # plt.show()
            plt.savefig(folderPath + name + "_complex_ " + metric[1] + "_" + linkages[linkage] + ".pdf")
            plt.close()


def main():
    parser = argparse.ArgumentParser(prog="Clustering Quality Analysis",
                                     description="Analyze quality results based on different parameters")
    parser.add_argument("-l", "--linkage", choices=["single", "complete"], default="single",
                        help="linkage of the hierachical clustering: single or complete, single by default")
    parser.add_argument("-s", "--shape_dependent", action="store_true", help="analyze results of when granule memberships are based on granule shape/relation type")
    parser.add_argument("-d", "--data_shape", action="store_true", help="analyze quality based on data shape")
    parser.add_argument("-r", "--relation_type", action="store_true", help="analyze quality based on relation type")
    parser.add_argument("-g", "--granule_number", action="store_true", help="analyze quality based on granule number")
    parser.add_argument("-x", "--xi", action="store_true", help="analyze quality based on xi (threshold) parameter")
    parser.add_argument("-c", "--complex", action="store_true", help="analyze quality based on the combination of relation type, granule number and xi parameter")
    parser.add_argument("--strict", action="store_true",
                        help="analyze strict (non-fuzzy) clustering results in addition to fuzzy ones")
    args = parser.parse_args()

    data, data_noise = read_data(args.shape_dependent, args.linkage, args.strict)
    if data is None:
        return
    if args.data_shape:
        analyze_data_shape(args.shape_dependent)
    if args.relation_type:
        analyze_relation_type(data, data_noise, args.shape_dependent)
    if args.granule_number:
        analyze_granule_number(data, data_noise, args.shape_dependent)
    if args.xi:
        analyze_xi(data, data_noise, args.shape_dependent)
    if args.complex:
        analyze_combination(args.linkage)


if __name__ == "__main__":
    main()
