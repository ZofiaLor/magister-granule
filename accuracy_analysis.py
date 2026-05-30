import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
# d = {}
# for name in names_roots:
#     with open("./dane_labelled/" + name + "/" + name + "1000.data", 'r') as f:
#         lines = f.read().splitlines()
#         data = []
#         for line in lines:
#             l = line.split()
#             if len(l) > 3:
#                 data.append([float(l[0]), float(l[1]), float(l[2]), float(l[3])])
#             else:
#                 data.append([float(l[0]), float(l[1]), float(l[2])])
#         d[name] = np.array(data)
#
# fig = plt.figure()
# ax = fig.add_subplot(2, 3, 1)
# ax.scatter(d["blobs"][:, 0], d["blobs"][:, 1], c=d["blobs"][:, 2])
# ax.set_title("Blobs")
# ax = fig.add_subplot(2, 3, 2)
# ax.scatter(d["circles"][:, 0], d["circles"][:, 1], c=d["circles"][:, 2])
# ax.set_title("Circles")
# ax = fig.add_subplot(2, 3, 3)
# ax.scatter(d["corners"][:, 0], d["corners"][:, 1], c=d["corners"][:, 2])
# ax.set_title("Corners")
# ax = fig.add_subplot(2, 3, 4)
# ax.scatter(d["crescents"][:, 0], d["crescents"][:, 1], c=d["crescents"][:, 2])
# ax.set_title("Crescents")
# ax = fig.add_subplot(2, 3, 5)
# ax.scatter(d["laguna"][:, 0], d["laguna"][:, 1], c=d["laguna"][:, 2])
# ax.set_title("Laguna")
# ax = fig.add_subplot(2, 3, 6, projection='3d')
# ax.scatter3D(d["spheres"][:, 0], d["spheres"][:, 1], d["spheres"][:, 2], c=d["spheres"][:, 3], alpha=0.5)
# ax.set_title("Spheres")
#
# plt.show()
#
# with open("./dane_labelled/laguna/laguna20000.data", 'r') as f:
#     lines = f.read().splitlines()
#     data = []
#     for line in lines:
#         l = line.split()
#         data.append([float(l[0]), float(l[1]), float(l[2])])
#     data = np.array(data)
#     plt.figure()
#     plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
#     plt.show()

granules_numbers = [50, 100, 200]
relation_types = ['t', 'e', 'g']
line_type = [':', '-', '--']
cmap = mpl.colormaps['tab10']
colors = cmap(np.linspace(0, 1, 9))
legend_labels = []

data_blobs = pandas.read_csv("wyniki/blobs_sl_accuracy.csv")
data_circles = pandas.read_csv("wyniki/circles_sl_accuracy.csv")
data_corners = pandas.read_csv("wyniki/corners_sl_accuracy.csv")
data_crescents = pandas.read_csv("wyniki/crescents_sl_accuracy.csv")
data_laguna = pandas.read_csv("wyniki/laguna_sl_accuracy.csv")
data_spheres = pandas.read_csv("wyniki/spheres_sl_accuracy.csv")

print("blobs\n", data_blobs[["accuracy", "recall", "precision"]].mean())
print("circles\n", data_circles[["accuracy", "recall", "precision"]].mean())
print("corners\n", data_corners[["accuracy", "recall", "precision"]].mean())
print("crescents\n", data_crescents[["accuracy", "recall", "precision"]].mean())
print("laguna\n", data_laguna[["accuracy", "recall", "precision"]].mean())
print("spheres\n", data_spheres[["accuracy", "recall", "precision"]].mean())

data = pandas.concat([data_blobs, data_circles, data_corners, data_crescents, data_laguna, data_spheres])

for name, shape in {"blobs": data_blobs, "circles": data_circles, "corners": data_corners, "crescents": data_crescents, "laguna": data_laguna, "spheres": data_spheres}.items():
    filtered_size = shape[["granules number", "relation type", "ksi", "accuracy", "recall", "precision"]].groupby(by=["granules number", "relation type", "ksi"]).mean().reset_index()
    print(filtered_size.to_string())

    plt.figure(figsize=(12, 10))
    for i in range(3):
        for j in range(3):
            d = filtered_size[(filtered_size['granules number'] == granules_numbers[i]) & (filtered_size['relation type'] == relation_types[j])]
            plt.plot(d['ksi'], d['accuracy'], linestyle=line_type[i])
            plt.title("Zależność dokładności grupowania danych " + name + r" od liczby granul, typu relacji i wartości $ \xi $ przy łączności single linkage")
            plt.xlabel(r"$ \xi $")
            plt.ylabel("dokładność")
            legend_labels.append(str(granules_numbers[i]) + " granul, typ relacji: " + relation_types[j])
    plt.legend(legend_labels)
    # plt.show()
    plt.savefig("img/iter80/" + name + "/" + name + "_complex_acc_sl.pdf")
    plt.close()

    plt.figure(figsize=(12, 10))
    for i in range(3):
        for j in range(3):
            d = filtered_size[(filtered_size['granules number'] == granules_numbers[i]) & (
                        filtered_size['relation type'] == relation_types[j])]
            plt.plot(d['ksi'], d['recall'], linestyle=line_type[i])
            plt.title(
                "Zależność trafności grupowania danych " + name + r" od liczby granul, typu relacji i wartości $ \xi $ przy łączności single linkage")
            plt.xlabel(r"$ \xi $")
            plt.ylabel("trafność")
            legend_labels.append(str(granules_numbers[i]) + " granul, typ relacji: " + relation_types[j])
    plt.legend(legend_labels)
    # plt.show()
    plt.savefig("img/iter80/" + name + "/" + name + "_complex_rec_sl.pdf")
    plt.close()

    plt.figure(figsize=(12, 10))
    for i in range(3):
        for j in range(3):
            d = filtered_size[(filtered_size['granules number'] == granules_numbers[i]) & (
                    filtered_size['relation type'] == relation_types[j])]
            plt.plot(d['ksi'], d['precision'], linestyle=line_type[i])
            plt.title(
                "Zależność precyzji grupowania danych " + name + r" od liczby granul, typu relacji i wartości $ \xi $ przy łączności single linkage")
            plt.xlabel(r"$ \xi $")
            plt.ylabel("precyzji")
            legend_labels.append(str(granules_numbers[i]) + " granul, typ relacji: " + relation_types[j])
    plt.legend(legend_labels)
    # plt.show()
    plt.savefig("img/iter80/" + name + "/" + name + "_complex_prec_sl.pdf")
    plt.close()

to_group = data[["relation type", "accuracy", "recall", "precision"]]
grouped = to_group.groupby(by=["relation type"], as_index=False).mean()
print(to_group.groupby(by=["relation type"]).mean().to_string())
plt.figure()
plt.bar(grouped["relation type"], grouped["accuracy"], color=cmap(np.linspace(0, 1, 3)))
plt.title('Accuracy by relation type')
plt.xlabel('relation type')
plt.ylabel('accuracy')
plt.show()

to_group = data[["granules number", "accuracy", "recall", "precision"]]
grouped = to_group.groupby(by=["granules number"]).mean()
print(grouped.to_string())

to_group = data[["ksi", "accuracy", "recall", "precision"]]
grouped = to_group.groupby(by=["ksi"], as_index=False).mean()
print(grouped.to_string())
plt.figure()
plt.plot(grouped["ksi"], grouped["accuracy"], marker='o')
plt.plot(grouped["ksi"], grouped["recall"], marker='o')
plt.plot(grouped["ksi"], grouped["precision"], marker='o')
# plt.bar(grouped["ksi"], grouped["accuracy"], color=cmap(np.linspace(0, 1, 19)), width=0.02)
plt.legend(["dokładność", "trafność", "precyzja"])
plt.title('Średnia dokładność, trafność i precyzja w zależności od parametru ksi')
plt.xlabel('ksi')
plt.ylabel('wynik')
plt.show()

# grouped = to_group.groupby(by=["ksi"], as_index=False).std()
# print(grouped.to_string())
# plt.figure()
# plt.bar(grouped["ksi"], grouped["accuracy"], color=cmap(np.linspace(0, 1, 19)), width=0.02)
# plt.title('Std of accuracy by ksi parameter')
# plt.xlabel('ksi')
# plt.ylabel('std of accuracy')
# plt.show()
