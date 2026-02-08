import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

granules_numbers = [50, 100, 200]
relation_types = ['t', 'e', 'g']
line_type = [':', '-', '--']
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 9))
legend_labels = []

data_blobs = pandas.read_csv("wyniki/blobs_accuracy.csv")
data_circles = pandas.read_csv("wyniki/circles_accuracy.csv")
data_corners = pandas.read_csv("wyniki/corners_accuracy.csv")
data_crescents = pandas.read_csv("wyniki/crescents_accuracy.csv")
data_laguna = pandas.read_csv("wyniki/laguna_accuracy.csv")
data_spheres = pandas.read_csv("wyniki/spheres_accuracy.csv")
data = pandas.concat([data_blobs, data_circles, data_corners, data_crescents, data_laguna, data_spheres])
filtered_size = data_laguna[data_laguna['data size'] == 10000]

plt.figure()
for i in range(3):
    for j in range(3):
        d = filtered_size[(filtered_size['granules number'] == granules_numbers[i]) & (filtered_size['relation type'] == relation_types[j])]
        plt.plot(d['ksi'], d['accuracy'], color=colors[i*3+j], linestyle=line_type[i])
        legend_labels.append(str(granules_numbers[i]) + " granules, relation type " + relation_types[j])
plt.legend(legend_labels)
plt.show()

to_group = data[["relation type", "accuracy"]]
grouped = to_group.groupby(by=["relation type"], as_index=False).mean()
print(to_group.groupby(by=["relation type"]).mean().to_string())
plt.figure()
plt.bar(grouped["relation type"], grouped["accuracy"], color=cmap(np.linspace(0, 1, 3)))
plt.title('Accuracy by relation type')
plt.xlabel('relation type')
plt.ylabel('accuracy')
plt.show()

to_group = data[["ksi", "accuracy"]]
grouped = to_group.groupby(by=["ksi"], as_index=False).mean()
print(grouped.to_string())
plt.figure()
plt.bar(grouped["ksi"], grouped["accuracy"], color=cmap(np.linspace(0, 1, 19)), width=0.02)
plt.title('Mean accuracy by ksi parameter')
plt.xlabel('ksi')
plt.ylabel('accuracy')
plt.show()

grouped = to_group.groupby(by=["ksi"], as_index=False).std()
print(grouped.to_string())
plt.figure()
plt.bar(grouped["ksi"], grouped["accuracy"], color=cmap(np.linspace(0, 1, 19)), width=0.02)
plt.title('Std of accuracy by ksi parameter')
plt.xlabel('ksi')
plt.ylabel('std of accuracy')
plt.show()
