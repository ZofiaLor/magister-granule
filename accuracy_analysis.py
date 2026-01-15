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

data = pandas.read_csv("wyniki/laguna_accuracy.csv")
filtered_size = data[data['data size'] == 10000]
for i in range(3):
    for j in range(3):
        d = filtered_size[(filtered_size['granules number'] == granules_numbers[i]) & (filtered_size['relation type'] == relation_types[j])]
        plt.plot(d['ksi'], d['accuracy'], color=colors[i*3+j], linestyle=line_type[i])
        legend_labels.append(str(granules_numbers[i]) + " granules, relation type " + relation_types[j])
plt.legend(legend_labels)
plt.show()

to_group = data[["data size", "granules number", "relation type", "accuracy"]]
print(to_group.groupby(by=["data size", "granules number", "relation type"]).mean().to_string())
