import numpy as np
import math

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filename = "spheres/spheres50000"

    with open("dane/" + filename + ".data") as f:
        data = f.read().splitlines()
    dims = []
    for i in range(len(data[0].split())):
        dims.append([])
    for line in data:
        coords = line.split()
        for i in range(len(coords)):
            dims[i].append(float(coords[i]))
    data = np.array(dims).transpose()
    labels = np.zeros(data.shape[0])

    if data.shape[1] == 2:
        fig, ax = plt.subplots()

        pts = ax.scatter(data[:, 0], data[:, 1])
        selector = SelectFromCollection(ax, pts)

        def accept(event):
            if event.key == "enter":
                selector.disconnect()
                ax.set_title("")
                fig.canvas.draw()
            elif event.key.isnumeric():
                for i in selector.ind:
                    labels[i] = int(event.key)
                print("Label ", event.key, " assigned to ", len(selector.ind), "points")

        fig.canvas.mpl_connect("key_press_event", accept)
        ax.set_title("Press enter to accept selected points.")

        plt.show()

        plt.plot()
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.show()
    elif "spheres" in filename:
        label1 = 0
        for i in range(len(data)):
            if abs(data[i, 0]) <= 1 and abs(data[i, 1]) <= 1 and abs(data[i, 2]) <= 1:
            # if math.sqrt(data[i, 0] ** 2 + data[i, 1] ** 2 + data[i, 2] ** 2) <= 1:
                labels[i] = 1
                label1 += 1
        print("Label 1 assigned to ", label1, " points")

        plt.plot()
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=labels)
        plt.show()

    with open("dane_labelled/" + filename + ".data", "w") as f:
        for i in range(len(labels)):
            f.write("\t".join([str(data[i, 0]), str(data[i, 1]), str(int(labels[i]))]))
            f.write("\n")
