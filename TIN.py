import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection


class TIN:
    def __init__(self, datapoints: np.ndarray):
        if (datapoints.ndim != 2):
            raise ValueError(
                "Incompatible Array Dimension in TIN object creation")
        if (datapoints.shape[1] != 3):
            raise ValueError(
                ("Incompatible Array shape in TIN object creation" +
                 "must be composed of three columns: x,y and z/height"))
        self.x = datapoints[:, 0]
        self.y = datapoints[:, 1]
        self.altitude = datapoints[:, 2]
        if not np.all(self.altitude >= 0):
            min_alt = np.min(self.altitude)
            print(f"Negative elevation value detected ({min_alt:.6f}), but processing continues.")

        points = np.column_stack((self.x, self.y))
        self.triangulation = Delaunay(points)
        self.triangulation.vertex_to_simplex

    def plot_triangulation(self, dot_size=1, auto_show=True,
                           title="TIN Triangulation"):

        fig = plt.figure(title)

        ax = plt.axes()
        ax.set_aspect('equal')
        ax.triplot(self.x,  self.y, self.triangulation.simplices)
        if dot_size > 0:
            ax.plot(self.x, self.y, 'o', markersize=dot_size)
        if auto_show:
            plt.show()
        return ax

    def find_elevation(self, points, Dis, Dis_min, Angle_min, display=False):
        if isinstance(points, dict):
            result1 = [value for value in points.values()]
            datapoints1 = np.array(result1)
        elif isinstance(points, np.ndarray):
            datapoints1 = points


        ground_points = []
        no_ground_points = []

        for point in datapoints1:
            x, y, true_z = point

            index = Delaunay.find_simplex(self.triangulation, [(x, y)], bruteforce=False, tol=None)

            if index == -1:
                continue
            p0, p1, p2 = self.triangulation.simplices[index][0]
            f0, f1, f2 = self.altitude[p0], self.altitude[p1], self.altitude[p2]
            p0 = self.triangulation.points[p0]
            p1 = self.triangulation.points[p1]
            p2 = self.triangulation.points[p2]

            A = (p1[1] - p0[1]) * (f2 - f0) - (p2[1] - p0[1]) * (f1 - f0)
            B = (f1 - f0) * (p2[0] - p0[0]) - (f2 - f0) * (p1[0] - p0[0])
            C = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
            D = -(A * p0[0] + B * p0[1] + C * f0)

            interpolate_z = abs(A * x + B * y + C * true_z + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)

            if interpolate_z <= Dis_min:
                ground_points.append(point)

            else:
                angles = []
                for (xi, yi), f in zip([p0, p1, p2], [f0, f1, f2]):

                    distance_to_vertex = np.sqrt((xi - x) ** 2 + (yi - y) ** 2 + (f - true_z) ** 2)
                    if distance_to_vertex == 0:

                        angles.append(0)
                    else:
                        angle_value = interpolate_z / distance_to_vertex
                        angle_value = np.clip(angle_value, -1.0, 1.0)
                        angle = np.arcsin(angle_value)
                        angles.append(angle)

                if Dis_min < interpolate_z < Dis and all(angle < np.radians(Angle_min) for angle in angles):
                    ground_points.append(point)
                else:
                    no_ground_points.append(point)
            if display:
                ax = self.plot_elevation_profile(autoShow=False, alpha=0.7)
                ax.plot([p0[0], p1[0], p2[0], p0[0]], [p0[1], p1[1], p2[1], p0[1]],
                        [f0, f1, f2, f0], marker=".", linewidth=2, c="red")
                ax.plot([x], [y], [true_z], marker="D", markersize=7, c="red")
                lab1 = mlines.Line2D([], [], color='red', marker='.', linestyle='None', markersize=10,
                                     label='Sample Points Used')
                lab2 = mlines.Line2D([], [], color='red', marker='D', linestyle='None', markersize=10,
                                     label='Interpolated Point')
                plt.legend(handles=[lab1, lab2])
                plt.show()

        return ground_points, no_ground_points


    def add_point(self, new_point: np.ndarray):

        self.x = np.append(self.x, new_point[0])
        self.y = np.append(self.y, new_point[1])
        self.altitude = np.append(self.altitude, new_point[2])
        points = np.column_stack((self.x, self.y))
        self.triangulation = Delaunay(points)


    def plot_elevation_profile1(self, autoShow=True,
                               alpha=1,  title="TIN Elevation Profile"):

        fig = plt.figure(title)
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(self.x, self.y, self.altitude, color='white', cmap="BrBG", alpha=alpha)
        ax.set_aspect('equal')
        #ax.set_box_aspect([1, 1, 0.5])
        if autoShow:
            plt.show()
        return ax

    def plot_elevation_profile(self, autoShow=True,
                               alpha=1,  title="TIN Elevation Profile", cbar_ticks=None, zmin=None, zmax=None):

        fig = plt.figure(title)
        ax = plt.axes(projection='3d')

        ax.set_axis_off()

        ax.plot_trisurf(
            self.x,
            self.y,
            self.altitude,
            color='white',
            cmap="BrBG",
            alpha=alpha,
            linewidth=0
        )

        ax.set_aspect('equal')

        ax.view_init(elev=18, azim=0)

        surf = ax.plot_trisurf(
            self.x,
            self.y,
            self.altitude,
            cmap="BrBG",
            alpha=alpha,
            linewidth=0
        )

        cbar = fig.colorbar(
            surf,  # 关联曲面对象
            ax=ax,
            orientation='horizontal',
            shrink=0.6,
            aspect=20,
            pad=0.1
        )
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        elif zmin is not None and zmax is not None:
            cbar.set_ticks(np.linspace(zmin, zmax, 5))

        if autoShow:
            plt.show()
        return ax

    def plot_dual_graph(self, auto_show=True, dot_size=2.0):

        n_triangles = len(self.triangulation.simplices)
        midpoints = self.triangulation.points[self.triangulation.simplices]
        midpoints = np.average(midpoints, axis=1)

        x = self.triangulation.neighbors[range(n_triangles)]
        tilehelp = np.tile(np.arange(n_triangles), (3, 1)).T
        tilehelp = tilehelp.reshape((-1,))

        x = np.reshape(x, (n_triangles*3))
        pair_indexes = np.zeros(2*n_triangles*3, dtype='int32')
        pair_indexes[0::2] = tilehelp
        pair_indexes[1::2] = x
        pair_indexes = np.reshape(pair_indexes, (3*n_triangles, 2))

        pair_indexes = np.delete(pair_indexes, np.where(pair_indexes < 0), axis=0)


        pair_indexes.sort(axis=1)
        pair_indexes = np.unique(pair_indexes, axis=0)

        lc = LineCollection(midpoints[pair_indexes], linewidths=1, colors="green")
        ax = self.plot_triangulation(auto_show=False, title="TIN Dual Graph")
        ax.set_aspect('equal')
        ax.add_collection(lc)
        ax.scatter(midpoints[:, 0], midpoints[:, 1], s=dot_size)
        lab = mlines.Line2D([], [], color="blue",
                            markersize=10, label="Triangulation")
        lab2 = mlines.Line2D([], [], color="green",
                             markersize=10, label="Dual Graph")
        plt.legend(handles=[lab, lab2])
        if auto_show:
            plt.show()
        return ax


    def find_triangle_for_point(self, point):
        simplex_index = self.triangulation.find_simplex(point)
        return simplex_index
