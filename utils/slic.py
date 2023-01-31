from skimage import measure
import numpy as np
import scipy
from skimage.segmentation import mark_boundaries
import cv2

from matplotlib import pyplot as plt

CONVERGENCE_DICT = {0: 0.0, 10: 0.0005, 20: 0.001, 100: 0.005, 200: 0.01, 500: 0.02}
CONN1 = np.ones((3, 3))
CONN1[0, 0] = 0
CONN1[0, 2] = 0
CONN1[2, 0] = 0
CONN1[2, 2] = 0
CONN1 = CONN1.astype(np.uint8)
CONN2 = np.ones((3, 3))
CONN2 = CONN2.astype(np.uint8)


class SLIC:
    def __init__(self, number_of_regions, pixel_features, gradient, original_image=None, log_freq=50):
        self.X, self.Y, _ = pixel_features.shape
        self.gradient = gradient
        self.pixel_features = pixel_features
        self.number_of_regions = number_of_regions
        self.original_image = original_image
        self.log_freq = log_freq

        self.convergence_factor = 0
        for key, value in CONVERGENCE_DICT.items():
            if self.number_of_regions >= key:
                self.convergence_factor = value

        self.labels = np.full((self.X, self.Y), -1)
        self.region_features = None

        self.distance = np.full((self.X, self.Y), np.inf)

    def init_regions(self):
        """ Initialize clusters based on an evenly spaced grid and the min value in the average channel gradient in
        5x5 neighbourhood around these evenly spaced points """
        ny = int(np.sqrt(self.number_of_regions * self.Y / self.X))
        nx = self.number_of_regions // ny
        ys = np.mean((np.linspace(0, self.Y, ny + 1)[1:], np.linspace(0, self.Y, ny + 1)[:-1]), axis=0).astype(int)
        xs = np.mean((np.linspace(0, self.X, nx + 1)[1:], np.linspace(0, self.X, nx + 1)[:-1]), axis=0).astype(int)

        label = 0
        for x in xs:
            for y in ys:
                mask = np.zeros((self.X, self.Y))
                mask[x, y] = 1
                mask = scipy.ndimage.binary_dilation(mask, structure=np.ones((5, 5)))
                lg_x, lg_y = np.argwhere((self.gradient == self.gradient[mask].min()) & mask)[0]
                self.labels[lg_x, lg_y] = label
                label += 1
        self.number_of_regions = label
        self.region_features = np.zeros((self.number_of_regions, self.pixel_features.shape[2]))

    def assign_regions(self):
        """ Assignment step:
        For each cluster:
            Find neighbourhood
            Calculate distance feature distance to neighbouring pixels
            If distance smaller than smallest distance from previous clusters assign label to pixel
        """
        for region_label in range(self.number_of_regions):
            region_feature = self.region_features[region_label]
            region = self.labels == region_label
            neighbourhood = cv2.dilate(region.astype(np.uint8), CONN2, iterations=5).astype(bool) & ~region
            distances = np.sum((self.pixel_features[neighbourhood] - region_feature) ** 2, axis=1)
            smaller_distance = self.distance[neighbourhood] > distances
            if np.any(smaller_distance):
                new_labels = self.labels[neighbourhood]
                new_labels[smaller_distance] = region_label
                self.labels[neighbourhood] = new_labels

                new_neighbourhood_distance = self.distance[neighbourhood]
                new_neighbourhood_distance[smaller_distance] = distances[smaller_distance]
                self.distance[neighbourhood] = new_neighbourhood_distance

    def update_regions(self):
        """ Update mean feature of each cluster """
        self.distance = np.full((self.X, self.Y), np.inf)
        for region_label in range(self.number_of_regions):
            self.region_features[region_label] = np.mean(self.pixel_features[self.labels == region_label], axis=0)
            region_feature = self.region_features[region_label]
            region = self.labels == region_label
            self.distance[region] = np.sum((self.pixel_features[region] - region_feature) ** 2, axis=1)

    def assert_connectivity(self):
        """ Perform a connected component analysis, set all pixels not belonging to the biggest shape to -1 (undefined)
        For each -1 shape assign it the label of a random neighbouring cluster."""
        for region_label in range(self.number_of_regions):
            shape_labels, shape_num = measure.label(self.labels == region_label, return_num=True, connectivity=1)
            if shape_num > 1:
                shapes = [shape_label == shape_labels for shape_label in range(1, np.max(shape_labels) + 1)]
                biggest_shape = None
                biggest_size = -1
                for shape in shapes:
                    if np.sum(shape) > biggest_size:
                        biggest_shape = shape
                        biggest_size = np.sum(shape)
                self.labels[self.labels == region_label] = -1
                self.labels[biggest_shape] = region_label

        if False:
            print(f'Post-Assert1: Number of changed labels {np.sum(self.old_labels != self.labels)}')
            self.log()

        shape_labels = measure.label(self.labels == -1, connectivity=1)
        shapes = [shape_label == shape_labels for shape_label in range(1, np.max(shape_labels) + 1)]
        for shape in shapes:
            neighbours = np.unique(self.labels[cv2.dilate(shape.astype(np.uint8), CONN1, iterations=1).astype(bool)])
            neighbours = neighbours[neighbours != -1]
            self.labels[shape] = np.random.choice(neighbours)

    def iterate(self, num_iter):
        """ Main Iteration Loop It first initializes the regions with self.init_regions() and then iteratively
        applies self.update_regions(), self.assign_regions(), self.assert_connectivity() """
        if np.all(self.labels == -1):
            self.init_regions()
        self.old_labels = self.labels.copy()
        for step in range(num_iter):
            print(step, end='\r')

            self.update_regions()
            self.assign_regions()
            self.assert_connectivity()

            # Show intermediate result
            if self.original_image is not None and (step % self.log_freq == 0):
                print()
                print(f'Number of changed labels {np.sum(self.old_labels != self.labels)}')
                self.log()

            # Convergence Check
            if np.sum(self.old_labels != self.labels) <= self.convergence_factor * self.labels.size:
                print(f'Converged at step {step}.')
                break
            self.old_labels = self.labels.copy()
        if self.original_image is not None:
            self.log()
        return self.labels

    def log(self):
        """ Mark boundaries on original image and plot image."""
        display = mark_boundaries(self.original_image, self.labels, outline_color=None, mode='outer',
                                  background_label=0)
        plt.imshow(display, cmap='nipy_spectral')
        plt.show()
