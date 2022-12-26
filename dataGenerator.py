import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice

class toyData():
    def __init__(self, dataset_n=None, random_seed_n=0, n_samples=500, plot=False):
        self.dataset_n = dataset_n
        self.random_seed_n = random_seed_n
        self.n_samples = n_samples
        self.plot = plot
        
    def generate_data(self):
        from sklearn import cluster, datasets, mixture
        np.random.seed(self.random_seed_n)

        # ============
        # Generate datasets. We choose the size big enough to see the scalability
        # of the algorithms, but not too big to avoid too long running times
        # ============
        n_samples = self.n_samples
        noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
        blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
        no_structure = np.random.rand(n_samples, 2), None

        # Anisotropicly distributed data
        random_state = 170
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        aniso = (X_aniso, y)

        # blobs with varied variances
        varied = datasets.make_blobs(
            n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
        )

        # ============
        # Set up cluster parameters
        # ============
        if self.plot:
            plt.figure(figsize=(9 * 2 + 3, 13))
            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01)

        plot_num = 1

        default_base = {
            "quantile": 0.3,
            "eps": 0.3,
            "damping": 0.9,
            "preference": -200,
            "n_neighbors": 3,
            "n_clusters": 3,
            "min_samples": 7,
            "xi": 0.05,
            "min_cluster_size": 0.1,
        }

        datasets = [
            (
                noisy_circles,
                {
                    "damping": 0.77,
                    "preference": -240,
                    "quantile": 0.2,
                    "n_clusters": 2,
                    "min_samples": 7,
                    "xi": 0.08,
                },
            ),
            (
                noisy_moons,
                {
                    "damping": 0.75,
                    "preference": -220,
                    "n_clusters": 2,
                    "min_samples": 7,
                    "xi": 0.1,
                },
            ),
            (
                varied,
                {
                    "eps": 0.18,
                    "n_neighbors": 2,
                    "min_samples": 7,
                    "xi": 0.01,
                    "min_cluster_size": 0.2,
                },
            ),
            (
                aniso,
                {
                    "eps": 0.15,
                    "n_neighbors": 2,
                    "min_samples": 7,
                    "xi": 0.1,
                    "min_cluster_size": 0.2,
                },
            ),
            (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
            (no_structure, {}),
        ]
        datasets = [i[0] for i in datasets]
        return datasets
    
    def get_colors(self, y):
        colors = np.array(
                list(
                    islice(
                        cycle(
                            [
                                "#377eb8",
                                "#ff7f00",
                                "#4daf4a",
                                "#f781bf",
                                "#a65628",
                                "#984ea3",
                                "#999999",
                                "#e41a1c",
                                "#dede00",
                            ]
                        ),
                        int(max(y) + 1),
                    )
                )
        )

        return colors
    
    def plot_data(self, X, y):
        colors = self.get_colors(y)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
        plt.xticks(())
        plt.yticks(())
        plt.show()
        
    def one_hot_encoding(self, array_y):
        y_one_hot_encoded = np.zeros((array_y.size, array_y.max() + 1))
        y_one_hot_encoded[np.arange(array_y.size), array_y] = 1
        return y_one_hot_encoded
    
    def main(self):
        # Returns all datasets
        datasets = self.generate_data()
        if self.dataset_n == None:
            X = [x[0] for x in datasets]
            y = [y[1] for y in datasets]
            y = self.one_hot_encoding(y)
            return (X, y)

        dataset = datasets[self.dataset_n]
        X, y = dataset[0], dataset[1]
        
        if self.plot == True:
            self.plot_data(X, y)
        
        y = self.one_hot_encoding(y)
        return (X, y)