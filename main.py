import os
import pickle

import base64
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from cycler import cycler
import mpld3

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

filepaths = ["./memes/" + name for name in os.listdir("./memes")]


def read_embeddings(filepath):
    f = open(filepath, "rb")
    embeddings = pickle.load(f)
    f.close()

    embeddings_list = list(embeddings.values())
    embeddings_keys = list(embeddings.keys())

    return embeddings_list, embeddings_keys


def prepare_images(embeddings_keys):
    images = []

    print("PREPARING IMAGES")
    for path in tqdm(embeddings_keys):
        data = open(path, 'rb').read()
        data_base64 = base64.b64encode(data)
        data_base64 = data_base64.decode()
        images.append(data_base64)

    images_html_data = ["<img src=data:image/jpeg;base64," + image + " width='255' height='255'>" for image in images]
    return images_html_data


def run_tsne(embeddings_list):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    np_embeddings_list = np.array(embeddings_list)
    tsne_results = tsne.fit_transform(np_embeddings_list)

    X_s = tsne_results[:, 0]
    Y_s = tsne_results[:, 1]

    tsne_points = []
    for i in range(len(X_s)):
        tsne_points.append([X_s[i], Y_s[i]])

    db = DBSCAN(eps=1.0, min_samples=4).fit(tsne_points)
    labels = db.labels_

    df_tsne_points = pd.DataFrame(tsne_points, columns =["x", "y"])
    print()
    print(df_tsne_points.head())
    return df_tsne_points, labels


def plot_clusters(points, labels, images):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y']))
    scatter = ax.scatter(data = points,x = "x", y = "y", c = labels, marker="o", s=25)
    #tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=embeddings_keys)
    tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels=images)
    mpld3.plugins.connect(fig, tooltip)

    #plt.show()
    mpld3.show()


if __name__ == "__main__":
    embeddings_filepath = "embeddings"
    embeddings, keys = read_embeddings(embeddings_filepath)
    images_html_data = prepare_images(keys)
    df_tsne_data, labels = run_tsne(embeddings)
    plot_clusters(df_tsne_data, labels, images_html_data)
