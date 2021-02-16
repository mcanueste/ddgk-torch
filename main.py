import os
import random

from dataset import MUTAGDataset
from training import (
    encode_source_graphs,
    score_target_graphs,
    calculate_distances,
    grid_search,
)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKING_DIR, "dataset")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
NUM_SOURCES = 16
NUM_THREADS = 8
SCORE_WINDOW = 10  # The window to average for scoring loss and accuracy calculation.

if __name__ == "__main__":
    dataset = MUTAGDataset(download_path=DATASET_DIR, download=True, overwrite=True)
    node_label_count = len(dataset.NodeLabels)
    edge_label_count = len(dataset.EdgeLabels)
    source_graphs = dict(random.sample(dataset.items(), NUM_SOURCES))

    encode_source_graphs(MODEL_DIR, source_graphs, epochs=600)
    scores = score_target_graphs(
        MODEL_DIR,
        source_graphs,
        dataset.graphs,
        node_label_count,
        edge_label_count,
        epochs=600,
    )
    dist_mat = calculate_distances(
        list(source_graphs.keys()), list(dataset.graphs.keys()), scores
    )

    print("Performing GridSearchCV w/ DDGK...")
    cv_score = grid_search(dataset, dist_mat)
    print("10-fold CV score: {:.4f}.".format(cv_score))
