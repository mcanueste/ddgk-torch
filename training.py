import collections
import multiprocessing
from typing import Dict, List
import numpy as np
import networkx as nx
import torch
import tqdm
from scipy.spatial import distance
from sklearn import svm
from sklearn.model_selection import ShuffleSplit, GridSearchCV

from model import GraphEncoder, CrossGraphAttention, CrossGraphAttentionLoss
from utils import (
    save_encoder,
    save_ddgk,
    encoder_save_path,
    one_hot_vector,
    get_labels_dict,
)


def encode(
    model_dir: str,
    graph_id: int,
    graph: nx.Graph,
    epochs: int = 10,
    learning_rate: float = 0.01,
    verbose: bool = False,
):
    encoder = GraphEncoder(graph.number_of_nodes())
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    loss_func = torch.nn.BCEWithLogitsLoss()

    x = one_hot_vector(torch.tensor(list(graph.nodes())), graph.number_of_nodes())
    y = torch.tensor(
        nx.adjacency_matrix(graph, weight=None).todense(), dtype=torch.float64
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        x = encoder(x)
        loss = loss_func(x, y)
        optimizer.step()
        if verbose:
            print(f"Encoding graph {graph_id}: Epoch: {epoch} | Loss: {loss.item()}")

    save_encoder(model_dir, graph_id, encoder.state_dict())


def score_target(
    model_dir: str,
    source_id: int,
    source: nx.Graph,
    target_id: int,
    target: nx.Graph,
    node_label_count: int,
    edge_label_count: int,
    epochs: int = 10,
    learning_rate: float = 0.01,
    verbose: bool = False,
):
    source_encoder = GraphEncoder(source.number_of_nodes()).load_encoder(
        encoder_save_path(model_dir, source_id), freeze=True
    )
    cross_graph_attention = CrossGraphAttention(
        source_encoder, source.number_of_nodes(), target.number_of_nodes()
    )
    optimizer = torch.optim.Adam(cross_graph_attention.parameters(), lr=learning_rate)
    loss_func = CrossGraphAttentionLoss(
        node_label_weight=0.5,
        edge_label_weight=0.5,
        labels_dict=get_labels_dict(source, target, node_label_count, edge_label_count),
    )

    target_x = one_hot_vector(
        torch.tensor(list(target.nodes())), target.number_of_nodes()
    )
    target_y = torch.tensor(
        nx.adjacency_matrix(target, weight=None).todense(), dtype=torch.float64
    )

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, attention_probs = cross_graph_attention(target_x)
        loss = loss_func(outputs, target_y, attention_probs)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose:
            print(f"{source_id}-{target_id}: [{epoch}] loss: {loss.item()}")

    save_ddgk(model_dir, source_id, target_id, cross_graph_attention.state_dict())
    return losses[-1]


def encode_source_graphs(
    save_dir: str,
    source_graphs: Dict[int, nx.Graph],
    epochs: int = 10,
    learning_rate: float = 0.01,
    num_threads: int = 8,
    verbose: bool = False,
):
    tqdm.tqdm.write(f"\n\nEncoding {len(source_graphs)} source graphs...\n\n")
    pool = multiprocessing.pool.ThreadPool(num_threads)
    with tqdm.tqdm(total=len(source_graphs)) as progress_bar:

        def encode_and_update_pbar(src_tuple):
            encode(
                save_dir,
                src_tuple[0],  # graph id
                src_tuple[1],  # graph
                epochs=epochs,
                learning_rate=learning_rate,
                verbose=verbose,
            )
            progress_bar.update(1)

        pool.map(encode_and_update_pbar, source_graphs.items())


def score_target_graphs(
    save_dir: str,
    source_graphs: Dict[int, nx.Graph],
    target_graphs: Dict[int, nx.Graph],
    node_label_count: int,
    edge_label_count: int,
    epochs: int = 10,
    learning_rate: float = 0.01,
    num_threads: int = 8,
    verbose: bool = False,
):
    tqdm.tqdm.write(f"\n\nScoring {len(target_graphs)} target graphs...\n\n")
    pool = multiprocessing.pool.ThreadPool(num_threads)
    scores = collections.defaultdict(dict)
    with tqdm.tqdm(total=len(target_graphs) * len(source_graphs)) as progress_bar:
        for tgt_id, tgt_graph in target_graphs.items():

            def score_and_update_pbar(src_tuple):
                scores[tgt_id][src_tuple[0]] = score_target(
                    save_dir,
                    src_tuple[0],  # graph id
                    src_tuple[1],  # graph
                    tgt_id,
                    tgt_graph,
                    node_label_count,
                    edge_label_count,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    verbose=verbose,
                )
                progress_bar.update(1)

            pool.map(score_and_update_pbar, source_graphs.items())
    return scores


def calculate_distances(
    source_ids: List[int],
    target_ids: List[int],
    scores: Dict[int, Dict[int, torch.Tensor]],
):
    scores_np = np.array(
        [
            [scores[target_id][source_id] for source_id in source_ids]
            for target_id in target_ids
        ]
    )
    # pairwise distance in sym matrix form
    return distance.squareform(distance.pdist(scores_np, metric="euclidean"))


def grid_search(dataset, distances: np.ndarray):
    labels = np.array([g.graph["label"] for g_id, g in dataset.items()])
    params = {
        "C": np.logspace(0, 8, 17).tolist(),
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["auto"],
        "max_iter": [-1],
    }
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=8191)

    # clf = GridSearchCV(svm.SVC(), params, cv=cv, iid=False)
    clf = GridSearchCV(svm.SVC(), params, cv=cv)
    clf.fit(distances, labels)
    return clf.best_score_
