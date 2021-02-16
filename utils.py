import os

import networkx as nx
import torch
from torch import Tensor


def encoder_save_path(save_dir: str, graph_id: int):
    return os.path.join(save_dir, f"encoders/{graph_id}")


def attention_save_path(save_dir: str, source_id: int, target_id: int):
    return os.path.join(save_dir, f"ddgk/{source_id}-{target_id}")


def save_encoder(save_dir: str, encoder_id: int, encoder_params):
    os.makedirs(os.path.join(save_dir, "encoders"), exist_ok=True)
    torch.save(encoder_params, encoder_save_path(save_dir, encoder_id))


def save_ddgk(save_dir: str, source_id: int, target_id: int, ddgk_params):
    os.makedirs(os.path.join(save_dir, "ddgk"), exist_ok=True)
    torch.save(ddgk_params, attention_save_path(save_dir, source_id, target_id))


def one_hot_vector(labels: Tensor, num_labels: int):
    one_hot = torch.zeros(labels.size()[0], num_labels)
    one_hot[torch.arange(labels.size()[0]), labels] = 1.0
    return one_hot


def revert_one_hot_vector(one_hot_labels: Tensor):
    return torch.argmax(one_hot_labels, dim=1)


def convert_to_probs(weights: torch.Tensor):
    sum = torch.sum(weights, dim=1, keepdim=True)
    clamped_sum = torch.clamp(sum, 1e-9, 1e9)  # for numerical stability
    return weights / clamped_sum


def get_node_labels(graph: nx.Graph, num_labels: int):
    labels = torch.tensor([graph.nodes[n]["label"] for n in graph.nodes()])
    one_hot_labels = one_hot_vector(labels, num_labels)
    return convert_to_probs(one_hot_labels)


def get_neighbor_node_labels(graph: nx.Graph, num_labels: int):
    neighbors_labels = torch.zeros((graph.number_of_nodes(), num_labels))

    for v in graph.nodes():
        for u in graph.neighbors(v):
            neighbors_labels[v, graph.nodes[u]["label"]] += 1.0

    return convert_to_probs(neighbors_labels)


def get_edge_labels(graph: nx.Graph, num_labels: int):
    labels = torch.zeros((graph.number_of_nodes(), num_labels))

    for i, n in enumerate(graph.nodes()):
        for u, v in graph.edges(n):
            labels[i, graph[u][v]["label"]] += 1.0

    return convert_to_probs(labels)


def get_neighbor_edge_labels(graph: nx.Graph, num_labels: int):
    labels = torch.zeros((graph.number_of_nodes(), num_labels))

    for i, v in enumerate(graph.nodes()):
        for u in graph.neighbors(v):
            for v1, v2, d in graph.edges(u, data=True):
                if v not in (v1, v2):
                    labels[i, d["label"]] += 1.0

    return convert_to_probs(labels)


def get_labels_dict(
    source: nx.Graph, target: nx.Graph, node_label_count: int, edge_label_count: int
):
    return {
        "source_node_labels": get_node_labels(source, node_label_count),
        "target_node_labels": get_node_labels(target, node_label_count),
        "source_neighbor_node_labels": get_neighbor_node_labels(
            source, node_label_count
        ),
        "target_neighbor_node_labels": get_neighbor_node_labels(
            target, node_label_count
        ),
        "source_edge_labels": get_edge_labels(source, edge_label_count),
        "target_edge_labels": get_edge_labels(target, edge_label_count),
        "source_neighbor_edge_labels": get_neighbor_edge_labels(
            source, edge_label_count
        ),
        "target_neighbor_edge_labels": get_neighbor_edge_labels(
            target, edge_label_count
        ),
    }


def get_logits(probs: torch.Tensor, labels: torch.Tensor):
    # We take the log because the result of the multiplication is already a probability.
    prob = torch.matmul(probs, labels)
    prob = torch.clamp(prob, 1e-12, 1.0)
    logits = torch.log(prob)
    return logits
