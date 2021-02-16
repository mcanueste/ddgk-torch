import collections
import os
from enum import Enum
from io import BytesIO
from typing import Dict, List, Callable, Type, Union, Tuple, Iterator, ItemsView
from urllib.request import urlopen
from zipfile import ZipFile

import networkx as nx
from torch.utils.data import Dataset


def download_mutag_dataset(
    download_dir: str,
    overwrite: bool = False,
    url: str = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip",
) -> str:
    if not overwrite and os.path.isdir(os.path.join(download_dir, "MUTAG")):
        raise FileExistsError(
            "Dataset already exists on the defined path! "
            "Set 'overwrite=True' if you wish to overwrite "
            "the existing data."
        )
    with urlopen(url) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(download_dir)

    return os.path.join(download_dir, "MUTAG")


def load_graph_labels(mutag_path: str) -> Dict[int, int]:
    file_path = os.path.join(mutag_path, "MUTAG_graph_labels.txt")
    file_mapping = collections.defaultdict(int)
    with open(file_path, "r") as f:
        for graph_id, label_str in enumerate(f, 1):
            file_mapping[graph_id] = int(label_str)
    return file_mapping


def load_graph_node_mapping(mutag_path: str) -> Dict[int, List[int]]:
    file_path = os.path.join(mutag_path, "MUTAG_graph_indicator.txt")
    file_mapping = collections.defaultdict(list)
    with open(file_path, "r") as f:
        for node_id, graph_id_str in enumerate(f, 1):
            file_mapping[int(graph_id_str)].append(node_id)
    return file_mapping


def load_nodes(mutag_path: str) -> Dict[int, int]:
    file_path = os.path.join(mutag_path, "MUTAG_node_labels.txt")
    file_mapping = collections.defaultdict(int)
    with open(file_path, "r") as f:
        for node_id, node_label_str in enumerate(f, 1):
            file_mapping[int(node_id)] = int(node_label_str)
    return file_mapping


def load_edges(mutag_path: str) -> Dict[int, List[int]]:
    file_path = os.path.join(mutag_path, "MUTAG_A.txt")
    file_mapping = collections.defaultdict(list)
    with open(file_path, "r") as f:
        for edge_id, edge_tup in enumerate(f, 1):
            file_mapping[edge_id] = [int(v) for v in edge_tup.split(",")]
    return file_mapping


def load_edge_labels(mutag_path: str) -> Dict[int, int]:
    file_path = os.path.join(mutag_path, "MUTAG_edge_labels.txt")
    file_mapping = collections.defaultdict(int)
    with open(file_path, "r") as f:
        for edge_id, edge_label_str in enumerate(f, 1):
            file_mapping[edge_id] = int(edge_label_str)
    return file_mapping


def load_mutag_supergraph(mutag_path: str) -> nx.Graph:
    nodes = load_nodes(mutag_path)
    edges = load_edges(mutag_path)
    edge_labels = load_edge_labels(mutag_path)
    g = nx.Graph()
    for node_id, node_label in nodes.items():
        g.add_node(node_id, label=node_label)
    for edge_id, edge_tuple in edges.items():
        g.add_edge(*edge_tuple, label=edge_labels[edge_id])
    return g


def load_mutag_graphs(mutag_path: str) -> Dict[int, nx.Graph]:
    graph_labels = load_graph_labels(mutag_path)
    graph_nodes = load_graph_node_mapping(mutag_path)
    supergraph = load_mutag_supergraph(mutag_path)
    return {
        graph_id
        - 1: nx.convert_node_labels_to_integers(
            nx.Graph(supergraph.subgraph(graph_nodes[graph_id]), label=graph_label),
            first_label=0,
        )
        for graph_id, graph_label in graph_labels.items()
    }


class DatasetException(Exception):
    pass


class MUTAGDataset(Dataset):
    """
    The MUTAG dataset consists of 188 chemical compounds divided into two
    classes according to their mutagenic effect on a bacterium.

    The chemical data was obtained form http://cdb.ics.uci.edu and converted
    to graphs, where vertices represent atoms and edges represent chemical
    bonds. Explicit hydrogen atoms have been removed and vertices are labeled
    by atom type and edges by bond type (single, double, triple or aromatic).
    Chemical data was processed using the Chemistry Development Kit (v1.4).
    """

    class NodeLabels(Enum):
        C = 0
        N = 1
        O = 2
        F = 3
        I = 4
        CL = 5
        BR = 6

    class EdgeLabels(Enum):
        AROMATIC = 0
        SINGLE = 1
        DOUBLE = 2
        TRIPLE = 3

    def __init__(
        self,
        download_path: str,
        download: bool = False,
        overwrite: bool = False,
        dataset_url: str = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip",
    ):
        self.download_path = download_path
        self.download = download
        self.overwrite = overwrite
        self.dataset_url = dataset_url
        self.mutag_dir = download_mutag_dataset(
            self.download_path, self.overwrite, self.dataset_url
        )
        self.graphs = load_mutag_graphs(self.mutag_dir)

    def __len__(self) -> int:
        return len(self.graphs.items())

    def __getitem__(self, idx) -> nx.Graph:
        return self.graphs[idx]

    def __iter__(self) -> Iterator[nx.Graph]:
        return iter(self.graphs.values())

    def items(self) -> ItemsView[int, nx.Graph]:
        return self.graphs.items()
