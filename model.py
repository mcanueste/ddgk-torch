from typing import Union, Dict

import networkx as nx
import torch
import torch.nn as nn

from utils import (
    get_node_labels,
    get_neighbor_node_labels,
    get_edge_labels,
    get_neighbor_edge_labels,
    one_hot_vector,
    get_logits,
    convert_to_probs,
    revert_one_hot_vector,
)


class LinearTanh(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearTanh, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_size: int = 4,
        expand: int = 4,
        num_dnn_layers: int = 4,
    ):
        super(GraphEncoder, self).__init__()
        expanded_embedding = embedding_size * expand
        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=num_nodes, out_features=embedding_size, bias=False
                ),
                LinearTanh(in_features=embedding_size, out_features=expanded_embedding),
            ]
        )
        for _ in range(max(num_dnn_layers - 1, 1)):
            self.layers.append(
                LinearTanh(
                    in_features=expanded_embedding, out_features=expanded_embedding
                )
            )
        self.layers.append(
            LinearTanh(in_features=expanded_embedding, out_features=num_nodes)
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def load_encoder(self, path: str, freeze: bool = True):
        self.load_state_dict(torch.load(path))
        self.eval()
        if freeze:
            for param in self.parameters():  # Freeze source graph
                param.requires_grad = False
        return self


class Attention(nn.Module):
    def __init__(self, source_num_nodes, target_num_nodes):
        super(Attention, self).__init__()
        self.linear = nn.Linear(
            in_features=target_num_nodes, out_features=source_num_nodes, bias=False
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class ReverseAttention(nn.Module):
    def __init__(self, source_num_nodes, target_num_nodes):
        super(ReverseAttention, self).__init__()
        self.linear = nn.Linear(
            in_features=source_num_nodes, out_features=target_num_nodes, bias=False
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class CrossGraphAttention(nn.Module):
    def __init__(self, source_encoder, source_num_nodes, target_num_nodes):
        super(CrossGraphAttention, self).__init__()
        self.layers = nn.ModuleList()
        self.attention = Attention(source_num_nodes, target_num_nodes)
        self.source_encoder = source_encoder
        self.reverse_attention = ReverseAttention(source_num_nodes, target_num_nodes)

    def forward(self, x):
        x = self.attention(x)
        attention_prob = x.detach().clone()
        x = self.source_encoder(x)
        x = self.reverse_attention(x)
        return x, attention_prob


class CrossGraphAttentionLoss:
    def __init__(
        self,
        node_label_weight: float = 0.0,
        edge_label_weight: float = 0.0,
        labels_dict: Union[None, Dict[str, torch.Tensor]] = None,
    ):
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.node_label_weight = node_label_weight
        self.edge_label_weight = edge_label_weight
        self.labels_dict = labels_dict

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_probs: Union[torch.Tensor, None] = None,
        *args,
        **kwargs
    ):
        loss = self.bce(x, y)
        if self.node_label_weight:
            node_loss = self.node_label_loss(attention_probs)
            node_loss += self.neighbor_node_label_loss(x)
            loss += self.node_label_weight * node_loss
        if self.edge_label_weight:
            edge_loss = self.edge_label_loss(attention_probs)
            edge_loss += self.neighbor_edge_label_loss(x)
            loss += self.edge_label_weight * edge_loss
        return loss

    def node_label_loss(self, attention_probs: torch.Tensor):
        logits = get_logits(attention_probs, self.labels_dict["source_node_labels"])
        loss = self.ce(
            logits, revert_one_hot_vector(self.labels_dict["target_node_labels"])
        )
        return loss

    def neighbor_node_label_loss(self, output: torch.Tensor):
        output_sig = torch.sigmoid(output)
        probs = convert_to_probs(output_sig)
        logits = get_logits(probs, self.labels_dict["target_node_labels"])
        loss = self.ce(
            logits,
            revert_one_hot_vector(self.labels_dict["target_neighbor_node_labels"]),
        )
        return loss

    def edge_label_loss(self, attention_probs: torch.Tensor):
        logits = get_logits(attention_probs, self.labels_dict["source_edge_labels"])
        loss = self.ce(
            logits, revert_one_hot_vector(self.labels_dict["target_edge_labels"])
        )
        return loss

    def neighbor_edge_label_loss(self, output: torch.Tensor):
        output_sig = torch.sigmoid(output)
        probs = convert_to_probs(output_sig)
        logits = get_logits(probs, self.labels_dict["target_edge_labels"])
        loss = self.ce(
            logits,
            revert_one_hot_vector(self.labels_dict["target_neighbor_edge_labels"]),
        )
        return loss
