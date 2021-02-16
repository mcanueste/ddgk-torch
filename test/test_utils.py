import os

import torch

from utils import (
    encoder_save_path,
    attention_save_path,
    one_hot_vector,
    revert_one_hot_vector,
)


def test_encoder_save_path():
    save_dir = "./models"
    assert encoder_save_path(save_dir, 1) == os.path.join(save_dir, "encoders/1")


def test_attention_save_path():
    save_dir = "./models"
    assert attention_save_path(save_dir, 1, 2) == os.path.join(save_dir, "ddgk/1-2")


def test_one_hot_vector():
    labels = torch.tensor([0, 1, 3, 2, 3, 0, 0, 1])
    one_hot_labels = one_hot_vector(labels, 4)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(expected, one_hot_labels)


def test_revert_one_hot_vector():
    one_hot_labels = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    labels = revert_one_hot_vector(one_hot_labels)
    expected = torch.tensor([0, 1, 3, 2, 3, 0, 0, 1])
    assert torch.equal(expected, labels)
