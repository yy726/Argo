from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ModelConfig:
    user_cardinality: int
    item_cardinality: int
    user_embedding_dim: int
    item_embedding_dim: int
    num_dense_features: int


DIN_SMALL_CONFIG = ModelConfig(
    user_cardinality=2000,
    item_cardinality=200000,
    user_embedding_dim=8,
    item_embedding_dim=8,  # for now this is not customizable yet
    num_dense_features=8,
)


@dataclass
class DCNv2Config:
    feature_config: Dict[str, Tuple[int, int]]  # a dict contains the cardinality and embedding dim of each feature
    num_cross_layers: int  # number of layers in cross net
    deep_net_hidden_dims: List[int]  # hidden dimensions of linear layers in deep net
    head_hidden_dim: int  # hidden dimension of prediction head
