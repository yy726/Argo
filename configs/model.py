from dataclasses import dataclass


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
