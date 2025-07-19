import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# Import ModelConfig if available, otherwise skip
try:
    from configs.model import ModelConfig
except ImportError:
    ModelConfig = None

"""
Multi-gate Mixture-of-Experts (MMoE) Model

This implementation is based on the paper:
    Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
    https://dl.acm.org/doi/10.1145/3219819.3220007

MMoE is designed for multi-task learning scenarios where different tasks can benefit from 
shared representations while maintaining task-specific modeling capabilities.

Key components:
- Expert Networks: Shared expert networks that capture different aspects of the input
- Gate Networks: Task-specific gate networks that determine expert weights for each task
- Tower Networks: Task-specific networks that make final predictions
"""


class ExpertNetwork(nn.Module):
    """Individual expert network in the MMoE architecture."""
    
    def __init__(self, input_dim: int, expert_hidden_dims: List[int], dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in expert_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Expert output of shape (batch_size, expert_output_dim)
        """
        return self.network(x)


class GateNetwork(nn.Module):
    """Gate network that computes expert weights for a specific task."""
    
    def __init__(self, input_dim: int, num_experts: int, gate_hidden_dims: Optional[List[int]] = None):
        super().__init__()
        
        if gate_hidden_dims is None:
            # Simple linear gate
            self.network = nn.Linear(input_dim, num_experts)
        else:
            # Multi-layer gate network
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in gate_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU()
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, num_experts))
            self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Gate weights of shape (batch_size, num_experts)
        """
        gate_logits = self.network(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        return gate_weights


class TowerNetwork(nn.Module):
    """Task-specific tower network for final predictions."""
    
    def __init__(self, input_dim: int, tower_hidden_dims: List[int], output_dim: int = 1, dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in tower_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Task prediction of shape (batch_size, output_dim)
        """
        return self.network(x)


class MMoEModel(nn.Module):
    """
    Multi-gate Mixture-of-Experts Model for multi-task learning.
    
    This model can handle multiple tasks simultaneously by using shared expert networks
    and task-specific gate and tower networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 3,
        num_tasks: int = 2,
        expert_hidden_dims: List[int] = [128, 64],
        gate_hidden_dims: Optional[List[int]] = None,
        tower_hidden_dims: List[int] = [64, 32],
        task_output_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features
            num_experts: Number of expert networks
            num_tasks: Number of tasks
            expert_hidden_dims: Hidden dimensions for expert networks
            gate_hidden_dims: Hidden dimensions for gate networks (None for linear gates)
            tower_hidden_dims: Hidden dimensions for tower networks
            task_output_dims: Output dimensions for each task (default: [1] for all tasks)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_hidden_dims, dropout_rate)
            for _ in range(num_experts)
        ])
        
        # Get expert output dimension
        expert_output_dim = self.experts[0].output_dim
        
        # Create gate networks (one per task)
        self.gates = nn.ModuleList([
            GateNetwork(input_dim, num_experts, gate_hidden_dims)
            for _ in range(num_tasks)
        ])
        
        # Create tower networks (one per task)
        if task_output_dims is None:
            task_output_dims = [1] * num_tasks
        
        self.towers = nn.ModuleList([
            TowerNetwork(expert_output_dim, tower_hidden_dims, task_output_dims[i], dropout_rate)
            for i in range(num_tasks)
        ])
    
    def forward(self, x, return_expert_weights: bool = False):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_expert_weights: Whether to return expert weights for analysis
        
        Returns:
            If return_expert_weights=False:
                List of task predictions, each of shape (batch_size, task_output_dim)
            If return_expert_weights=True:
                Tuple of (task_predictions, expert_weights)
                where expert_weights is a list of tensors of shape (batch_size, num_experts)
        """
        batch_size = x.size(0)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # (batch_size, expert_output_dim)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs: (batch_size, num_experts, expert_output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Compute task-specific mixtures and predictions
        task_predictions = []
        expert_weights_list = []
        
        for task_id in range(self.num_tasks):
            # Compute gate weights for this task
            gate_weights = self.gates[task_id](x)  # (batch_size, num_experts)
            expert_weights_list.append(gate_weights)
            
            # Compute weighted mixture of expert outputs
            # gate_weights: (batch_size, num_experts, 1)
            gate_weights_expanded = gate_weights.unsqueeze(-1)
            
            # Weighted sum: (batch_size, expert_output_dim)
            task_input = torch.sum(expert_outputs * gate_weights_expanded, dim=1)
            
            # Task-specific prediction
            task_pred = self.towers[task_id](task_input)
            task_predictions.append(task_pred)
        
        if return_expert_weights:
            return task_predictions, expert_weights_list
        else:
            return task_predictions


class MMoERecommendationModel(MMoEModel):
    """
    MMoE model specifically designed for recommendation systems.
    
    This variant includes embedding layers for user and item IDs, and is configured
    for typical recommendation tasks like CTR prediction and rating prediction.
    """
    
    def __init__(
        self,
        user_cardinality: int,
        item_cardinality: int,
        embedding_dim: int = 64,
        num_dense_features: int = 0,
        num_experts: int = 3,
        num_tasks: int = 2,
        expert_hidden_dims: List[int] = [128, 64],
        gate_hidden_dims: Optional[List[int]] = None,
        tower_hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.1
    ):
        """
        Args:
            user_cardinality: Number of unique users
            item_cardinality: Number of unique items
            embedding_dim: Dimension of user and item embeddings
            num_dense_features: Number of additional dense features
            num_experts: Number of expert networks
            num_tasks: Number of tasks (e.g., CTR prediction, rating prediction)
            expert_hidden_dims: Hidden dimensions for expert networks
            gate_hidden_dims: Hidden dimensions for gate networks
            tower_hidden_dims: Hidden dimensions for tower networks
            dropout_rate: Dropout rate for regularization
        """
        
        # Calculate input dimension
        input_dim = embedding_dim * 2 + num_dense_features  # user + item + dense features
        
        super().__init__(
            input_dim=input_dim,
            num_experts=num_experts,
            num_tasks=num_tasks,
            expert_hidden_dims=expert_hidden_dims,
            gate_hidden_dims=gate_hidden_dims,
            tower_hidden_dims=tower_hidden_dims,
            task_output_dims=[1] * num_tasks,  # Assume single output per task
            dropout_rate=dropout_rate
        )
        
        # Embedding layers
        self.user_embedding = nn.Embedding(user_cardinality, embedding_dim)
        self.item_embedding = nn.Embedding(item_cardinality, embedding_dim)
        
        self.embedding_dim = embedding_dim
        self.num_dense_features = num_dense_features
    
    def forward(self, user_ids, item_ids, dense_features=None, return_expert_weights=False):
        """
        Args:
            user_ids: User IDs tensor of shape (batch_size,) or (batch_size, 1)
            item_ids: Item IDs tensor of shape (batch_size,) or (batch_size, 1)
            dense_features: Dense features tensor of shape (batch_size, num_dense_features)
            return_expert_weights: Whether to return expert weights for analysis
        
        Returns:
            List of task predictions or tuple with expert weights
        """
        # Handle input shapes
        if user_ids.dim() > 1:
            user_ids = user_ids.squeeze(-1)
        if item_ids.dim() > 1:
            item_ids = item_ids.squeeze(-1)
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item_ids)  # (batch_size, embedding_dim)
        
        # Concatenate embeddings
        features = torch.cat([user_emb, item_emb], dim=-1)
        
        # Add dense features if provided, or create zero tensor if expected but not provided
        if dense_features is not None:
            features = torch.cat([features, dense_features], dim=-1)
        elif self.num_dense_features > 0:
            # Create zero tensor for missing dense features to maintain dimension consistency
            batch_size = user_emb.size(0)
            zero_dense_features = torch.zeros(batch_size, self.num_dense_features, 
                                            device=user_emb.device, dtype=user_emb.dtype)
            features = torch.cat([features, zero_dense_features], dim=-1)
        
        # Forward through MMoE
        return super().forward(features, return_expert_weights)


if __name__ == "__main__":
    # Test basic MMoE model
    print("Testing basic MMoE model...")
    model = MMoEModel(
        input_dim=128,
        num_experts=3,
        num_tasks=2,
        expert_hidden_dims=[64, 32],
        tower_hidden_dims=[32, 16]
    )
    
    batch_size = 32
    x = torch.randn(batch_size, 128)
    
    # Test forward pass
    predictions = model(x)
    print(f"Number of task predictions: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"Task {i} prediction shape: {pred.shape}")
    
    # Test with expert weights
    predictions, expert_weights = model(x, return_expert_weights=True)
    print(f"Expert weights shapes: {[w.shape for w in expert_weights]}")
    
    print("\nTesting MMoE recommendation model...")
    # Test recommendation model
    rec_model = MMoERecommendationModel(
        user_cardinality=1000,
        item_cardinality=5000,
        embedding_dim=32,
        num_dense_features=8,
        num_experts=4,
        num_tasks=2
    )
    
    user_ids = torch.randint(0, 1000, (batch_size,))
    item_ids = torch.randint(0, 5000, (batch_size,))
    dense_features = torch.randn(batch_size, 8)
    
    predictions = rec_model(user_ids, item_ids, dense_features)
    print(f"Recommendation model predictions: {[p.shape for p in predictions]}")
    
    print("All tests passed!") 