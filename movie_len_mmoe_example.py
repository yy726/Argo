"""
MovieLens MMoE Model Example

This example demonstrates how to use the Multi-gate Mixture-of-Experts (MMoE) model
for multi-task learning on the MovieLens dataset. We'll predict both:
1. Click-through rate (CTR) - binary classification
2. Rating prediction - regression

Usage:
    python movie_len_mmoe_example.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model.mmoe import MMoERecommendationModel
from data.dataset import prepare_movie_len_dataset
from configs.model import MOVIE_LEN_ITEM_CARDINALITY


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function that combines losses from different tasks.
    Uses uncertainty weighting to balance different task losses.
    """
    
    def __init__(self, num_tasks: int):
        super().__init__()
        # Learnable task-specific weights (log variance)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of task predictions
            targets: List of task targets
        
        Returns:
            Combined loss and individual task losses
        """
        losses = []
        
        # Task 0: CTR prediction (binary classification)
        ctr_loss = nn.BCEWithLogitsLoss()(predictions[0].squeeze(), targets[0].float())
        precision_0 = torch.exp(-self.log_vars[0])
        weighted_loss_0 = precision_0 * ctr_loss + self.log_vars[0]
        losses.append(ctr_loss)
        
        # Task 1: Rating prediction (regression)
        rating_loss = nn.MSELoss()(predictions[1].squeeze(), targets[1].float())
        precision_1 = torch.exp(-self.log_vars[1])
        weighted_loss_1 = precision_1 * rating_loss + self.log_vars[1]
        losses.append(rating_loss)
        
        # Combined loss
        total_loss = weighted_loss_0 + weighted_loss_1
        
        return total_loss, losses


def create_multi_task_targets(labels):
    """
    Convert single labels to multi-task targets:
    - CTR: 1 if rating >= 4, 0 otherwise
    - Rating: original rating scaled to [0, 1]
    """
    labels = labels.numpy()
    
    # CTR targets (binary)
    ctr_targets = (labels >= 0.8).astype(np.float32)  # Assuming labels are already normalized
    
    # Rating targets (continuous, already normalized in dataset)
    rating_targets = labels.astype(np.float32)
    
    return torch.tensor(ctr_targets), torch.tensor(rating_targets)


def train_mmoe_model():
    """Train MMoE model on MovieLens dataset."""
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 5
    embedding_dim = 64
    num_experts = 4
    
    print("Preparing MovieLens dataset...")
    train_dataset, eval_dataset, unique_ids = prepare_movie_len_dataset(
        history_seq_length=6, 
        eval_ratio=0.2
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Get cardinalities
    num_users = train_dataset.data['user_id'].max() + 1
    num_items = MOVIE_LEN_ITEM_CARDINALITY
    
    print(f"Number of users: {num_users}, Number of items: {num_items}")
    
    # Initialize MMoE model
    model = MMoERecommendationModel(
        user_cardinality=num_users,
        item_cardinality=num_items,
        embedding_dim=embedding_dim,
        num_dense_features=8,  # From the dataset
        num_experts=num_experts,
        num_tasks=2,  # CTR and Rating prediction
        expert_hidden_dims=[128, 64],
        tower_hidden_dims=[64, 32],
        dropout_rate=0.1
    )
    
    # Multi-task loss and optimizer
    criterion = MultiTaskLoss(num_tasks=2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        ctr_losses = 0
        rating_losses = 0
        num_batches = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Extract features
            user_ids = features['user_id'].squeeze()
            item_ids = features['item_id'].squeeze()
            dense_features = features['dense_features']
            
            # Create multi-task targets
            ctr_targets, rating_targets = create_multi_task_targets(labels)
            targets = [ctr_targets, rating_targets]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids, dense_features)
            
            # Compute loss
            loss, individual_losses = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            ctr_losses += individual_losses[0].item()
            rating_losses += individual_losses[1].item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, "
                      f"CTR Loss: {individual_losses[0].item():.4f}, "
                      f"Rating Loss: {individual_losses[1].item():.4f}")
        
        scheduler.step()
        
        # Epoch statistics
        avg_loss = total_loss / num_batches
        avg_ctr_loss = ctr_losses / num_batches
        avg_rating_loss = rating_losses / num_batches
        
        print(f"Epoch {epoch+1} completed - "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Avg CTR Loss: {avg_ctr_loss:.4f}, "
              f"Avg Rating Loss: {avg_rating_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % 2 == 0:
            evaluate_model(model, eval_loader)
    
    return model


def evaluate_model(model, eval_loader):
    """Evaluate the MMoE model."""
    model.eval()
    
    ctr_correct = 0
    ctr_total = 0
    rating_mse = 0
    num_samples = 0
    
    with torch.no_grad():
        for features, labels in eval_loader:
            # Extract features
            user_ids = features['user_id'].squeeze()
            item_ids = features['item_id'].squeeze()
            dense_features = features['dense_features']
            
            # Create multi-task targets
            ctr_targets, rating_targets = create_multi_task_targets(labels)
            
            # Forward pass
            predictions = model(user_ids, item_ids, dense_features)
            
            # CTR evaluation (accuracy)
            ctr_probs = torch.sigmoid(predictions[0].squeeze())
            ctr_preds = (ctr_probs > 0.5).float()
            ctr_correct += (ctr_preds == ctr_targets).sum().item()
            ctr_total += len(ctr_targets)
            
            # Rating evaluation (MSE)
            rating_preds = predictions[1].squeeze()
            rating_mse += ((rating_preds - rating_targets) ** 2).sum().item()
            num_samples += len(rating_targets)
    
    ctr_accuracy = ctr_correct / ctr_total
    rating_rmse = np.sqrt(rating_mse / num_samples)
    
    print(f"Evaluation - CTR Accuracy: {ctr_accuracy:.4f}, Rating RMSE: {rating_rmse:.4f}")
    
    model.train()


def analyze_expert_weights(model, eval_loader):
    """Analyze expert utilization across different tasks."""
    model.eval()
    
    task_expert_weights = [[] for _ in range(2)]
    
    with torch.no_grad():
        for features, _ in eval_loader:
            user_ids = features['user_id'].squeeze()
            item_ids = features['item_id'].squeeze()
            dense_features = features['dense_features']
            
            # Get expert weights
            _, expert_weights = model(user_ids, item_ids, dense_features, return_expert_weights=True)
            
            for task_id, weights in enumerate(expert_weights):
                task_expert_weights[task_id].append(weights.cpu())
    
    # Analyze expert utilization
    for task_id in range(2):
        all_weights = torch.cat(task_expert_weights[task_id], dim=0)
        avg_weights = all_weights.mean(dim=0)
        
        task_name = "CTR" if task_id == 0 else "Rating"
        print(f"\n{task_name} Task - Average Expert Weights:")
        for expert_id, weight in enumerate(avg_weights):
            print(f"  Expert {expert_id}: {weight:.4f}")
    
    model.train()


if __name__ == "__main__":
    print("🚀 Starting MMoE training on MovieLens dataset...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Train the model
        trained_model = train_mmoe_model()
        
        print("\n📊 Analyzing expert weights...")
        # Reload eval dataset for analysis
        _, eval_dataset, _ = prepare_movie_len_dataset(history_seq_length=6, eval_ratio=0.2)
        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
        
        analyze_expert_weights(trained_model, eval_loader)
        
        print("\n✅ MMoE training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise 