import torch
import torch.nn as nn
import math

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_query = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.attention_scores = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Linear projections
        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax and compute attention
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention = torch.matmul(attention_weights, value)

        # Reshape and apply final linear layer
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.attention_scores(attention)

def train_masked_attention():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 32
    seq_len = 8
    embed_dim = 64
    n_heads = 4
    n_epochs = 100
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, embed_dim)
    target = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create a causal mask (lower triangular matrix)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = ~mask  # Invert to get lower triangular
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    # Initialize model and optimizer
    model = MaskedMultiHeadAttention(embed_dim, n_heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x, x, x, mask)  # Using same input for query, key, value
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_output = model(x, x, x, mask)
        test_loss = criterion(test_output, target)
        print(f"\nTest Loss: {test_loss.item():.4f}")
    
    return model, test_loss.item()

if __name__ == "__main__":
    model, final_loss = train_masked_attention() 