import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Linear layers to compute the queries, keys, and values
        self.query_layer = nn.Linear(input_dim, attention_dim)
        self.key_layer = nn.Linear(input_dim, attention_dim)
        self.value_layer = nn.Linear(input_dim, attention_dim)
        
    def forward(self, inputs):
        # Shape of inputs: [batch_size, sequence_length, input_dim]
        
        # Compute the queries, keys, and values
        queries = self.query_layer(inputs)  # [batch_size, sequence_length, attention_dim]
        keys = self.key_layer(inputs)  # [batch_size, sequence_length, attention_dim]
        values = self.value_layer(inputs)  # [batch_size, sequence_length, attention_dim]
        
        # Compute the dot product of the queries and keys, and normalize it
        dot_product = torch.matmul(queries, keys.transpose(1, 2))  # [batch_size, sequence_length, sequence_length]
        dot_product = dot_product / (self.attention_dim ** 0.5)  # Normalize by the square root of the attention dim
        
        # Apply the softmax function to the dot product to get the attention weights
        attention_weights = nn.functional.softmax(dot_product, dim=-1)  # [batch_size, sequence_length, sequence_length]
        
        # Compute the weighted sum of the values using the attention weights
        weighted_sum = torch.matmul(attention_weights, values)  # [batch_size, sequence_length, attention_dim]
        
        return weighted_sum, attention_weights

class Transformer(nn.Module):
    def __init__(self, input_dim, attention_dim, num_layers, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Create a list of layers
        self.layers = nn.ModuleList([
            TransformerLayer(input_dim, attention_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Linear layer to project the output to the desired output dimension
        self.output_projection = nn.Linear(input_dim, input_dim)
        
    def forward(self, inputs):
        # Shape of inputs: [batch_size, sequence_length, input_dim]
        
        # Apply the layers
        for layer in self.layers:
            inputs = layer(inputs)
        
        # Project the output to the desired output dimension
        outputs = self.output_projection(inputs)
        
        return outputs
    
class TransformerLayer(nn.Module):
    def __init__(self, input_dim, attention_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Self-attention module
        self.self_attention = SelfAttention(input_dim, attention_dim)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, 4 * attention_dim),
            nn.ReLU(),
            nn.Linear(4 * attention_dim, input_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm

