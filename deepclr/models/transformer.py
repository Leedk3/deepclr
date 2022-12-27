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
        
        return weighted_sum #weighted_sum, attention_weights

class Transformer3D(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout):
        super().__init__()
        
        # Create the transformer layers
        self.transformer = nn.Transformer(num_layers, d_model, num_heads, dff, dropout)
        
    def forward(self, x):
        # x is a batch of 3D point clouds, represented as a sequence of NxC matrices
        
        # Transpose the batch dimensions and the sequence dimensions to prepare for the transformer
        x = x.transpose(0, 1)
        
        # Apply the transformer
        output = self.transformer(x)
        
        # Transpose the output back to the original shape
        output = output.transpose(0, 1)
        
        return output

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
        
        # Linear layer to project the output of the self-attention module
        self.projection = nn.Linear(input_dim, input_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, inputs):
        # Apply self-attention
        attention_output = self.self_attention(inputs)
        print("inputs : ", inputs.shape)
        print("attention_output : ", attention_output.shape)

        # Project the output of the self-attention module
        output = self.projection(attention_output)
        print("output : ", output.shape)

        # Add the projected output to the input and apply layer normalization
        output = self.layer_norm(inputs + output)
        print("output : ", output.shape)

        
        return output