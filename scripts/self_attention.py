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

def main():
    # Set the batch size and number of data points
    batch_size = 2
    num_points = 1000

    # Generate the batch indices
    indices = torch.randperm(num_points)
    # batches = indices.split(batch_size)

    # Generate random data points
    data = torch.randn(num_points, 3)    
    # print(data)
    self_attention = SelfAttention(input_dim=3, attention_dim=128)
    weighted_sum, attention_weights = self_attention(data)
    print(weighted_sum, attention_weights)
if __name__ == '__main__':
    main()