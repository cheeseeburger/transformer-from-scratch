import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        # d_model = size of embedding (e.g. 512)
        # num_heads = how many attention heads (e.g. 8)
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head
        
        # Linear layers to project Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Core attention formula: softmax(QK^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax gives us attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        return torch.matmul(attention_weights, V), attention_weights
    
    def split_heads(self, x, batch_size):
        # Reshape for multi-head: (batch, seq_len, d_model)
        # -> (batch, num_heads, seq_len, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Project inputs
        Q = self.split_heads(self.W_q(Q), batch_size)
        K = self.split_heads(self.W_k(K), batch_size)
        V = self.split_heads(self.W_v(V), batch_size)
        
        # Apply attention
        x, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads back
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        
        return self.W_o(x)