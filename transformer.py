import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads,
            d_ff, num_layers, max_seq_len, dropout
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads,
            d_ff, num_layers, max_seq_len, dropout
        )
        
        # Final linear layer maps to vocab size
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
    
    def make_src_mask(self, src, pad_idx=0):
        # Mask padding tokens in source
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt, pad_idx=0):
        batch_size, tgt_len = tgt.size()
        
        # Mask padding
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Causal mask — prevents looking at future tokens
        tgt_causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len)
        ).bool().to(tgt.device)
        
        # Combine both masks
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        return tgt_mask
    
    def forward(self, src, tgt):
        # Build masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode using encoder output
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocab size
        output = self.output_layer(decoder_output)
        
        return output


# Quick sanity test — run this file directly to check everything works
if __name__ == "__main__":
    src_vocab = 1000
    tgt_vocab = 1000
    
    model = Transformer(
    src_vocab_size=5000,  # change for your task
    tgt_vocab_size=5000,
    d_model=256,          # bigger = more powerful but slower
    num_heads=8,
    num_layers=4,         # more layers = deeper understanding
    d_ff=1024,
    max_seq_len=100
 )
    
    # Dummy input
    src = torch.randint(1, src_vocab, (2, 10))  # batch=2, seq_len=10
    tgt = torch.randint(1, tgt_vocab, (2, 10))
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # Should be (2, 10, 1000)
    print("Transformer working correctly!")