import torch
import torch.nn as nn
from transformer import Transformer

# Simple task: learn to reverse a sequence
# Input:  [1, 2, 3, 4, 5]
# Output: [5, 4, 3, 2, 1]

VOCAB_SIZE = 20
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 500
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512

def generate_batch(batch_size, seq_len, vocab_size):
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = torch.flip(src, dims=[1])  # reversed = target
    return src, tgt

model = Transformer(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_seq_len=SEQ_LEN
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    src, tgt = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    # Teacher forcing: feed target shifted right
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    output = model(src, tgt_input)

    # Reshape for loss
    output = output.reshape(-1, VOCAB_SIZE)
    tgt_output = tgt_output.reshape(-1)

    loss = criterion(output, tgt_output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

print("Training complete!")
torch.save(model.state_dict(), "transformer_model.pth")
print("Model saved!") 
