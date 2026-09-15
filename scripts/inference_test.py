import torch
from transformers import AutoModel

# 1. Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load pre-trained model from Hugging Face
# (Variants: 'quietflamingo/orthrus-base-4-track' or 'quietflamingo/orthrus-large-4-track')
model_name = "quietflamingo/orthrus-base-4-track"
print(f"Loading model '{model_name}'...")
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

# 3. Prepare example RNA sequence
sequence = "AUGGCCAAUGUGCUCAAGUUCAAGCUCAAGUUC"  # Arbitrary transcript sequence

# Sequence to one-hot encoding
seq_ohe = model.seq_to_oh(sequence)  # Tensor of shape (length, 4)
x = seq_ohe.unsqueeze(0).to(device)  # Add batch dimension -> (1, length, 4)
lengths = torch.tensor([x.shape[1]], device=device)

# 4. Inference / compute embeddings
with torch.no_grad():
    # Overall transcript representation (pooled embedding)
    embedding = model.representation(x, lengths, channel_last=True)
    
    # Position-specific representation (unpooled)
    unpooled = model(x, channel_last=True)

print("Inference successful!")
print("Pooled Embedding Shape:  ", embedding.shape)  # e.g. (1, 256)
print("Unpooled Embedding Shape:", unpooled.shape)   # e.g. (1, length, 256)