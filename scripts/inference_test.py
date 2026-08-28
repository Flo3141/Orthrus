import torch
from transformers import AutoModel

# 1. Device auswählen (GPU falls vorhanden, sonst CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Device: {device}")

# 2. Vortrainiertes Modell von Hugging Face laden
# (Varianten: 'quietflamingo/orthrus-base-4-track' oder 'quietflamingo/orthrus-large-4-track')
model_name = "quietflamingo/orthrus-base-4-track"
print(f"Lade Modell '{model_name}'...")
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

# 3. Beispiel-RNA-Sequenz vorbereiten
sequence = "AUGGCCAAUGUGCUCAAGUUCAAGCUCAAGUUC"  # Beliebige Transkriptsequenz

# Sequence to One-Hot Encoding
seq_ohe = model.seq_to_oh(sequence)  # Tensor der Form (Länge, 4)
x = seq_ohe.unsqueeze(0).to(device)  # Batch-Dimension hinzufügen -> (1, Länge, 4)
lengths = torch.tensor([x.shape[1]], device=device)

# 4. Inferenz / Embeddings berechnen
with torch.no_grad():
    # Gesamt-Repräsentation des Transkripts (Pooled Embedding)
    embedding = model.representation(x, lengths, channel_last=True)
    
    # Positionsspezifische Repräsentation (Unpooled)
    unpooled = model(x, channel_last=True)

print("Inferenz erfolgreich!")
print("Pooled Embedding Shape:  ", embedding.shape)  # z.B. (1, 256)
print("Unpooled Embedding Shape:", unpooled.shape)   # z.B. (1, Länge, 256)