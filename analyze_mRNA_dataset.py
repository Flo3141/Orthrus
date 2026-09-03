import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mrna_bench as mb
df_human = mb.load_dataset("rnahl-human").data_df
df_mouse = mb.load_dataset("rnahl-mouse").data_df


print("=== HUMAN DATASET INFO ===")
print(df_human.info())
print("\nErste Zeilen:")
print(df_human[['gene', 'chromosome', 'target']].head())

print("\n=== MOUSE DATASET INFO ===")
print(df_mouse.info())


def analyze_genes(df, name="Dataset"):
    total_transcripts = len(df)
    unique_genes = df['gene'].nunique()
    transcripts_per_gene = df['gene'].value_counts()
    
    print(f"\n--- Gen-Statistiken für {name} ---")
    print(f"Gesamtanzahl Transkripte / Einträge: {total_transcripts}")
    print(f"Einzigartige Gene: {unique_genes}")
    print(f"Transkripte pro Gen (Mittelwert): {total_transcripts / unique_genes:.2f}")
    print(f"Gene mit den meisten Isoformen/Einträgen:\n{transcripts_per_gene.head(5)}")

analyze_genes(df_human, "Human")
analyze_genes(df_mouse, "Mouse")

# Gemeinsame Gennamen (Orthologen-Proxy bei identischem Symbol)
shared_gene_symbols = set(df_human['gene'].str.upper()).intersection(set(df_mouse['gene'].str.upper()))
print(f"\nÜberlappende Gensymbole (Human vs. Mouse): {len(shared_gene_symbols)}")


# Verteilung der Target-Variable (PC1 Halbwertszeit)
print("\n=== VERTEILUNG DER TARGETS (PC1 Half-life) ===")
stats_target = pd.DataFrame({
    'Human': df_human['target'].describe(),
    'Mouse': df_mouse['target'].describe()
})
print(stats_target)


# Verteilung der Chromosomen
print("\n=== CHROMOSOMEN-VERTEILUNG (Top 10) ===")
print("Human:")
print(df_human['chromosome'].value_counts().head(10))
print("\nMouse:")
print(df_mouse['chromosome'].value_counts().head(10))


# Analyse der Sequenzlängen
df_human['seq_len'] = df_human['sequence'].str.len()
df_mouse['seq_len'] = df_mouse['sequence'].str.len()

print("\n=== SEQUENZLÄNGEN-STATISTIK (in Nukleotiden) ===")
stats_len = pd.DataFrame({
    'Human Seq Length': df_human['seq_len'].describe(),
    'Mouse Seq Length': df_mouse['seq_len'].describe()
})
print(stats_len)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Target-Verteilung (KDE / Histogramm)
sns.histplot(df_human['target'], kde=True, ax=axes[0, 0], color='blue', label='Human', stat="density", alpha=0.4)
sns.histplot(df_mouse['target'], kde=True, ax=axes[0, 0], color='orange', label='Mouse', stat="density", alpha=0.4)
axes[0, 0].set_title("Verteilung der Halbwertszeit-Targets (PC1)")
axes[0, 0].set_xlabel("Target-Wert (PC1 Half-Life)")
axes[0, 0].legend()

# B. Sequenzlängen-Verteilung (Log-Skala)
sns.kdeplot(df_human['seq_len'], ax=axes[0, 1], color='blue', label='Human', log_scale=True)
sns.kdeplot(df_mouse['seq_len'], ax=axes[0, 1], color='orange', label='Mouse', log_scale=True)
axes[0, 1].set_title("Verteilung der Sequenzlängen (log-skaliert)")
axes[0, 1].set_xlabel("Länge der reifen mRNA (bp)")
axes[0, 1].legend()

# C. Transkripte je Chromosom (Human, sortiert)
chrom_order_human = sorted(df_human['chromosome'].dropna().unique())
sns.countplot(data=df_human, x='chromosome', order=chrom_order_human, ax=axes[1, 0], color='steelblue')
axes[1, 0].set_title("Transkripte je Chromosom (Human)")
axes[1, 0].tick_params(axis='x', rotation=90)

# D. Zusammenhang Sequenzlänge vs. Target
sns.scatterplot(data=df_human.sample(min(2000, len(df_human))), x='seq_len', y='target', ax=axes[1, 1], alpha=0.3, s=15)
axes[1, 1].set_xscale('log')
axes[1, 1].set_title("Human: mRNA-Länge vs. Stabilität (Sample n=2000)")
axes[1, 1].set_xlabel("Länge (bp, log-Skala)")
axes[1, 1].set_ylabel("Target (PC1)")

plt.tight_layout()
plt.show()