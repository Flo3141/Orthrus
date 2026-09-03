from pathlib import Path
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


# Output-Verzeichnis für Plots anlegen
plots_dir = Path(__file__).resolve().parent / "plots" if '__file__' in locals() else Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)


# Plot-Definitionen
def plot_target_distribution(ax):
    sns.histplot(df_human['target'], kde=True, ax=ax, color='blue', label='Human', stat="density", alpha=0.4)
    sns.histplot(df_mouse['target'], kde=True, ax=ax, color='orange', label='Mouse', stat="density", alpha=0.4)
    ax.set_title("Verteilung der Halbwertszeit-Targets (PC1)")
    ax.set_xlabel("Target-Wert (PC1 Half-Life)")
    ax.legend()


def plot_seq_len_distribution(ax):
    sns.kdeplot(df_human['seq_len'], ax=ax, color='blue', label='Human', log_scale=True)
    sns.kdeplot(df_mouse['seq_len'], ax=ax, color='orange', label='Mouse', log_scale=True)
    ax.set_title("Verteilung der Sequenzlängen (log-skaliert)")
    ax.set_xlabel("Länge der reifen mRNA (bp)")
    ax.legend()


def plot_chromosome_distribution(ax):
    chrom_order_human = sorted(df_human['chromosome'].dropna().unique())
    sns.countplot(data=df_human, x='chromosome', order=chrom_order_human, ax=ax, color='steelblue')
    ax.set_title("Transkripte je Chromosom (Human)")
    ax.tick_params(axis='x', rotation=90)


def plot_seq_len_vs_target(ax):
    sample_n = min(2000, len(df_human))
    sns.scatterplot(data=df_human.sample(sample_n, random_state=42), x='seq_len', y='target', ax=ax, alpha=0.3, s=15)
    ax.set_xscale('log')
    ax.set_title(f"Human: mRNA-Länge vs. Stabilität (Sample n={sample_n})")
    ax.set_xlabel("Länge (bp, log-Skala)")
    ax.set_ylabel("Target (PC1)")


# 1. Einzelne Plots als hochauflösende Grafiken speichern
individual_plots = [
    (plot_target_distribution, "target_distribution.png"),
    (plot_seq_len_distribution, "seq_len_distribution.png"),
    (plot_chromosome_distribution, "transcripts_per_chromosome_human.png"),
    (plot_seq_len_vs_target, "seq_len_vs_target_human.png"),
]

for plot_fn, filename in individual_plots:
    fig_single, ax_single = plt.subplots(figsize=(8, 6))
    plot_fn(ax_single)
    plt.tight_layout()
    fig_single.savefig(plots_dir / filename, dpi=300)
    plt.close(fig_single)

# 2. Gesamtübersicht (2x2 Grid) generieren und speichern
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_target_distribution(axes[0, 0])
plot_seq_len_distribution(axes[0, 1])
plot_chromosome_distribution(axes[1, 0])
plot_seq_len_vs_target(axes[1, 1])

plt.tight_layout()
fig.savefig(plots_dir / "mrna_dataset_overview.png", dpi=300)
print(f"\nPlots wurden erfolgreich im Unterordner '{plots_dir}' gespeichert.")

plt.show()