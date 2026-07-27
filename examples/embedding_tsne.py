"""Demo didattica: proietta con t-SNE un piccolo insieme di embedding BGE-M3."""

from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

SENTENCES = [
    "Python è un linguaggio di programmazione molto diffuso.",
    "Java viene utilizzato per sviluppare applicazioni enterprise.",
    "Il compilatore traduce il codice sorgente in codice macchina.",
    "SQLite è un database relazionale leggero.",
    "PostgreSQL è un sistema di gestione di database relazionali.",
    "Una query SQL permette di recuperare dati da una tabella.",
    "Il gatto dorme sul divano.",
    "Il cane corre nel giardino.",
    "I leoni sono grandi felini africani.",
    "La pasta viene cotta in acqua bollente.",
    "La pizza viene preparata con farina, acqua e lievito.",
    "Il risotto richiede una cottura graduale con il brodo.",
]
QUERY = "dove dorme il gatto?"


def main() -> None:
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    all_texts = SENTENCES + [QUERY]
    output = model.encode(
        all_texts,
        batch_size=8,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = np.asarray(output["dense_vecs"])

    print(f"Numero di testi: {len(all_texts)}")
    print(f"Forma degli embedding: {embeddings.shape}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_files = {
        "completi": OUTPUT_DIR / "embeddings.npy",
        "frasi": OUTPUT_DIR / "sentence_embeddings.npy",
        "query": OUTPUT_DIR / "query_embedding.npy",
    }
    np.save(output_files["completi"], embeddings, allow_pickle=False)
    np.save(output_files["frasi"], embeddings[:-1], allow_pickle=False)
    np.save(output_files["query"], embeddings[-1:], allow_pickle=False)
    for label, output_file in output_files.items():
        print(f"Embedding {label} salvati in: {output_file}")

    points_2d = TSNE(
        n_components=2,
        perplexity=min(5, len(all_texts) - 1),
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(embeddings)
    sentence_points = points_2d[:-1]
    query_point = points_2d[-1]

    plt.figure(figsize=(14, 9))
    plt.scatter(
        sentence_points[:, 0],
        sentence_points[:, 1],
        label="Frasi",
        alpha=0.75,
        s=80,
    )
    plt.scatter(
        query_point[0],
        query_point[1],
        label="Query",
        marker="X",
        s=220,
    )
    for index, sentence in enumerate(SENTENCES):
        x, y = sentence_points[index]
        plt.annotate(
            sentence,
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )
    plt.annotate(
        f"QUERY: {QUERY}",
        xy=(query_point[0], query_point[1]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )
    plt.title("Visualizzazione 2D degli embedding con t-SNE")
    plt.xlabel("Componente t-SNE 1")
    plt.ylabel("Componente t-SNE 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
