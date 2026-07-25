from FlagEmbedding import BGEM3FlagModel
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


# Frasi appartenenti a gruppi semanticamente correlati.
sentences = [
    # Programmazione
    "Python è un linguaggio di programmazione molto diffuso.",
    "Java viene utilizzato per sviluppare applicazioni enterprise.",
    "Il compilatore traduce il codice sorgente in codice macchina.",
    # Database
    "SQLite è un database relazionale leggero.",
    "PostgreSQL è un sistema di gestione di database relazionali.",
    "Una query SQL permette di recuperare dati da una tabella.",
    # Animali
    "Il gatto dorme sul divano.",
    "Il cane corre nel giardino.",
    "I leoni sono grandi felini africani.",
    # Cucina
    "La pasta viene cotta in acqua bollente.",
    "La pizza viene preparata con farina, acqua e lievito.",
    "Il risotto richiede una cottura graduale con il brodo.",
]

# La query sarà rappresentata con un colore diverso.
query = "dove dorme il gatto?"


def main() -> None:
    # Su Apple Silicon iniziamo con FP16 disattivato per maggiore stabilità.
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=False,
    )

    # Inseriamo la query insieme alle altre frasi.
    # È importante applicare t-SNE a tutti i vettori contemporaneamente.
    all_texts = sentences + [query]

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

    # La perplexity deve essere inferiore al numero di campioni.
    # Con pochi punti è opportuno mantenerla bassa.
    perplexity = min(5, len(all_texts) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )

    points_2d = tsne.fit_transform(embeddings)

    # Tutti i punti tranne l'ultimo appartengono alle frasi normali.
    sentence_points = points_2d[:-1]

    # L'ultimo punto rappresenta la query.
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

    # Aggiungiamo un'etichetta a ciascuna frase.
    for index, sentence in enumerate(sentences):
        x, y = sentence_points[index]

        plt.annotate(
            sentence,
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )

    # Etichetta dedicata alla query.
    plt.annotate(
        f"QUERY: {query}",
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
