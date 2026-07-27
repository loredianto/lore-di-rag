# lore-di-rag

Pipeline RAG locale minimale: carica PDF e file testuali, crea chunk, genera
embedding BGE-M3 e salva un indice NumPy ricercabile.

## Installazione

```bash
python -m pip install -e .
```

Per includere anche la demo grafica t-SNE:

```bash
python -m pip install -e ".[viz]"
```

## Uso

Inserisci i documenti in `data/input/`, quindi crea l'indice:

```bash
python -m lore_di_rag index
```

Gli artefatti vengono salvati in `data/indexes/default/`:

- `embeddings.npy`: matrice `float32` degli embedding;
- `chunks.jsonl`: testo e metadati allineati alle righe della matrice;
- `manifest.json`: configurazione e versione del formato.

Esegui una ricerca semantica:

```bash
python -m lore_di_rag search "la mia domanda" --top-k 5
```

Percorsi, modello e parametri di chunking possono essere cambiati con le
opzioni mostrate da `python -m lore_di_rag index --help`.

È disponibile anche il wrapper diretto:

```bash
python scripts/build_index.py
```

La demo originale degli embedding è conservata in
`examples/embedding_tsne.py` e non fa parte della pipeline di indicizzazione.
