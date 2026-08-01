# jvector-rag

Mini progetto Java di **RAG (Retrieval-Augmented Generation) completamente locale**:

- **JVector** (DataStax) — indice vettoriale ANN in Java puro, con persistenza su disco stile DiskANN
- **ONNX Runtime** — embeddings generati in locale con `all-MiniLM-L6-v2` (nessun servizio esterno)
- **Ollama** oppure **llama.cpp** — LLM locale per la generazione delle risposte (scelta con `--backend`)

Cross-platform: lo stesso fat jar funziona su **Windows e Linux** (e macOS), perché le librerie
native (ONNX Runtime, tokenizer HuggingFace) per tutte le piattaforme sono incluse nel jar.

## Requisiti

| Componente | Versione | Note |
|---|---|---|
| JDK | 17+ | il progetto è compilato con `--release 17` |
| Maven | 3.6+ | solo per la build |
| Ollama **o** llama.cpp | recente | Ollama: con un modello, es. `ollama pull llama3.2`; llama.cpp: un modello GGUF |
| Connessione | solo al primo avvio | per scaricare il modello di embedding (~23 MB) da HuggingFace |

## Build

```bash
mvn package
```

Produce `target/jvector-rag-1.0.0.jar` (fat jar eseguibile, ~170 MB con le native di tutti gli OS).

## Uso

```bash
# avvia Ollama (se non già in esecuzione)...
ollama serve
# ...oppure llama.cpp:
llama-server -m modello.gguf -c 8192 --port 8080

# in un altro terminale (con llama.cpp aggiungi --backend llamacpp):
java -jar target/jvector-rag-1.0.0.jar <directory-documenti>
```

> **Attenzione (llama.cpp) — finestra di contesto:** il prompt RAG contiene il system
> prompt, i chunk recuperati (default: 10 chunk da ~1000 caratteri, quindi migliaia di
> token) e la domanda. Se il totale supera la context window con cui è stato avviato
> `llama-server` (opzione `-c`, il cui default può essere ben più piccolo di 8192 a
> seconda della versione), il server tronca il contesto o risponde con un errore:
> risposte degradate o prive di senso. Avvia quindi il server con `-c 8192` o più,
> aumentandola ulteriormente se alzi `--top-k`. Con Ollama il problema non si pone
> perché la gestione del contesto è automatica.

Il primo avvio scarica il modello di embedding in `./models/` e indicizza tutti i file
`.md` e `.txt` trovati (ricorsivamente) nella directory indicata, poi apre la chat:

```
> Quante lune ha il pianeta Zorblax?
Fonti recuperate:
  [score 0,849] pianeti.md: # Il sistema solare di Zorblax ...
Risposta:
Il pianeta Zorblax ha 17 lune confermate.
```

Nella chat: `/exit` (o riga vuota, o EOF) per uscire.

### Opzioni

| Opzione | Default | Descrizione |
|---|---|---|
| `--backend ollama\|llamacpp` | `ollama` | backend LLM per la generazione |
| `--ollama-model <nome>` | `llama3.2` | modello richiesto al backend (con Ollama: `ollama list` per l'elenco; llama-server lo ignora, usa il modello di `-m`) |
| `--ollama-url <url>` | `:11434` (ollama), `:8080` (llamacpp) | URL del server LLM |
| `--top-k <n>` | `10` | numero di chunk recuperati per ogni domanda |
| `--model-dir <dir>` | `./models` | dove scaricare il modello di embedding |
| `--rebuild` | — | ignora la cache e ricostruisce l'indice da zero |
| `--no-llm` | — | solo retrieval (nessuna chiamata a Ollama), utile per test |

## Come funziona

```
file .md/.txt ──▶ chunking (~1000 char, overlap 200) ──▶ embeddings ONNX (MiniLM, 384 dim)
                                                              │
                                                              ▼
risposta ◀── LLM (Ollama o llama.cpp) ◀── prompt con contesto ◀── top-K chunk simili ◀── indice JVector
                                                              ▲
                                              embedding della domanda (stesso modello)
```

1. **Chunking** (`TextChunker`): finestra scorrevole con taglio su fine riga/spazio.
2. **Embedding** (`EmbeddingModel`): tokenizzazione HuggingFace + inferenza ONNX Runtime,
   mean pooling con attention mask e normalizzazione L2.
3. **Indicizzazione** (`DocumentIndex`): grafo ANN di JVector (algoritmo Vamana, famiglia
   DiskANN/HNSW), similarità coseno. JVector indicizza solo vettori: la mappatura
   ordinale → testo è mantenuta dall'applicazione.
4. **Retrieval + generazione** (`ChatLoop`, `LlmClient` con `OllamaClient`/`LlamaCppClient`): i chunk più simili alla domanda
   vengono inseriti nel prompt come contesto; il system prompt istruisce il modello a
   rispondere solo dal contesto.

## Persistenza dell'indice

Alla prima indicizzazione l'indice viene salvato in `<directory-documenti>/.jvector-cache/`:

```
.jvector-cache/
├── manifest.properties   # impronta dei file + parametri (modello, chunking)
├── chunks.bin            # metadati dei chunk (sorgente + testo)
└── graph.bin             # grafo JVector con vettori inline (memory-mapped)
```

Agli avvii successivi, se **file (percorso/dimensione/data di modifica), parametri di
chunking e modello di embedding non sono cambiati**, l'indice viene caricato da disco
in memory-mapping senza ricalcolare alcun embedding. Altrimenti viene ricostruito
automaticamente. `--rebuild` forza la ricostruzione.

> Nota Windows: la directory `.jvector-cache` è nascosta solo per convenzione del nome
> (punto iniziale); su Windows resta una normale cartella visibile.

## Struttura del progetto

```
src/main/java/com/example/rag/
├── Main.java              # entry point: argomenti, orchestrazione, logica di cache
├── ModelDownloader.java   # download del modello ONNX da HuggingFace (primo avvio)
├── EmbeddingModel.java    # tokenizer HF + ONNX Runtime: testo → embedding
├── TextChunker.java       # suddivisione dei documenti in chunk
├── DocumentIndex.java     # indice JVector: build, ricerca, persistenza su disco
├── IndexCache.java        # manifest + metadati della cache su disco
├── LlmClient.java         # interfaccia comune ai backend LLM (chat system+user → risposta)
├── OllamaClient.java      # backend Ollama: POST /api/chat
├── LlamaCppClient.java    # backend llama.cpp: POST /v1/chat/completions (OpenAI-compatible)
└── ChatLoop.java          # REPL: retrieval, stampa fonti, generazione
```

## Troubleshooting

| Problema | Soluzione |
|---|---|
| `server LLM non raggiungibile` | avvia Ollama con `ollama serve` oppure llama.cpp con `llama-server -m modello.gguf` |
| `Ollama ha risposto HTTP 404` | il modello non è installato: `ollama pull llama3.2` |
| Contesto troppo piccolo con llama.cpp | aumenta la context window del server: `llama-server -c 8192` (i prompt RAG includono i chunk recuperati) |
| Errore di download del modello | serve connessione a huggingface.co (solo al primo avvio) |
| File saltato in indicizzazione | il progetto legge UTF-8: converti il file (es. da CP1252) |
| Warning `jdk.incubator.vector` | solo prestazionale, ignorabile (con JVector 3.0.6 e Java 21 il flag non ha effetto) |

## Possibili estensioni

- Supporto PDF/DOCX (es. Apache PDFBox, poi stesso pipeline di chunking)
- Reranking dei risultati con un cross-encoder
- Compressione PQ/BQ dei vettori su disco (già supportata da JVector) per indici molto grandi
- Streaming della risposta di Ollama token-per-token
