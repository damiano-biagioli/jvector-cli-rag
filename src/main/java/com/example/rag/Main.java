package com.example.rag;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Entry point dell'applicazione RAG.
 *
 * <p>Flusso all'avvio:</p>
 * <ol>
 *   <li>scarica il modello di embedding ONNX (solo la prima volta);</li>
 *   <li>scansiona la directory indicata raccogliendo i file .md e .txt;</li>
 *   <li>se esiste una cache valida in {@code <directory>/.jvector-cache} la carica da disco,
 *       altrimenti calcola gli embedding, costruisce l'indice JVector e lo salva in cache;</li>
 *   <li>avvia la chat: ogni domanda viene convertita in embedding, si recuperano i chunk
 *       piu' simili con JVector e si interroga il modello locale (Ollama o llama.cpp).</li>
 * </ol>
 *
 * <p>Uso:</p>
 * <pre>
 *   java -jar jvector-rag-1.0.0.jar &lt;directory-docs&gt; [opzioni]
 * </pre>
 *
 * Opzioni:
 * <ul>
 *   <li>{@code --backend ollama|llamacpp}  backend LLM per la generazione (default: ollama)</li>
 *   <li>{@code --ollama-model <nome>}  modello richiesto al backend (default: llama3.2;
 *       ignorato da llama-server, che serve il modello caricato con -m)</li>
 *   <li>{@code --ollama-url <url>}     URL del server LLM (default: http://localhost:11434
 *       con ollama, http://localhost:8080 con llamacpp)</li>
 *   <li>{@code --top-k <n>}            chunk recuperati per domanda (default: 10)</li>
 *   <li>{@code --model-dir <dir>}      dove scaricare il modello di embedding (default: ./models)</li>
 *   <li>{@code --rebuild}              ignora la cache e ricostruisce l'indice da zero</li>
 *   <li>{@code --no-llm}               solo retrieval, nessuna chiamata al LLM (utile per test)</li>
 * </ul>
 */
public class Main {

    public static void main(String[] args) throws Exception {
        // --- parsing degli argomenti ---
        Path docsDir = null;
        Path modelDir = Path.of("models");
        String backend = "ollama";
        // null = non specificato: il default dipende dal backend (deciso dopo il parsing)
        String llmUrl = null;
        String llmModel = "llama3.2";
        int topK = 10;
        boolean noLlm = false;
        boolean rebuild = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--backend" -> backend = args[++i];
                case "--ollama-model" -> llmModel = args[++i];
                case "--ollama-url" -> llmUrl = args[++i];
                case "--top-k" -> topK = Integer.parseInt(args[++i]);
                case "--model-dir" -> modelDir = Path.of(args[++i]);
                case "--no-llm" -> noLlm = true;
                case "--rebuild" -> rebuild = true;
                default -> {
                    if (args[i].startsWith("--")) {
                        usage();
                        return;
                    }
                    docsDir = Path.of(args[i]);
                }
            }
        }

        if (!backend.equals("ollama") && !backend.equals("llamacpp")) {
            usage();
            return;
        }
        // URL di default del backend: 11434 per Ollama, 8080 per llama-server
        if (llmUrl == null) {
            llmUrl = backend.equals("llamacpp") ? "http://localhost:8080" : "http://localhost:11434";
        }

        if (docsDir == null || !Files.isDirectory(docsDir)) {
            usage();
            return;
        }

        // --- 1. modello di embedding (download solo al primo avvio) ---
        ModelDownloader.ensureModelFiles(modelDir);
        EmbeddingModel embeddings = new EmbeddingModel(modelDir);

        // --- 2. raccolta dei documenti .md/.txt (ricorsiva, ordine stabile) ---
        List<Path> files;
        try (var walk = Files.walk(docsDir)) {
            files = walk.filter(Files::isRegularFile)
                    .filter(p -> {
                        String n = p.getFileName().toString().toLowerCase();
                        return n.endsWith(".md") || n.endsWith(".txt");
                    })
                    // la cache stessa e' dentro docsDir ma non contiene .md/.txt: nessun filtro extra
                    .sorted()
                    .toList();
        }
        if (files.isEmpty()) {
            System.out.println("Nessun file .md o .txt trovato in " + docsDir.toAbsolutePath());
            return;
        }
        System.out.println("Trovati " + files.size() + " file.");

        // --- 3. indice: da cache se valida, altrimenti ricostruito e salvato ---
        Path cacheDir = docsDir.resolve(IndexCache.DIR_NAME);
        DocumentIndex index;
        if (!rebuild && IndexCache.isValid(cacheDir, docsDir, files)) {
            // caricamento istantaneo: nessun embedding da ricalcolare
            System.out.println("Cache valida: caricamento indice da " + cacheDir + " ...");
            index = DocumentIndex.load(cacheDir.resolve(IndexCache.GRAPH),
                    IndexCache.loadChunks(cacheDir));
            System.out.println("Indice caricato: " + index.size() + " chunk.");
        } else {
            System.out.println(rebuild
                    ? "Rigenerazione forzata dell'indice (--rebuild)."
                    : "Cache assente o non valida: indicizzazione dei documenti...");
            index = new DocumentIndex();
            for (Path f : files) {
                String rel = docsDir.relativize(f).toString();
                try {
                    index.addDocument(rel, Files.readString(f), embeddings);
                    System.out.println("  indicizzato " + rel);
                } catch (Exception e) {
                    // un file illeggibile (es. encoding non UTF-8) non blocca l'indicizzazione
                    System.out.println("  ATTENZIONE: salto " + rel + " (" + e.getMessage() + ")");
                }
            }
            if (index.size() == 0) {
                System.out.println("Nessun contenuto indicizzabile, esco.");
                return;
            }
            System.out.println("Costruzione indice JVector (" + index.size() + " chunk)...");
            index.build();
            // persistenza: ai prossimi avvii l'indice verra' caricato da disco
            index.save(cacheDir.resolve(IndexCache.GRAPH));
            IndexCache.save(cacheDir, docsDir, files, index.chunks());
            System.out.println("Indice pronto e salvato in cache.");
        }

        // --- 4. chat RAG: il client concreto dipende dal backend scelto ---
        LlmClient llm = backend.equals("llamacpp")
                ? new LlamaCppClient(llmUrl, llmModel)
                : new OllamaClient(llmUrl, llmModel);
        new ChatLoop(embeddings, index, llm, topK, noLlm).run();
    }

    private static void usage() {
        System.out.println("""
                Uso: java -jar jvector-rag-1.0.0.jar <directory-docs> [opzioni]
                  --backend ollama|llamacpp   backend LLM (default ollama)
                  --ollama-model <nome>       modello richiesto al backend (default llama3.2;
                                              ignorato da llama-server, che usa il modello di -m)
                  --ollama-url <url>          URL del server LLM (default :11434 con ollama,
                                              :8080 con llamacpp)
                  --top-k <n>                 default 10
                  --model-dir <dir>           default ./models
                  --rebuild                   ignora la cache e ricostruisce l'indice
                  --no-llm                    solo retrieval (nessuna chiamata al LLM)
                """);
    }
}
