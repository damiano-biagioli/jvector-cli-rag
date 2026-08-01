package com.example.rag;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Indice vettoriale basato su JVector, con persistenza su disco.
 *
 * <p>JVector indicizza solo i vettori (identificati da un ordinale intero):
 * questa classe mantiene anche la mappatura ordinale -&gt; chunk di testo,
 * necessaria per il RAG.</p>
 *
 * <p>L'indice puo' trovarsi in due stati:</p>
 * <ul>
 *   <li><b>in memoria</b> ({@link OnHeapGraphIndex}): appena costruito dagli embedding
 *       calcolati all'avvio, tramite {@link #addDocument} + {@link #build()};</li>
 *   <li><b>on disk</b> ({@link OnDiskGraphIndex}): caricato da un indice salvato in
 *       precedenza con {@link #save}, tramite {@link #load}. Il file viene aperto in
 *       memory-mapping (approccio DiskANN) e i vettori full-precision, scritti inline
 *       nel grafo, vengono letti dal disco a ogni ricerca.</li>
 * </ul>
 */
public class DocumentIndex implements AutoCloseable {

    /** Un frammento di documento indicizzato: file sorgente, ordinale nel grafo, testo. */
    public record Chunk(String source, int ordinal, String text) {}

    /** Risultato di una ricerca: chunk + punteggio di similarita' coseno (0..1). */
    public record ScoredChunk(Chunk chunk, float score) {}

    /**
     * Factory dei vettori JVector: sceglie automaticamente l'implementazione
     * piu' veloce disponibile (SIMD via Vector API se abilitata).
     */
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Metadati dei chunk, in posizione corrispondente all'ordinale nel grafo. */
    private final List<Chunk> chunks = new ArrayList<>();
    /** Vettori in memoria (usati solo durante la costruzione dell'indice). */
    private final List<VectorFloat<?>> vectors = new ArrayList<>();

    // --- stato per la modalita' in memoria ---
    private RandomAccessVectorValues ravv;
    private OnHeapGraphIndex heapIndex;

    // --- stato per la modalita' on disk ---
    private OnDiskGraphIndex diskIndex;
    private ReaderSupplier readerSupplier;

    /**
     * Suddivide un documento in chunk, ne calcola gli embedding e li accoda
     * per la successiva costruzione del grafo con {@link #build()}.
     *
     * @param source percorso del file (mostrato poi come "fonte" nelle risposte)
     * @param content contenuto testuale del documento
     * @param model  modello di embedding gia' caricato
     */
    public void addDocument(String source, String content, EmbeddingModel model) throws Exception {
        for (String piece : TextChunker.chunk(content)) {
            float[] embedding = model.embed(piece);
            // l'ordinale del chunk coincide con la sua posizione in lista
            chunks.add(new Chunk(source, chunks.size(), piece));
            vectors.add(toVector(embedding));
        }
    }

    /**
     * Costruisce il grafo ANN (algoritmo Vamana, famiglia DiskANN/HNSW)
     * sui vettori accumulati con {@link #addDocument}.
     */
    public void build() {
        if (vectors.isEmpty()) {
            throw new IllegalStateException("Nessun vettore da indicizzare");
        }
        ravv = new ListRandomAccessVectorValues(vectors, vectors.get(0).length());
        // funzione di scoring usata durante la costruzione: coseno su vettori in memoria
        BuildScoreProvider bsp =
                BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.COSINE);
        // M=16: grado del grafo; beamWidth=100: profondita' di ricerca in costruzione;
        // 1.2, 1.2: fattori di overflow del grado e di rilassamento della diversita'
        try (GraphIndexBuilder builder =
                     new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f)) {
            heapIndex = builder.build(ravv);
        } catch (IOException e) {
            throw new RuntimeException("Errore nella costruzione dell'indice", e);
        }
    }

    /**
     * Salva il grafo su disco con le opzioni di default: i vettori full-precision
     * vengono scritti inline nel file, permettendo ricerche esatte senza
     * ricaricare nulla in memoria.
     */
    public void save(Path graphFile) throws IOException {
        if (heapIndex == null) {
            throw new IllegalStateException("Indice non ancora costruito");
        }
        // OnDiskGraphIndex.write non crea le directory padre: le creiamo noi
        if (graphFile.getParent() != null) {
            java.nio.file.Files.createDirectories(graphFile.getParent());
        }
        OnDiskGraphIndex.write(heapIndex, ravv, graphFile);
    }

    /**
     * Carica da disco (memory-mapped) un indice salvato con {@link #save},
     * senza ricalcolare gli embedding.
     *
     * @param graphFile file del grafo ({@code graph.bin})
     * @param chunks    metadati dei chunk caricati dalla cache (stesso ordine degli ordinali)
     */
    public static DocumentIndex load(Path graphFile, List<Chunk> chunks) throws IOException {
        DocumentIndex di = new DocumentIndex();
        di.chunks.addAll(chunks);
        // il Supplier apre un nuovo reader memory-mapped per ogni thread che cerca
        di.readerSupplier = new SimpleMappedReader.Supplier(graphFile);
        di.diskIndex = OnDiskGraphIndex.load(di.readerSupplier);
        return di;
    }

    /**
     * Cerca i {@code topK} chunk piu' simili alla query per similarita' coseno.
     * Funziona sia in modalita' in memoria sia on disk.
     */
    public List<ScoredChunk> search(float[] query, int topK) {
        VectorFloat<?> q = toVector(query);
        SearchResult result;
        if (diskIndex != null) {
            // modalita' on disk: scoring esatto sui vettori inline letti dal file mappato
            ScoreFunction.ExactScoreFunction esf =
                    diskIndex.getView().rerankerFor(q, VectorSimilarityFunction.COSINE);
            SearchScoreProvider ssp = new SearchScoreProvider(esf);
            try (GraphSearcher searcher = new GraphSearcher(diskIndex)) {
                result = searcher.search(ssp, topK, Bits.ALL);
            } catch (IOException e) {
                throw new RuntimeException("Errore durante la ricerca sull'indice", e);
            }
        } else {
            // modalita' in memoria: ricerca esatta sui vettori gia' in RAM
            result = GraphSearcher.search(
                    q, topK, ravv, VectorSimilarityFunction.COSINE, heapIndex, Bits.ALL);
        }

        // dagli ordinali ai chunk di testo
        List<ScoredChunk> out = new ArrayList<>();
        for (SearchResult.NodeScore ns : result.getNodes()) {
            out.add(new ScoredChunk(chunks.get(ns.node), ns.score));
        }
        return out;
    }

    /** Metadati dei chunk indicizzati (usati per salvarli nella cache). */
    public List<Chunk> chunks() {
        return chunks;
    }

    /** Numero di chunk indicizzati. */
    public int size() {
        return chunks.size();
    }

    /** Converte un float[] nel tipo vettoriale di JVector. */
    private static VectorFloat<?> toVector(float[] data) {
        VectorFloat<?> v = VTS.createFloatVector(data.length);
        for (int i = 0; i < data.length; i++) {
            v.set(i, data[i]);
        }
        return v;
    }

    /** Rilascia i reader memory-mapped, se l'indice era stato caricato da disco. */
    @Override
    public void close() throws IOException {
        if (readerSupplier != null) {
            readerSupplier.close();
        }
    }
}
