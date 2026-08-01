package com.example.rag;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Gestisce la cache su disco dell'indice, nella directory {@code <documenti>/.jvector-cache/}:
 *
 * <pre>
 *   manifest.properties  "impronta" dei documenti e dei parametri di indicizzazione
 *   chunks.bin           metadati dei chunk (file sorgente + testo)
 *   graph.bin            grafo JVector con i vettori inline (scritto da DocumentIndex.save)
 * </pre>
 *
 * La cache viene considerata valida solo se i file indicizzati (percorso, dimensione,
 * data di modifica), i parametri di chunking e il modello di embedding non sono cambiati
 * dall'ultima indicizzazione; in caso contrario l'indice viene ricostruito da zero.
 */
public class IndexCache {

    /** Nome della sottodirectory di cache creata dentro la directory dei documenti. */
    public static final String DIR_NAME = ".jvector-cache";
    /** Nome del file del grafo JVector serializzato. */
    public static final String GRAPH = "graph.bin";

    private static final String MANIFEST = "manifest.properties";
    private static final String CHUNKS = "chunks.bin";

    /**
     * Identificativo del modello di embedding usato: se in futuro si cambia modello,
     * gli embedding salvati non sarebbero compatibili e la cache va invalidata.
     */
    private static final String MODEL_ID = "all-MiniLM-L6-v2_quint8-avx2";

    /**
     * Verifica se la cache esiste ed e' coerente con lo stato attuale dei documenti.
     *
     * @param cacheDir directory di cache ({@code <documenti>/.jvector-cache})
     * @param docsDir  directory dei documenti
     * @param files    file .md/.txt trovati attualmente (ordinati come all'indicizzazione)
     * @return true se l'indice in cache puo' essere riutilizzato senza ricalcolare nulla
     */
    public static boolean isValid(Path cacheDir, Path docsDir, List<Path> files) {
        Path manifestPath = cacheDir.resolve(MANIFEST);
        // tutti e tre i file devono esistere, altrimenti la cache e' incompleta
        if (!Files.isRegularFile(manifestPath)
                || !Files.isRegularFile(cacheDir.resolve(CHUNKS))
                || !Files.isRegularFile(cacheDir.resolve(GRAPH))) {
            return false;
        }

        Properties props = new Properties();
        try (InputStream in = Files.newInputStream(manifestPath)) {
            props.load(in);
        } catch (IOException e) {
            return false;
        }

        // i parametri che influenzano gli embedding devono coincidere
        if (!MODEL_ID.equals(props.getProperty("embedding.model"))
                || !String.valueOf(TextChunker.CHUNK_SIZE).equals(props.getProperty("chunk.size"))
                || !String.valueOf(TextChunker.OVERLAP).equals(props.getProperty("chunk.overlap"))) {
            return false;
        }

        // anche l'insieme dei file (con dimensione e data di modifica) deve coincidere
        if (files.size() != parseLong(props.getProperty("file.count"), -1)) {
            return false;
        }
        for (int i = 0; i < files.size(); i++) {
            Path f = files.get(i);
            try {
                boolean same = docsDir.relativize(f).toString().equals(props.getProperty("file." + i + ".path"))
                        && Files.size(f) == parseLong(props.getProperty("file." + i + ".size"), -2)
                        && Files.getLastModifiedTime(f).toMillis()
                                == parseLong(props.getProperty("file." + i + ".mtime"), -3);
                if (!same) {
                    return false;
                }
            } catch (IOException e) {
                return false;
            }
        }
        return true;
    }

    /**
     * Scrive manifest e metadati dei chunk nella cache.
     * (Il grafo viene scritto separatamente da {@link DocumentIndex#save}.)
     */
    public static void save(Path cacheDir, Path docsDir, List<Path> files,
                            List<DocumentIndex.Chunk> chunks) throws IOException {
        Files.createDirectories(cacheDir);

        Properties props = new Properties();
        props.setProperty("embedding.model", MODEL_ID);
        props.setProperty("chunk.size", String.valueOf(TextChunker.CHUNK_SIZE));
        props.setProperty("chunk.overlap", String.valueOf(TextChunker.OVERLAP));
        props.setProperty("file.count", String.valueOf(files.size()));
        for (int i = 0; i < files.size(); i++) {
            Path f = files.get(i);
            props.setProperty("file." + i + ".path", docsDir.relativize(f).toString());
            props.setProperty("file." + i + ".size", String.valueOf(Files.size(f)));
            props.setProperty("file." + i + ".mtime",
                    String.valueOf(Files.getLastModifiedTime(f).toMillis()));
        }
        try (OutputStream out = Files.newOutputStream(cacheDir.resolve(MANIFEST))) {
            props.store(out, "jvector-rag index cache - non modificare a mano");
        }

        saveChunks(cacheDir.resolve(CHUNKS), chunks);
    }

    /** Carica i metadati dei chunk da una cache precedentemente salvata. */
    public static List<DocumentIndex.Chunk> loadChunks(Path cacheDir) throws IOException {
        Path file = cacheDir.resolve(CHUNKS);
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            int count = in.readInt();
            List<DocumentIndex.Chunk> chunks = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                String source = in.readUTF();
                String text = in.readUTF();
                // l'ordinale e' la posizione nella lista: corrisponde a quello assegnato
                // al vettore nel grafo al momento dell'indicizzazione
                chunks.add(new DocumentIndex.Chunk(source, i, text));
            }
            return chunks;
        }
    }

    /** Formato binario minimale: [numero chunk] poi per ogni chunk (sorgente, testo). */
    private static void saveChunks(Path file, List<DocumentIndex.Chunk> chunks) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file)))) {
            out.writeInt(chunks.size());
            for (DocumentIndex.Chunk c : chunks) {
                out.writeUTF(c.source());
                out.writeUTF(c.text());
            }
        }
    }

    private static long parseLong(String value, long fallback) {
        try {
            return value == null ? fallback : Long.parseLong(value);
        } catch (NumberFormatException e) {
            return fallback;
        }
    }
}
