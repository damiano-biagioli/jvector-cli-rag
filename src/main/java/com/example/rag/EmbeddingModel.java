package com.example.rag;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Genera embedding in locale con il modello sentence-transformers/all-MiniLM-L6-v2
 * in formato ONNX (384 dimensioni), senza dipendere da servizi esterni.
 *
 * <p>Pipeline per ogni testo:</p>
 * <ol>
 *   <li>tokenizzazione WordPiece (tokenizer HuggingFace, binding DJL con nativi
 *       per Windows/Linux/macOS);</li>
 *   <li>inferenza ONNX Runtime (CPU);</li>
 *   <li>mean pooling sull'ultimo hidden state, pesato con l'attention mask;</li>
 *   <li>normalizzazione L2: con vettori unitari la similarita' coseno usata
 *       da JVector coincide col prodotto scalare.</li>
 * </ol>
 */
public class EmbeddingModel implements AutoCloseable {

    /**
     * Limite di token per testo: il modello ne supporta 512, ma i chunk sono
     * corti e troncare a 256 dimezza il tempo di inferenza senza perdere nulla.
     */
    private static final int MAX_TOKENS = 256;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    /** Nomi degli input accettati dal grafo ONNX (variano tra le esportazioni). */
    private final Set<String> inputNames;

    /**
     * @param modelDir directory contenente {@code tokenizer.json} e {@code model.onnx}
     *                 (scaricati da {@link ModelDownloader})
     */
    public EmbeddingModel(Path modelDir) throws IOException, OrtException {
        this.tokenizer = HuggingFaceTokenizer.newInstance(modelDir.resolve("tokenizer.json"));
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // meta' dei core: compromesso ragionevole tra velocita' e reattivita' del sistema
        opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
        this.session = env.createSession(modelDir.resolve("model.onnx").toString(), opts);
        this.inputNames = session.getInputNames();
    }

    /**
     * Calcola l'embedding normalizzato (384 dimensioni) di un testo.
     * Metodo costo-lineare nel numero di token; chiamato una volta per chunk
     * in indicizzazione e una volta per domanda in chat.
     */
    public float[] embed(String text) throws OrtException {
        // 1. tokenizzazione
        Encoding enc = tokenizer.encode(text);
        long[] ids = truncate(enc.getIds());
        long[] mask = truncate(enc.getAttentionMask());
        long[] types = truncate(enc.getTypeIds());

        // 2. prepara solo gli input che il grafo ONNX dichiara davvero
        //    (alcune esportazioni non hanno token_type_ids)
        Map<String, OnnxTensor> inputs = new HashMap<>();
        try {
            if (inputNames.contains("input_ids")) {
                inputs.put("input_ids", tensor(new long[][]{ids}));
            }
            if (inputNames.contains("attention_mask")) {
                inputs.put("attention_mask", tensor(new long[][]{mask}));
            }
            if (inputNames.contains("token_type_ids")) {
                inputs.put("token_type_ids", tensor(new long[][]{types}));
            }

            // 3. inferenza: output = last_hidden_state [1, seqLen, 384]
            try (OrtSession.Result result = session.run(inputs)) {
                float[][][] hidden = (float[][][]) result.get(0).getValue();
                // 4. pooling + normalizzazione
                return meanPoolAndNormalize(hidden[0], mask);
            }
        } finally {
            // i tensori ONNX allocano memoria nativa: vanno sempre chiusi
            inputs.values().forEach(OnnxTensor::close);
        }
    }

    /** Crea un tensore int64 di forma [1, seqLen], formato atteso dal modello. */
    private OnnxTensor tensor(long[][] data) throws OrtException {
        return OnnxTensor.createTensor(env, data);
    }

    /** Tronca le sequenze oltre MAX_TOKENS (il modello richiede input a lunghezza fissa). */
    private static long[] truncate(long[] a) {
        if (a.length <= MAX_TOKENS) {
            return a;
        }
        long[] t = new long[MAX_TOKENS];
        System.arraycopy(a, 0, t, 0, MAX_TOKENS);
        return t;
    }

    /**
     * Mean pooling: media degli embedding dei token (escluso il padding, grazie
     * all'attention mask), seguita da normalizzazione L2.
     * E' la procedura standard prevista da all-MiniLM-L6-v2.
     */
    private static float[] meanPoolAndNormalize(float[][] tokenEmbeddings, long[] mask) {
        int dim = tokenEmbeddings[0].length;
        float[] mean = new float[dim];
        int count = 0;
        for (int i = 0; i < tokenEmbeddings.length; i++) {
            if (i < mask.length && mask[i] == 0) {
                continue; // token di padding: non contribuisce alla media
            }
            count++;
            for (int j = 0; j < dim; j++) {
                mean[j] += tokenEmbeddings[i][j];
            }
        }
        if (count > 0) {
            for (int j = 0; j < dim; j++) {
                mean[j] /= count;
            }
        }
        // normalizzazione L2 -> vettore unitario
        double norm = 0;
        for (float v : mean) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int j = 0; j < dim; j++) {
                mean[j] /= (float) norm;
            }
        }
        return mean;
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
