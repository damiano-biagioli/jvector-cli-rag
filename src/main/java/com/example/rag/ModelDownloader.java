package com.example.rag;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Scarica i file del modello di embedding da HuggingFace, solo se non gia' presenti.
 *
 * <p>Modello: sentence-transformers/all-MiniLM-L6-v2, variante ONNX quantizzata
 * uint8 per AVX2 (~23 MB invece di ~90 MB della versione fp32, con qualita'
 * quasi identica). Servono due file:</p>
 * <ul>
 *   <li>{@code tokenizer.json} — tokenizer WordPiece completo, caricato dalla libreria DJL;</li>
 *   <li>{@code model.onnx} — il grafo ONNX del modello, eseguito da ONNX Runtime.</li>
 * </ul>
 */
public class ModelDownloader {

    private static final String HF_BASE =
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/";
    private static final String MODEL_URL = HF_BASE + "onnx/model_quint8_avx2.onnx";
    private static final String TOKENIZER_URL = HF_BASE + "tokenizer.json";

    /**
     * Garantisce che {@code modelDir} contenga tokenizer e modello,
     * scaricandoli se mancanti.
     */
    public static void ensureModelFiles(Path modelDir) throws IOException, InterruptedException {
        Files.createDirectories(modelDir);
        downloadIfMissing(TOKENIZER_URL, modelDir.resolve("tokenizer.json"));
        downloadIfMissing(MODEL_URL, modelDir.resolve("model.onnx"));
    }

    /**
     * Scarica un file solo se non esiste gia'. Scrive prima su un file temporaneo
     * {@code .part} e poi lo rinomina: un'interruzione a meta' download non lascia
     * file corrotti che verrebbero scambiati per completi agli avvii successivi.
     */
    private static void downloadIfMissing(String url, Path dest) throws IOException, InterruptedException {
        if (Files.exists(dest) && Files.size(dest) > 0) {
            return;
        }
        System.out.println("Download: " + url);
        HttpClient client = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL) // HuggingFace redirige su CDN
                .build();
        HttpRequest request = HttpRequest.newBuilder(URI.create(url)).GET().build();
        HttpResponse<InputStream> response = client.send(request, HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            throw new IOException("HTTP " + response.statusCode() + " scaricando " + url);
        }
        Path tmp = dest.resolveSibling(dest.getFileName() + ".part");
        try (InputStream in = response.body()) {
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
        }
        Files.move(tmp, dest, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("  salvato in " + dest.toAbsolutePath());
    }
}
