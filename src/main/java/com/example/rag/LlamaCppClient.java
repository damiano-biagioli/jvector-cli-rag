package com.example.rag;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Client minimale per il server HTTP di llama.cpp ({@code llama-server}).
 *
 * <p>Usa l'endpoint OpenAI-compatible {@code POST {baseUrl}/v1/chat/completions}
 * con {@code stream=false} (risposta unica a fine generazione, come per
 * {@link OllamaClient}): la struttura della conversazione (system + user) resta
 * identica. Formato della richiesta:</p>
 *
 * <pre>
 * {
 *   "model": "...",   // ignorato: llama-server serve il modello caricato con -m
 *   "messages": [ {"role":"system",...}, {"role":"user",...} ],
 *   "stream": false
 * }
 * </pre>
 *
 * <p>La risposta e' in formato OpenAI: il testo generato e' in
 * {@code choices[0].message.content}.</p>
 *
 * <p>Avvio tipico del server (porta di default 8080):</p>
 * <pre>llama-server -m modello.gguf -c 8192 --port 8080</pre>
 * Attenzione a {@code -c}: il contesto di default puo' essere troppo piccolo per
 * i prompt RAG, che includono i chunk recuperati.
 */
public class LlamaCppClient implements LlmClient {

    private final String baseUrl;
    private final String model;
    private final HttpClient http;
    private final Gson gson = new Gson();

    /**
     * @param baseUrl URL del server llama.cpp (tipicamente http://localhost:8080)
     * @param model   nome del modello: inviato nella richiesta ma ignorato da
     *                llama-server, che usa il modello caricato all'avvio con {@code -m}
     */
    public LlamaCppClient(String baseUrl, String model) {
        // normalizza: niente slash finale, cosi' gli endpoint si compongono puliti
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.model = model;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    /**
     * Invia una conversazione (system + user) e restituisce la risposta del modello.
     *
     * @throws java.net.ConnectException se il server llama.cpp non e' raggiungibile
     * @throws IOException se il server risponde con un errore
     */
    @Override
    public String chat(String systemPrompt, String userMessage) throws IOException, InterruptedException {
        JsonObject system = new JsonObject();
        system.addProperty("role", "system");
        system.addProperty("content", systemPrompt);

        JsonObject user = new JsonObject();
        user.addProperty("role", "user");
        user.addProperty("content", userMessage);

        JsonArray messages = new JsonArray();
        messages.add(system);
        messages.add(user);

        JsonObject body = new JsonObject();
        body.addProperty("model", model);
        body.add("messages", messages);
        body.addProperty("stream", false);

        HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/v1/chat/completions"))
                // timeout generoso: i prompt RAG sono lunghi e la generazione puo' essere lenta
                .timeout(Duration.ofMinutes(5))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(body)))
                .build();

        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new IOException("llama-server ha risposto HTTP " + response.statusCode()
                    + ": " + response.body());
        }
        // formato OpenAI: choices[0].message.content
        JsonObject json = gson.fromJson(response.body(), JsonObject.class);
        return json.getAsJsonArray("choices")
                .get(0).getAsJsonObject()
                .getAsJsonObject("message")
                .get("content").getAsString();
    }
}
