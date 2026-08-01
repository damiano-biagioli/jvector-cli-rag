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
 * Client minimale per l'API chat di Ollama.
 *
 * <p>Effettua una {@code POST {baseUrl}/api/chat} con {@code stream=false}
 * (risposta unica alla fine della generazione: piu' semplice da gestire
 * dello streaming token-per-token). Formato della richiesta:</p>
 *
 * <pre>
 * {
 *   "model": "llama3.2",
 *   "messages": [ {"role":"system",...}, {"role":"user",...} ],
 *   "stream": false
 * }
 * </pre>
 *
 * La risposta contiene il testo generato in {@code message.content}.
 */
public class OllamaClient {

    private final String baseUrl;
    private final String model;
    private final HttpClient http;
    private final Gson gson = new Gson();

    /**
     * @param baseUrl URL del server Ollama (tipicamente http://localhost:11434)
     * @param model   nome del modello come noto a Ollama (vedere {@code ollama list})
     */
    public OllamaClient(String baseUrl, String model) {
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
     * @throws java.net.ConnectException se il server Ollama non e' raggiungibile
     * @throws IOException se il server risponde con un errore (es. modello non installato)
     */
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

        HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/api/chat"))
                // timeout generoso: il primo caricamento del modello in RAM puo' essere lento
                .timeout(Duration.ofMinutes(5))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(body)))
                .build();

        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new IOException("Ollama ha risposto HTTP " + response.statusCode()
                    + ": " + response.body()
                    + "\n(il modello '" + model + "' e' installato? controlla con: ollama list)");
        }
        JsonObject json = gson.fromJson(response.body(), JsonObject.class);
        return json.getAsJsonObject("message").get("content").getAsString();
    }
}
