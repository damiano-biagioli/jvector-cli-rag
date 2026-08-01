package com.example.rag;

import java.io.IOException;

/**
 * Astrazione minima del backend LLM usato per la generazione della risposta.
 *
 * <p>Implementazioni disponibili (selezionabili con {@code --backend}):</p>
 * <ul>
 *   <li>{@link OllamaClient} — API nativa di Ollama ({@code POST /api/chat})</li>
 *   <li>{@link LlamaCppClient} — llama-server di llama.cpp, endpoint OpenAI-compatible
 *       ({@code POST /v1/chat/completions})</li>
 * </ul>
 *
 * <p>Il resto dell'applicazione (retrieval, prompt, REPL) dipende solo da questa
 * interfaccia: aggiungere un nuovo backend significa solo scrivere una nuova
 * implementazione e registrarla in {@code Main}.</p>
 */
public interface LlmClient {

    /**
     * Invia una conversazione (system + user) e restituisce la risposta del modello.
     *
     * @throws java.net.ConnectException se il server LLM non e' raggiungibile
     * @throws IOException se il server risponde con un errore
     */
    String chat(String systemPrompt, String userMessage) throws IOException, InterruptedException;
}
