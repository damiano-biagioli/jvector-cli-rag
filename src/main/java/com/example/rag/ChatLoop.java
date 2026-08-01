package com.example.rag;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;
import java.util.stream.Collectors;

/**
 * REPL della chat RAG.
 *
 * <p>Per ogni domanda dell'utente:</p>
 * <ol>
 *   <li>calcola l'embedding della domanda;</li>
 *   <li>cerca i top-K chunk piu' simili nell'indice JVector e mostra le fonti
 *       (file, punteggio, anteprima) per rendere il retrieval ispezionabile;</li>
 *   <li>costruisce il prompt con il contesto recuperato e lo invia al backend LLM;</li>
 *   <li>stampa la risposta.</li>
 * </ol>
 *
 * Con {@code --no-llm} si ferma al punto 2: utile per verificare la qualita'
 * del retrieval senza dipendere dal backend LLM.
 */
public class ChatLoop {

    /**
     * Istruzioni per il modello: rispondere solo dal contesto.
     * Fondamentale per il RAG: riduce le allucinazioni su argomenti
     * non coperti dai documenti indicizzati.
     */
    private static final String SYSTEM_PROMPT = """
            Sei un assistente che risponde in italiano, in modo conciso e preciso.
            Rispondi SOLO basandoti sul contesto fornito dall'utente.
            Se la risposta non e' contenuta nel contesto, dillo esplicitamente.
            """;

    private final EmbeddingModel embeddings;
    private final DocumentIndex index;
    private final LlmClient llm;
    private final int topK;
    private final boolean noLlm;

    public ChatLoop(EmbeddingModel embeddings, DocumentIndex index, LlmClient llm,
                    int topK, boolean noLlm) {
        this.embeddings = embeddings;
        this.index = index;
        this.llm = llm;
        this.topK = topK;
        this.noLlm = noLlm;
    }

    public void run() throws Exception {
        // charset di default del sistema: la scelta piu' robusta per la console
        // sia su Windows sia su Linux (funziona anche con input da pipe)
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        System.out.println();
        System.out.println("Chat pronta" + (noLlm ? " (modalita' solo retrieval)" : "")
                + ". Scrivi una domanda, /exit per uscire.");

        while (true) {
            System.out.print("\n> ");
            String question = reader.readLine();
            // null = EOF (es. pipe terminata); blank o /exit = uscita volontaria
            if (question == null || question.isBlank() || question.equalsIgnoreCase("/exit")) {
                break;
            }

            // --- retrieval ---
            float[] queryVector = embeddings.embed(question);
            List<DocumentIndex.ScoredChunk> found = index.search(queryVector, topK);

            // le fonti vengono sempre mostrate: rendono verificabile la risposta
            System.out.println("Fonti recuperate:");
            for (DocumentIndex.ScoredChunk sc : found) {
                String preview = sc.chunk().text().replaceAll("\\s+", " ");
                if (preview.length() > 120) {
                    preview = preview.substring(0, 120) + "...";
                }
                System.out.printf("  [score %.3f] %s: %s%n", sc.score(), sc.chunk().source(), preview);
            }

            if (noLlm) {
                continue; // modalita' solo retrieval: nessuna generazione
            }

            // --- generazione: contesto + domanda nel messaggio utente ---
            String context = found.stream()
                    .map(sc -> sc.chunk().text())
                    .collect(Collectors.joining("\n---\n"));
            String userMessage = "Contesto:\n" + context + "\n\nDomanda: " + question;

            try {
                String answer = llm.chat(SYSTEM_PROMPT, userMessage);
                System.out.println("\nRisposta:\n" + answer.trim());
            } catch (java.net.ConnectException e) {
                // errore atteso e gestito: server LLM spento
                System.out.println("Errore: server LLM non raggiungibile. Verifica che sia in"
                        + " esecuzione (es. 'ollama serve' oppure 'llama-server -m modello.gguf').");
            } catch (Exception e) {
                System.out.println("Errore chiamando il server LLM: " + e.getMessage());
            }
        }
        System.out.println("Ciao!");
    }
}
