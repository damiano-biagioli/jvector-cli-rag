package com.example.rag;

import java.util.ArrayList;
import java.util.List;

/**
 * Suddivide il testo dei documenti in chunk per l'indicizzazione.
 *
 * <p>Strategia: finestra scorrevole di {@link #CHUNK_SIZE} caratteri con overlap
 * di {@link #OVERLAP} caratteri, cercando di spezzare su un fine riga o su uno
 * spazio per non troncare le frasi a meta'. La dimensione e' scelta per restare
 * entro il limite di token del modello di embedding (~1000 caratteri ≈ 250 token).</p>
 *
 * <p>NOTA: questi parametri influenzano gli embedding prodotti e sono salvati nel
 * manifest della cache: modificarli invalida automaticamente gli indici esistenti.</p>
 */
public class TextChunker {

    /** Lunghezza massima di un chunk in caratteri. */
    public static final int CHUNK_SIZE = 1000;
    /** Sovrapposizione tra chunk consecutivi, per non perdere il contesto ai confini. */
    public static final int OVERLAP = 200;

    public static List<String> chunk(String text) {
        String normalized = text.replace("\r\n", "\n").trim();
        List<String> chunks = new ArrayList<>();
        int pos = 0;
        while (pos < normalized.length()) {
            int end = Math.min(pos + CHUNK_SIZE, normalized.length());
            if (end < normalized.length()) {
                // prova a spezzare su fine riga, altrimenti sull'ultimo spazio,
                // ma solo se il punto di taglio non e' troppo vicino all'inizio
                int newline = normalized.lastIndexOf('\n', end);
                int space = normalized.lastIndexOf(' ', end);
                if (newline > pos + CHUNK_SIZE / 2) {
                    end = newline;
                } else if (space > pos + CHUNK_SIZE / 2) {
                    end = space;
                }
            }
            String piece = normalized.substring(pos, end).trim();
            if (!piece.isEmpty()) {
                chunks.add(piece);
            }
            if (end >= normalized.length()) {
                break;
            }
            // avanza la finestra con overlap; il max() garantisce progresso (no loop infiniti)
            pos = Math.max(end - OVERLAP, pos + 1);
        }
        return chunks;
    }
}
