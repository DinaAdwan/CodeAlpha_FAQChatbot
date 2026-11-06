# CodeAlpha_FAQChatbot

**CodeAlpha AI Internship — Task 2**

A retrieval-based FAQ chatbot. It preprocesses text with NLTK, vectorizes both the
knowledge base and the user's question with TF-IDF, and answers with the FAQ that
has the highest cosine similarity.

## Pipeline
1. **Preprocess** — lowercase → strip punctuation → tokenize → remove stopwords → POS-aware lemmatization.
2. **Vectorize** — TF-IDF over word 1–2 grams (`question + tags` as the searchable document).
3. **Match** — cosine similarity between the query vector and every FAQ vector.
4. **Decide** — if the top score is below the threshold, the bot admits uncertainty and suggests near matches instead of hallucinating.

## Design notes
- Question words (`how`, `what`, `where`…) are deliberately **kept** in the vocabulary — they carry intent.
- Each FAQ carries `tags` that widen recall without polluting the displayed answer.
- The confidence threshold is tunable live from the sidebar.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py     # UI
python chatbot.py        # CLI smoke test
```

## Extending
Swap `faqs.json` for any other domain — no code changes needed.
