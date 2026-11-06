"""FAQ retrieval engine: NLTK preprocessing + TF-IDF + cosine similarity."""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- one-time NLTK assets ---------------------------------------------------
for pkg, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
# Question words carry intent ("how", "what", "where") - keep them.
STOPWORDS -= {"how", "what", "when", "where", "which", "who", "why", "not", "no"}
LEMMATIZER = WordNetLemmatizer()

_POS_MAP = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}


def _wordnet_pos(tag: str) -> str:
    return _POS_MAP.get(tag[:1].upper(), wordnet.NOUN)


def preprocess(text: str) -> str:
    """Lowercase -> strip punctuation -> tokenize -> drop stopwords -> lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return " ".join(
        LEMMATIZER.lemmatize(tok, _wordnet_pos(tag))
        for tok, tag in tagged
        if tok not in STOPWORDS and tok not in string.punctuation and len(tok) > 1
    )


@dataclass
class Match:
    question: str
    answer: str
    score: float


class FAQChatbot:
    """Retrieval chatbot. Falls back to suggestions below `threshold`."""

    def __init__(self, faq_path: str | Path = "faqs.json", threshold: float = 0.25):
        self.faqs = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.threshold = threshold

        # Each FAQ's searchable document = question + tags (tags widen recall).
        corpus = [
            preprocess(f"{item['question']} {' '.join(item.get('tags', []))}")
            for item in self.faqs
        ]
        # Word 1-2 grams so phrases like "cosine similarity" stay meaningful.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def rank(self, query: str, top_k: int = 3) -> List[Match]:
        cleaned = preprocess(query)
        if not cleaned:
            return []
        scores = cosine_similarity(self.vectorizer.transform([cleaned]), self.matrix)[0]
        best = scores.argsort()[::-1][:top_k]
        return [
            Match(self.faqs[i]["question"], self.faqs[i]["answer"], float(scores[i]))
            for i in best
        ]
    GREETINGS = {"hi", "hello", "hey", "salam", "yo", "hola", "good morning", "good evening"}
    THANKS = {"thanks", "thank you", "thx", "shukran", "appreciate it"}
    BYE = {"bye", "goodbye", "see you", "exit", "quit"}
    
    def ask(self, query: str) -> Tuple[str, List[Match]]:
        """Returns (reply, ranked_matches). Reply is a fallback if confidence is low."""
        # Small-talk shortcut: greetings never match an FAQ, so handle them directly.
        cleaned = query.lower().strip(" !.?")
        if cleaned in self.GREETINGS:
            return "Hello! 👋 Ask me anything about the CodeAlpha internship — tasks, certificates, submission, or the AI concepts behind them.", []
        if cleaned in self.THANKS:
            return "You're welcome! Anything else?", []
        if cleaned in self.BYE:
            return "Good luck with your tasks! 🚀", []

        matches = self.rank(query)
   
        if not matches or matches[0].score < self.threshold:
            suggestions = [m.question for m in matches[:3] if m.score > 0.05]
            reply = "I'm not sure I understood that."
            if suggestions:
                reply += " Did you mean one of these?\n" + "\n".join(
                    f"• {q}" for q in suggestions
                )
            else:
                reply += " Try rephrasing, or ask about tasks, certificates, or submission."
            return reply, matches
        return matches[0].answer, matches


if __name__ == "__main__":
    bot = FAQChatbot()
    for q in ["how many projects must i finish?", "what do i name my repo", "pizza"]:
        reply, ranked = bot.ask(q)
        print(f"\nQ: {q}\nA: {reply}\n   (top score {ranked[0].score:.2f})")
