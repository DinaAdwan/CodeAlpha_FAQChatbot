"""CodeAlpha Task 2 — FAQ Chatbot (Streamlit chat UI)."""
from __future__ import annotations

import streamlit as st

from chatbot import FAQChatbot

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")


@st.cache_resource
def load_bot(threshold: float) -> FAQChatbot:
    return FAQChatbot("faqs.json", threshold=threshold)


with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.25, 0.05,
        help="Below this cosine similarity the bot admits it doesn't know.",
    )
    show_scores = st.toggle("Show match scores", value=False)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

bot = load_bot(threshold)

st.title("💬 FAQ Chatbot")
st.caption("CodeAlpha AI Internship — Task 2 · NLTK + TF-IDF + cosine similarity")

st.session_state.setdefault("messages", [])

# Starter suggestions
if not st.session_state.messages:
    st.info("Ask me anything about the internship. For example:")
    cols = st.columns(2)
    samples = [
        "How many tasks do I need?",
        "How should I name my GitHub repo?",
        "What is cosine similarity?",
        "How do I contact CodeAlpha?",
    ]
    for i, sample in enumerate(samples):
        if cols[i % 2].button(sample, use_container_width=True):
            st.session_state.pending = sample
            st.rerun()

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_scores and msg.get("scores"):
            st.caption(" · ".join(f"{q} ({s:.2f})" for q, s in msg["scores"]))

prompt = st.chat_input("Type your question…") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply, matches = bot.ask(prompt)
    scores = [(m.question, m.score) for m in matches]

    with st.chat_message("assistant"):
        st.markdown(reply)
        if show_scores and scores:
            st.caption(" · ".join(f"{q} ({s:.2f})" for q, s in scores))

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "scores": scores}
    )
