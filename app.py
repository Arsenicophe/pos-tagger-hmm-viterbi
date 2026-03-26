"""
POS Tagger HMM + Viterbi — Arsene Mekondion
Application Streamlit pour l'etiquetage morphosyntaxique
"""

import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import string

# ──────────────────────────────────────────────
# Fonctions utilitaires (auto-contenu pour le deploiement)
# ──────────────────────────────────────────────

PUNCT = set(string.punctuation)
NOUN_SUFFIX = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er",
               "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness",
               "or", "ry", "scape", "ship", "ty"]
VERB_SUFFIX = ["ate", "ify", "ise", "ize"]
ADJ_SUFFIX = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish",
              "ive", "less", "ly", "ous"]
ADV_SUFFIX = ["ward", "wards", "wise"]


def assign_unk(tok):
    """Assigne un token inconnu selon la morphologie du mot."""
    if any(c.isdigit() for c in tok):
        return "--unk_digit--"
    elif any(c in PUNCT for c in tok):
        return "--unk_punct--"
    elif any(c.isupper() for c in tok):
        return "--unk_upper--"
    elif any(tok.endswith(s) for s in NOUN_SUFFIX):
        return "--unk_noun--"
    elif any(tok.endswith(s) for s in VERB_SUFFIX):
        return "--unk_verb--"
    elif any(tok.endswith(s) for s in ADJ_SUFFIX):
        return "--unk_adj--"
    elif any(tok.endswith(s) for s in ADV_SUFFIX):
        return "--unk_adv--"
    return "--unk--"


def get_word_tag(line, vocab):
    if not line.split():
        return "--n--", "--s--"
    word, tag = line.split()
    if word not in vocab:
        word = assign_unk(word)
    return word, tag


# ──────────────────────────────────────────────
# Construction du modele HMM
# ──────────────────────────────────────────────

def create_dictionaries(training_corpus, vocab):
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    prev_tag = "--s--"
    for word_tag in training_corpus:
        word, tag = get_word_tag(word_tag, vocab)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1
        prev_tag = tag
    return emission_counts, transition_counts, tag_counts


def create_transition_matrix(alpha, tag_counts, transition_counts):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    tag_to_idx = {t: i for i, t in enumerate(all_tags)}
    A = np.zeros((num_tags, num_tags))
    for (prev_tag, tag), count in transition_counts.items():
        A[tag_to_idx[prev_tag], tag_to_idx[tag]] = count
    tag_count_vec = np.array([tag_counts[t] for t in all_tags])
    A = (A + alpha) / (tag_count_vec.reshape(-1, 1) + alpha * num_tags)
    return A


def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    tag_to_idx = {t: i for i, t in enumerate(all_tags)}
    num_words = len(vocab)
    B = np.zeros((num_tags, num_words))
    for (tag, word), count in emission_counts.items():
        if word in vocab:
            B[tag_to_idx[tag], vocab[word]] = count
    tag_count_vec = np.array([tag_counts[t] for t in all_tags])
    B = (B + alpha) / (tag_count_vec.reshape(-1, 1) + alpha * num_words)
    return B


# ──────────────────────────────────────────────
# Algorithme de Viterbi (vectorise)
# ──────────────────────────────────────────────

def viterbi(A, B, vocab, states, tag_counts, words):
    """Execute Viterbi complet (init + forward + backward) sur une liste de mots."""
    num_tags = len(states)
    T = len(words)

    if T == 0:
        return []

    # Convertir les mots en indices vocab
    word_indices = []
    for w in words:
        if w in vocab:
            word_indices.append(vocab[w])
        else:
            unk = assign_unk(w)
            word_indices.append(vocab.get(unk, vocab.get("--unk--", 0)))

    # Pre-calcul des logs
    log_A = np.log(A)
    log_B = np.log(B)

    # Initialisation
    best_probs = np.full((num_tags, T), -np.inf)
    best_paths = np.zeros((num_tags, T), dtype=int)

    s_idx = states.index("--s--")
    best_probs[:, 0] = log_A[s_idx, :] + log_B[:, word_indices[0]]

    # Forward (vectorise)
    for i in range(1, T):
        prob_matrix = (best_probs[:, i - 1].reshape(-1, 1)
                       + log_A
                       + log_B[:, word_indices[i]].reshape(1, -1))
        best_probs[:, i] = np.max(prob_matrix, axis=0)
        best_paths[:, i] = np.argmax(prob_matrix, axis=0)

    # Backward
    z = [0] * T
    pred = [""] * T
    z[T - 1] = int(np.argmax(best_probs[:, T - 1]))
    pred[T - 1] = states[z[T - 1]]
    for i in range(T - 1, 0, -1):
        z[i - 1] = best_paths[z[i], i]
        pred[i - 1] = states[z[i - 1]]

    return pred


# ──────────────────────────────────────────────
# Chargement du modele (cache par Streamlit)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modele HMM...")
def load_model():
    """Charge les donnees et construit les matrices A et B une seule fois."""
    data_dir = Path(__file__).parent / "data"

    with open(data_dir / "WSJ_02-21.pos", "r") as f:
        training_corpus = f.readlines()

    with open(data_dir / "hmm_vocab.txt", "r") as f:
        voc_l = f.read().split("\n")

    vocab = {word: i for i, word in enumerate(sorted(voc_l))}

    emission_counts, transition_counts, tag_counts = create_dictionaries(
        training_corpus, vocab
    )
    states = sorted(tag_counts.keys())

    alpha = 0.001
    A = create_transition_matrix(alpha, tag_counts, transition_counts)
    B = create_emission_matrix(alpha, tag_counts, emission_counts, vocab)

    return A, B, vocab, states, tag_counts


# ──────────────────────────────────────────────
# Couleurs par categorie de tag
# ──────────────────────────────────────────────

TAG_COLORS = {
    # Noms
    "NN": "#4CAF50", "NNS": "#4CAF50", "NNP": "#388E3C", "NNPS": "#388E3C",
    # Verbes
    "VB": "#2196F3", "VBD": "#2196F3", "VBG": "#1976D2", "VBN": "#1976D2",
    "VBP": "#1565C0", "VBZ": "#1565C0",
    # Adjectifs
    "JJ": "#FF9800", "JJR": "#FF9800", "JJS": "#FF9800",
    # Adverbes
    "RB": "#9C27B0", "RBR": "#9C27B0", "RBS": "#9C27B0",
    # Determinants / Pronoms
    "DT": "#607D8B", "PRP": "#795548", "PRP$": "#795548",
    # Prepositions / Conjonctions
    "IN": "#00BCD4", "CC": "#009688",
    # Ponctuation
    ".": "#9E9E9E", ",": "#9E9E9E", ":": "#9E9E9E",
    # Nombres
    "CD": "#F44336",
}

TAG_DESCRIPTIONS = {
    "CC": "Conjonction de coordination",
    "CD": "Nombre cardinal",
    "DT": "Determinant",
    "EX": "There existentiel",
    "FW": "Mot etranger",
    "IN": "Preposition / conj. subord.",
    "JJ": "Adjectif",
    "JJR": "Adjectif comparatif",
    "JJS": "Adjectif superlatif",
    "MD": "Modal",
    "NN": "Nom singulier",
    "NNS": "Nom pluriel",
    "NNP": "Nom propre singulier",
    "NNPS": "Nom propre pluriel",
    "PDT": "Pre-determinant",
    "POS": "Possessif",
    "PRP": "Pronom personnel",
    "PRP$": "Pronom possessif",
    "RB": "Adverbe",
    "RBR": "Adverbe comparatif",
    "RBS": "Adverbe superlatif",
    "RP": "Particule",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verbe (base)",
    "VBD": "Verbe (passe)",
    "VBG": "Verbe (gerondif)",
    "VBN": "Verbe (participe passe)",
    "VBP": "Verbe (present, non 3e pers.)",
    "VBZ": "Verbe (present, 3e pers.)",
    "WDT": "Determinant Wh-",
    "WP": "Pronom Wh-",
    "WP$": "Pronom possessif Wh-",
    "WRB": "Adverbe Wh-",
}


def get_color(tag):
    return TAG_COLORS.get(tag, "#E0E0E0")


# ──────────────────────────────────────────────
# Interface Streamlit
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="POS Tagger - Arsene Mekondion",
        page_icon="🏷",
        layout="wide",
    )

    st.title("POS Tagger — HMM & Viterbi")
    st.caption("Par Arsene Mekondion | Modele de Markov Cache vectorise avec NumPy")

    # Charger le modele
    A, B, vocab, states, tag_counts = load_model()

    # Filtrer les tags speciaux pour l'affichage
    display_states = [s for s in states if not s.startswith("--")]

    # Sidebar : legende des tags
    with st.sidebar:
        st.header("Legende des tags POS")
        for tag in sorted(display_states):
            desc = TAG_DESCRIPTIONS.get(tag, tag)
            color = get_color(tag)
            st.markdown(
                f'<span style="background-color:{color};color:white;'
                f'padding:2px 8px;border-radius:4px;font-size:13px;">'
                f'{tag}</span> {desc}',
                unsafe_allow_html=True,
            )

    # Zone de saisie
    text = st.text_area(
        "Entre une phrase en anglais :",
        value="The old man sat on the bench and watched the children play in the park .",
        height=100,
    )

    if st.button("Analyser", type="primary") and text.strip():
        words = text.strip().split()

        # Viterbi
        tags = viterbi(A, B, vocab, states, tag_counts, words)

        if not tags:
            st.warning("Aucun mot a analyser.")
            return

        # ── Affichage colore ──
        st.subheader("Resultat")

        html_parts = []
        for word, tag in zip(words, tags):
            color = get_color(tag)
            html_parts.append(
                f'<span style="display:inline-block;margin:4px;">'
                f'<span style="font-size:16px;font-weight:bold;">{word}</span><br/>'
                f'<span style="background-color:{color};color:white;'
                f'padding:1px 6px;border-radius:3px;font-size:12px;">'
                f'{tag}</span></span>'
            )

        st.markdown(
            '<div style="line-height:2.8;">' + " ".join(html_parts) + "</div>",
            unsafe_allow_html=True,
        )

        # ── Tableau detaille ──
        st.subheader("Detail")
        df = pd.DataFrame({
            "Mot": words,
            "Tag POS": tags,
            "Description": [TAG_DESCRIPTIONS.get(t, t) for t in tags],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Stats ──
        st.subheader("Distribution des tags")
        tag_series = pd.Series(tags)
        counts = tag_series.value_counts()
        st.bar_chart(counts)


if __name__ == "__main__":
    main()
