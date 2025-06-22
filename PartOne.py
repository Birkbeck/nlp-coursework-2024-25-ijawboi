#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
import re
import math
import pickle
import pandas as pd
from pathlib import Path
from collections import counter


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text: str, d: dict):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """

    sents = sent_tokenize (text)
    if not sents:
        return  0.0

    tokens = word_tokenize (text)
    words = [t for t in tokens if t.isalpha()]

    n_sent = len(sents)
    n_word = len(words)
    n_syl = sum(count_syl (w,d) for w in words)

    if n_word == 0:
        return 0.0

    return 0.39 * (n_word / n_sent) + 11.8 * (n_syl / n_word) - 15.59



def count_syl(word: str, d: dict):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word_lwr = word.lower()
    if word_lwr in d:
        proctn = d[word_lwr]
        syl_counts = [sum(bool(PHONEME_DIGIT_RE.search(ph)) for ph in pron) for pron in proctn]
        return min(syl_counts)

    clusters = VOWEL_RE.findall(word_lwr)
    return max(1, len(clusters))


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    rows = []
    for fp in path.glob("*.txt"):
        m = re.match(r"(.+?)-(.+?)-(\d{4})\.txt$", fp.name)
        if not m:
            raise ValueError(f"Unexpected filename pattern -> {fp.name}")

        raw_title, raw_author, year = m.groups()
        rows.append({
            "text": fp.read_text(encoding="utf-8"),
            "title": raw_title.replace("_", " ").strip(),
            "author": raw_author.replace("_", " ").strip(),
            "year": int(year)
        })

        df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    store_path.mkdir(parents=True, exist_ok=True)
    pickle_fp = store_path / out_name

    if pickle_fp.exists():
        return pd.read_pickle(pickle_fp)

    texts = df["text"].tolist()
    docs = list(nlp.pipe(texts, batch_size=8, n_process=-1))

    df = df.copy()
    df["parsed"] = docs

    df.to_pickle(pickle_fp)
    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = [tok.lower() for tok in word_tokenize(text) if tok.isalpha()]
    if not tokens:
        return 0.0
    types = set(tokens)
    return len(types) / len(tokens)


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb: str):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    target = target_verb.lower()
    subj_anyverb = Counter()
    subj_target = Counter()

    for tok in doc:
        if tok.dep_ == "nsubj":
            lemma = tok.lemma_.lower()
            subj_anyverb[lemma] += 1
            if tok.head.lemma_.lower() == target:
                subj_target[lemma] += 1

    total_any = sum(subj_anyverb.values())
    total_target = sum(subj_target.values())

    pmi_scores = {}
    for subj, c_xy in subj_target.items():
        p_xy = c_xy / total_target
        p_x = subj_anyverb[subj] / total_any

        pmi_scores[subj] = math.log2(p_xy / p_x) if p_x else 0.0

    ranked = sorted(pmi_scores.items(),
                    key=lambda kv: (-kv[1], -subj_target[kv[0]]))[:top_n]
    return ranked



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    counter = Counter(tok.lemma_.lower()
                      for tok in doc if tok.pos_ == "ADJ")
    return counter.most_common(top_n)



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    print(adjective_counts(df))

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")


