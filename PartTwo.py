
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# put near the top of PartTwo.py, after imports
import re, spacy
nlp_tok = spacy.load("en_core_web_sm", disable=["parser", "ner"])

NAME_RE = re.compile(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b")          # remove full names
NUM_RE  = re.compile(r"\b\d[\d,]*\b")                          # strip numbers

def custom_tokenizer(text):
    # quick clean-ups
    text = NAME_RE.sub(" ", text)      # personal names → space
    text = NUM_RE.sub(" ",  text)      # numbers        → space
    doc = nlp_tok(text.lower())
    return [
        tok.lemma_ for tok in doc
        if tok.is_alpha
        and tok.pos_ in {"NOUN", "VERB", "ADJ", "PROPN"}
        and len(tok) > 2
    ]


DATA_PATH = Path(__file__).resolve().parent / "texts" / "speeches" / "hansard40000.csv"

def load_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    #(a.i)
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")

    #(a.ii)
    top4 = df["party"].value_counts().nlargest(4).index
    df = df[df["party"].isin(top4) & (df["party"] != "Speaker")]


    #(a.iii)
    df = df[df["speech_class"] == "Speech"]

    #(a.iv)
    df = df[df["speech"].str.len() >= 1000]

    print("Filtered dataframe shape:", df.shape)
    return df.reset_index(drop=True)

def tfidf_split(df: pd.DataFrame, ngram_range=(1,1)):
    vect = TfidfVectorizer(stop_words="english",
                           max_features=3000,
                           ngram_range=ngram_range)

    X = vect.fit_transform(df["speech"])
    y = df["party"]
    return train_test_split(X, y, test_size=0.2,
                            stratify=y, random_state=26)

def train_and_report(model, Xtr, Xte, ytr, yte, title=""):
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    macro_f1 = round(f1_score(yte, preds, average="macro"), 3)
    print(f"\n=== {title} ===")
    print("macro-F1:", macro_f1)
    print(classification_report(yte, preds))


if __name__ == "__main__":
    df = load_and_filter(DATA_PATH)

    # unigrams
    Xtr, Xte, ytr, yte = tfidf_split(df, ngram_range=(1, 1))

    train_and_report(
        RandomForestClassifier(n_estimators=300, random_state=26),
        Xtr, Xte, ytr, yte, "RandomForest (uni)"
    )
    train_and_report(
        LinearSVC(),
        Xtr, Xte, ytr, yte, "Linear SVM (uni)"
    )

    # 1–3 grams
    Xtr3, Xte3, ytr3, yte3 = tfidf_split(df, ngram_range=(1, 3))

    train_and_report(
        LinearSVC(),
        Xtr3, Xte3, ytr3, yte3, "Linear SVM (1–3)"
    )
    train_and_report(
        RandomForestClassifier(n_estimators=300, random_state=26),
        Xtr3, Xte3, ytr3, yte3, "RandomForest (1–3)"
    )
    print(">>> reached custom block")

    # custom tokenizer
    vect_cust = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        token_pattern=None,  # <- disables the default regex & its warning
        stop_words="english",
        ngram_range=(1, 2),
        max_features=3000,
        min_df=3,
        max_df=0.9
    )

    X_cust = vect_cust.fit_transform(df["speech"])
    y = df["party"]

    XtrC, XteC, ytrC, yteC = train_test_split(
        X_cust, y, test_size=0.2, stratify=y, random_state=26
    )
    print(">>> running custom tokenizer experiment")

    svm_cust = LinearSVC(C=3, class_weight="balanced")  # balanced handles the party skew
    train_and_report(
        svm_cust,
        XtrC, XteC, ytrC, yteC,
        title=f"SVM custom C=3 ({X_cust.shape[1]} feats)"
    )
    print("--- custom block finished OK")

