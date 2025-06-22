
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

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
    Xtr, Xte, ytr, yte = tfidf_split(df, ngram_range=(1,1))

    train_and_report(RandomForestClassifier(n_estimators=300, random_state=26),
                     Xtr, Xte, ytr, yte, "RandomForest (uni)")

    train_and_report(LinearSVC(),
                     Xtr, Xte, ytr, yte, "Linear SVM (uni)")

# n-grams (1-3)
    Xtr3, Xte3, ytr3, yte3 = tfidf_split(df, (1, 3))
    train_and_report(LinearSVC(), Xtr3, Xte3, ytr3, yte3, "Linear SVM (1–3)")
    train_and_report(RandomForestClassifier(n_estimators=300, random_state=26),
                     Xtr3, Xte3, ytr3, yte3, "RandomForest (1–3)")
