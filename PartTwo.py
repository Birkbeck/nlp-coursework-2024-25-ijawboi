
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

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

def train_and_report(clf, X_train, X_test, y_train, y_test, title=""):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"\n=== {title} ===")
    print("macro-F1:", f1_score(y_test, preds, average="macro").round(3))
    print(classification_report(y_test, preds))