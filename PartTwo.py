
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "texts" / "speeches" / "hansard40000.csv"

def load_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    #(a.i)
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")

    #(a.ii)
    top4 = df["party"].value_counts().nlargest(4).index
    df = df[df["party"].isin(top4) & (df["party"] != "Speaker")]
