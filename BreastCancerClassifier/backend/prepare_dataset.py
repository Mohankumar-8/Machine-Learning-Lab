"""Step 1 - Create the project dataset (train / test CSV files).

Loads the WDBC dataset bundled with scikit-learn, and performs a stratified
80/20 split so both classes appear in the same proportion in train and test.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from config import TRAIN_CSV, TEST_CSV, RANDOM_STATE


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()                       # 30 features + 'target'
    df = df.drop_duplicates().reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_STATE)

    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"Total samples : {len(df)}")
    print(f"Train samples : {len(train_df)}  -> {TRAIN_CSV}")
    print(f"Test samples  : {len(test_df)}  -> {TEST_CSV}")
    print("Class balance (train):")
    print(train_df['target'].map({0: 'Malignant', 1: 'Benign'}).value_counts())


if __name__ == "__main__":
    main()
