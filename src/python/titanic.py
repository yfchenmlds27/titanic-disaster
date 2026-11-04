"""Titanic survival prediction with logistic regression (Python)."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUT_PATH = os.path.join(DATA_DIR, "predictions_python.csv")

COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def extract_title(name_series):
    titles = name_series.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles.where(titles.isin(COMMON_TITLES), "Rare")


def fit_stats(df):
    """Learn imputation values from the TRAINING data only."""
    return {
        "age_by_title": df.groupby("Title")["Age"].median(),
        "age_overall": df["Age"].median(),
        "fare": df["Fare"].median(),
        "embarked": df["Embarked"].mode()[0],
    }


def transform(df, stats):
    df = df.copy()

    n = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Title"].map(stats["age_by_title"]))
    df["Age"] = df["Age"].fillna(stats["age_overall"])
    print(f"- Filled {n} missing Age values with the median age for each Title")

    n = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(stats["fare"])
    print(f"- Filled {n} missing Fare values with median fare ({stats['fare']:.2f})")

    n = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(stats["embarked"])
    print(f"- Filled {n} missing Embarked values with mode ('{stats['embarked']}')")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print("- Added FamilySize = SibSp + Parch + 1, and IsAlone = (FamilySize == 1)")

    df["Sex"] = (df["Sex"] == "female").astype(int)
    print("- Encoded Sex as 1 = female, 0 = male")

    df["LogFare"] = np.log1p(df["Fare"])
    print("- Added LogFare = log(1 + Fare) to reduce skew")

    df = pd.get_dummies(df, columns=["Pclass", "Embarked", "Title"], drop_first=True, dtype=int)
    print("- One-hot encoded Pclass, Embarked and Title")

    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Fare"], errors="ignore")
    print("- Dropped PassengerId, Name, Ticket, Cabin (mostly missing) and raw Fare")
    return df


def main():
    section("1. Loading training data")
    for path in (TRAIN_PATH, TEST_PATH):
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found. See the README for how to download the data.")
    train = pd.read_csv(TRAIN_PATH)
    print(f"Loaded train.csv: {train.shape[0]} rows, {train.shape[1]} columns")
    print(train.head())

    section("2. Exploring the data")
    missing = train.isna().sum()
    print("Missing values per column:")
    print(missing[missing > 0])
    print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
    print("\nSurvival rate by Sex:")
    print(train.groupby("Sex")["Survived"].mean().round(3))
    print("\nSurvival rate by Pclass:")
    print(train.groupby("Pclass")["Survived"].mean().round(3))

    section("3. Adjusting the data")
    train["Title"] = extract_title(train["Name"])
    print("- Extracted Title from Name (uncommon titles grouped as 'Rare'):")
    print(train["Title"].value_counts().to_string())
    stats = fit_stats(train)
    train_clean = transform(train, stats)
    X = train_clean.drop(columns=["Survived"])
    y = train_clean["Survived"]
    feature_cols = X.columns.tolist()
    print(f"\nFinal features ({len(feature_cols)}): {feature_cols}")

    section("4. Training logistic regression")
    # Kaggle's test.csv has no Survived labels, so we hold out 20% of
    # train.csv as a validation set to estimate out-of-sample accuracy.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split train.csv into {len(X_tr)} training rows and {len(X_val)} validation rows")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    print("Model fitted. Coefficients (largest effect first):")
    for col, coef in sorted(zip(feature_cols, model.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {col:15s} {coef:+.3f}")

    section("5. Accuracy on the training set")
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"Training accuracy:            {train_acc:.4f}")
    print(f"Held-out validation accuracy: {val_acc:.4f}")

    section("6. Loading test.csv and predicting")
    test = pd.read_csv(TEST_PATH)
    print(f"Loaded test.csv: {test.shape[0]} rows, {test.shape[1]} columns")
    test["Title"] = extract_title(test["Name"])
    print("Applying the same transformations (using training-set statistics):")
    test_clean = transform(test, stats)
    test_clean = test_clean.reindex(columns=feature_cols, fill_value=0)

    print("Refitting the model on all 891 training rows before predicting test.csv")
    model.fit(X, y)
    test_pred = model.predict(test_clean)
    print(f"Predicted survival for {len(test_pred)} passengers")
    print(f"Predicted survival rate on test set: {test_pred.mean():.3f}")

    section("7. Accuracy on the test set")
    if "Survived" in test.columns:
        print(f"Test accuracy: {accuracy_score(test['Survived'], test_pred):.4f}")
    else:
        print("test.csv has no 'Survived' column (Kaggle withholds the labels),")
        print("so true test accuracy cannot be computed locally.")
        print(f"Estimated test accuracy (held-out validation): {val_acc:.4f}")
        print("Submit the predictions file to Kaggle for the official test score.")

    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred}).to_csv(
        OUT_PATH, index=False
    )
    print("\nSaved predictions to src/data/predictions_python.csv")
    print("Done!")


if __name__ == "__main__":
    main()
