import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

DATA_FILE = Path("data/processed/landmarks_full.csv")
MODEL_FILE = Path("models/isl_alphabet_rf.joblib")


df = pd.read_csv(DATA_FILE)

feature_cols = [
    col for col in df.columns
    if col.startswith(("h1_", "h2_"))
]

if df.empty:
    raise ValueError("Dataset is empty.")

if len(feature_cols) != 126:
    raise ValueError(f"Expected 126 features, got {len(feature_cols)}")

df[feature_cols] = df[feature_cols].fillna(0)

train = df[df["split"] == "Training"]
validation = df[df["split"] == "Validation"]
test = df[df["split"] == "Testing"]

if train.empty or validation.empty or test.empty:
    raise ValueError("One or more dataset splits are empty.")

X_train, y_train = train[feature_cols], train["label"]
X_val, y_val = validation[feature_cols], validation["label"]
X_test, y_test = test[feature_cols], test["label"]

classes = sorted(df["label"].unique())


# Compare configurations using validation data only.
configs = [
    {"n_estimators": 200, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 400, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 400, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 400, "min_samples_leaf": 1, "max_features": 0.5},
]

best_model = None
best_score = -1
best_config = None

print("\nComparing model configurations...")

for config in configs:
    model = RandomForestClassifier(
        **config,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)

    overall_f1 = f1_score(
        y_val, val_predictions,
        labels=classes, average="macro", zero_division=0
    )

    score = overall_f1

    print(
        f"{config} | "
        f"Accuracy: {accuracy_score(y_val, val_predictions):.2%} | "
        f"Overall macro-F1: {overall_f1:.3f}"
    )

    if score > best_score:
        best_score = score
        best_model = model
        best_config = config

print("\nSelected configuration:", best_config)

# Final evaluation on the untouched test split.
test_predictions = best_model.predict(X_test)

print(f"\nTest accuracy: {accuracy_score(y_test, test_predictions):.2%}")

print("\nTest classification report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=classes,
    zero_division=0
))

print("\nConfusion matrix:")
cm = confusion_matrix(y_test, test_predictions, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df)


MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "feature_columns": feature_cols,
        "classes": classes
    },
    MODEL_FILE
)

print(f"\nBest model saved to: {MODEL_FILE}")
