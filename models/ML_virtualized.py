import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["VIRTUALIZED", "NOT_VIRTUALIZED"]
WINDOW_SIZE = 1000
N_RUNS = 6
SEED = 42


def load_data(base_dir):
    windows, labels, capture_ids, run_ids = [], [], [], []
    for label, cls in enumerate(CLASSES):
        for run in range(1, N_RUNS + 1):
            for cap in (1, 2):
                stem = f"R{run:02d}C{cap:02d}"
                candidates = [base_dir / cls / (stem + ext) for ext in (".TXT", ".txt")]
                path = next((p for p in candidates if p.is_file()), None)
                if path is None:
                    raise FileNotFoundError(f"Missing capture: {candidates[0]}")
                raw = pd.read_csv(path, usecols=["adc_raw"])["adc_raw"].to_numpy(dtype=np.float32)
                n = len(raw) // WINDOW_SIZE
                if n == 0:
                    raise ValueError(f"Capture has fewer than {WINDOW_SIZE} samples: {path}")
                x = raw[:n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)
                windows.append(x)
                labels.extend([label] * n)
                capture_ids.extend([f"{cls}/{stem}"] * n)
                run_ids.extend([f"{cls}/R{run:02d}"] * n)
                print(f"Loaded {cls}/{stem}: {n} windows", flush=True)
    return (np.concatenate(windows), np.asarray(labels, dtype=np.int8),
            np.asarray(capture_ids), np.asarray(run_ids))


def evaluate(y_true, y_pred, capture_ids, run_ids, experiment, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\nFold {experiment}: accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASSES, zero_division=0, digits=4))
    print("Confusion matrix [VIRTUALIZED, NOT_VIRTUALIZED]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    summary = {"model": model_name, "fold": experiment, "accuracy": accuracy,
               "macro_f1": macro_f1, "weighted_f1": weighted_f1,
               "test_windows": len(y_true)}
    capture_rows, run_rows = [], []
    for ids, output in ((capture_ids, capture_rows), (run_ids, run_rows)):
        key = "capture" if ids is capture_ids else "run"
        for group in np.unique(ids):
            mask = ids == group
            correct = accuracy_score(y_true[mask], y_pred[mask])
            fraction_not_virtualized = float(np.mean(y_pred[mask] == 1))
            output.append({"model": model_name, "fold": experiment, key: group,
                           "true_class": CLASSES[int(y_true[mask][0])],
                           "windows": int(mask.sum()), "accuracy": correct,
                           "pred_virtualized_fraction": 1 - fraction_not_virtualized,
                           "pred_not_virtualized_fraction": fraction_not_virtualized})
    return summary, capture_rows, run_rows


def save_results(out_dir, model, summaries, capture_rows, run_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{model}_virtualized"
    pd.DataFrame(summaries).to_csv(f"{prefix}_results.csv", index=False)
    pd.DataFrame(capture_rows).to_csv(f"{prefix}_by_capture.csv", index=False)
    pd.DataFrame(run_rows).to_csv(f"{prefix}_by_run.csv", index=False)


def save_confusion_matrix(cm, classes, out_dir, model_name, fold_name):
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.9), max(5, len(classes) * 0.8)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=False
    )
    ax.set_title(f"{model_name.upper()} - {fold_name}")
    fig.tight_layout()
    name = f"{model_name}_{fold_name}"
    fig.savefig(cm_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(cm_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def get_args(models):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=models, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_model(name):
    if name == "nb":
        return GaussianNB()
    if name == "svm":
        return LinearSVC(C=1.0, dual=False, max_iter=3000, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             objective="binary:logistic", eval_metric="logloss",
                             random_state=SEED, n_jobs=-1, tree_method="hist")
    raise ValueError(name)


def main():
    args = get_args(["nb", "svm", "rf", "xgboost"])
    X, y, captures, runs = load_data(args.data_dir)
    summaries, by_capture, by_run = [], [], []
    total_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for fold in range(1, N_RUNS + 1):
        test_ids = {f"{cls}/R{fold:02d}" for cls in CLASSES}
        test_mask = np.isin(runs, list(test_ids))
        train_mask = ~test_mask
        assert not (set(runs[train_mask]) & set(runs[test_mask]))
        assert len(np.unique(captures[test_mask])) == 4
        assert len(np.unique(captures[train_mask])) == 20
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        if args.model in ("nb", "svm"):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        print(f"\nTraining {args.model.upper()} fold {fold}/6: {len(X_train)} train, {len(X_test)} test", flush=True)
        model = build_model(args.model)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fold_cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASSES))))
        total_cm += fold_cm
        save_confusion_matrix(fold_cm, CLASSES, args.output_dir, args.model, f"fold_{fold}")
        result, cr, rr = evaluate(y_test, pred, captures[test_mask], runs[test_mask], fold, args.model)
        summaries.append(result)
        by_capture.extend(cr)
        by_run.extend(rr)
        save_results(args.output_dir, args.model, summaries, by_capture, by_run)
        del model, X_train, X_test
        gc.collect()
    save_confusion_matrix(total_cm, CLASSES, args.output_dir, args.model, "overall")
    df = pd.DataFrame(summaries)
    print("\nFINAL RESULTS\n", df.to_string(index=False))
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        print(f"{metric}: {df[metric].mean():.4f} +/- {df[metric].std():.4f}")


if __name__ == "__main__":
    main()
