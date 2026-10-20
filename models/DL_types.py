import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CLASSES = ['BENIGN', 'DDOS', 'ROOTKIT', 'RANSOMWARE', 'SPYWARE']
TASK = "types"
WINDOW_SIZE = 1000
N_RUNS = 6
SEED = 42


def load_data(base_dir, classes):
    windows, labels, capture_ids, run_ids = [], [], [], []
    for label, cls in enumerate(classes):
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


def evaluate(y_true, y_pred, capture_ids, run_ids, experiment, model_name, classes):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\nFold {experiment}: accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, zero_division=0, digits=4))
    print("Confusion matrix:", classes)
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))))
    summary = {"model": model_name, "fold": experiment, "accuracy": accuracy,
               "macro_f1": macro_f1, "weighted_f1": weighted_f1,
               "test_windows": len(y_true)}
    capture_rows, run_rows = [], []
    for ids, output in ((capture_ids, capture_rows), (run_ids, run_rows)):
        key = "capture" if ids is capture_ids else "run"
        for group in np.unique(ids):
            mask = ids == group
            correct = accuracy_score(y_true[mask], y_pred[mask])
            row = {"model": model_name, "fold": experiment, key: group,
                   "true_class": classes[int(y_true[mask][0])],
                   "windows": int(mask.sum()), "accuracy": correct}
            for idx, cls in enumerate(classes):
                row[f"pred_{cls.lower()}_fraction"] = float(np.mean(y_pred[mask] == idx))
            output.append(row)
    return summary, capture_rows, run_rows


def save_results(out_dir, model, task, summaries, capture_rows, run_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{model}_{task}"
    pd.DataFrame(summaries).to_csv(f"{prefix}_results.csv", index=False)
    pd.DataFrame(capture_rows).to_csv(f"{prefix}_by_capture.csv", index=False)
    pd.DataFrame(run_rows).to_csv(f"{prefix}_by_run.csv", index=False)


def get_args(models):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=models, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / TASK)
    return parser.parse_args()

import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, SimpleRNN, GRU

EPOCHS = 10
BATCH_SIZE = 64


def build_model(name, n_classes):
    if name == "cnn":
        layers = [Input(shape=(WINDOW_SIZE, 1)), Conv1D(32, 5, activation="relu"), MaxPooling1D(2),
                  Conv1D(64, 5, activation="relu"), MaxPooling1D(2),
                  Conv1D(128, 3, activation="relu"), MaxPooling1D(2),
                  GlobalAveragePooling1D(), Dense(128, activation="relu"), Dropout(0.5),
                  Dense(n_classes, activation="softmax")]
    elif name == "mlp":
        layers = [Input(shape=(WINDOW_SIZE,)), Dense(512, activation="relu"), Dropout(0.5),
                  Dense(256, activation="relu"), Dropout(0.5),
                  Dense(128, activation="relu"), Dense(n_classes, activation="softmax")]
    elif name == "rnn":
        layers = [Input(shape=(WINDOW_SIZE, 1)), SimpleRNN(64), Dense(128, activation="relu"),
                  Dropout(0.5), Dense(n_classes, activation="softmax")]
    elif name == "gru":
        layers = [Input(shape=(WINDOW_SIZE, 1)), GRU(64), Dense(128, activation="relu"),
                  Dropout(0.5), Dense(n_classes, activation="softmax")]
    else:
        raise ValueError(name)
    model = Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = get_args(["cnn", "mlp", "rnn", "gru"])
    classes = CLASSES
    X, y, captures, runs = load_data(Path(__file__).resolve().parent, classes)
    summaries, by_capture, by_run = [], [], []
    for fold in range(1, N_RUNS + 1):
        test_ids = {f"{cls}/R{fold:02d}" for cls in classes}
        test_mask = np.isin(runs, list(test_ids))
        train_mask = ~test_mask
        assert not (set(runs[train_mask]) & set(runs[test_mask]))
        assert len(np.unique(captures[test_mask])) == 2 * len(classes)
        assert len(np.unique(captures[train_mask])) == 10 * len(classes)
        X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
        y_train, y_test = y[train_mask], y[test_mask]
        mean = np.mean(X_train, dtype=np.float64)
        std = np.std(X_train, dtype=np.float64)
        if std == 0:
            std = 1.0
        X_train -= mean
        X_train /= std
        X_test -= mean
        X_test /= std
        if args.model != "mlp":
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
        tf.keras.backend.clear_session()
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        model = build_model(args.model, len(classes))
        print(f"\nTraining {args.model.upper()} fold {fold}/6: {len(X_train)} train, {len(X_test)} test", flush=True)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=1)
        probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        pred = np.argmax(probabilities, axis=1)
        result, cr, rr = evaluate(y_test, pred, captures[test_mask], runs[test_mask], fold, args.model, classes)
        summaries.append(result)
        by_capture.extend(cr)
        by_run.extend(rr)
        save_results(args.output_dir, args.model, TASK, summaries, by_capture, by_run)
        del model, X_train, X_test
        tf.keras.backend.clear_session()
        gc.collect()
    df = pd.DataFrame(summaries)
    print("\nFINAL RESULTS\n", df.to_string(index=False))
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        print(f"{metric}: {df[metric].mean():.4f} +/- {df[metric].std():.4f}")


if __name__ == "__main__":
    main()
