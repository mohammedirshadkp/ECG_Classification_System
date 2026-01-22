import os
import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
from utils import log


class ECGDataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.X_train_flat = None
        self.X_test_flat = None
        self.X_train_cnn = None
        self.X_test_cnn = None
        self.y_train = None
        self.y_test = None

    def _map_label(self, scp_codes):
        if isinstance(scp_codes, str):
            try:
                scp_codes = eval(scp_codes)
            except:
                return None
        codes = set(scp_codes.keys())
        if "NORM" in codes:
            return "NORM"
        if any(x in codes for x in ["MI", "AMI", "IMI"]):
            return "MI"
        return None

    def load_and_process(self):
        log("Loading PTB-XL metadata")
        df = pd.read_csv(config.CSV_PATH)
        df["label"] = df["scp_codes"].apply(self._map_label)
        df = df.dropna(subset=["label"])
        df = df[df["label"].isin(["NORM", "MI"])]
        df["label_id"] = df["label"].map(config.LABEL_MAP)

        # Balance data by sampling equal NORM and MI 
        n_norm = min(config.SAMPLES_PER_CLASS, len(df[df["label"] == "NORM"]))
        n_mi = min(config.SAMPLES_PER_CLASS, len(df[df["label"] == "MI"]))
        df = pd.concat([
            df[df["label"] == "NORM"].sample(n_norm, random_state=config.RANDOM_STATE),
            df[df["label"] == "MI"].sample(n_mi, random_state=config.RANDOM_STATE)
        ]).sample(frac=1, random_state=config.RANDOM_STATE)

        log(f"Total ECG records used: {len(df)}")

        signals, labels = [], []
        for _, row in df.iterrows():
            try:
                record_path = os.path.join(config.DATA_DIR, row["filename_lr"])
                signal, _ = wfdb.rdsamp(record_path)
                ecg = signal[:config.MAX_SAMPLES, 0]
                if len(ecg) < config.MAX_SAMPLES:
                    ecg = np.pad(ecg, (0, config.MAX_SAMPLES - len(ecg)), mode="constant")
                signals.append(ecg)
                labels.append(row["label_id"])
            except:
                continue

        X = np.array(signals)
        y = np.array(labels)

        # Train-test split on raw signals first
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )

        # Scale using train statistics only
        X_train_flat = self.scaler.fit_transform(X_train)
        X_test_flat = self.scaler.transform(X_test)

        X_train_cnn = X_train_flat[..., np.newaxis]
        X_test_cnn = X_test_flat[..., np.newaxis]

        self.X_train_flat = X_train_flat
        self.X_test_flat = X_test_flat
        self.X_train_cnn = X_train_cnn
        self.X_test_cnn = X_test_cnn
        self.y_train = y_train
        self.y_test = y_test

        return self
