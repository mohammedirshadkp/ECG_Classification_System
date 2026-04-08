import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
from utils import log

class ECGDataLoader:
    def __init__(self):
        self.scaler = StandardScaler()

    def _apply_filter(self, data, lowcut=0.5, highcut=40.0, fs=100.0, order=3):
        # Bandpass filter to remove baseline wander and noise
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def load_and_process(self):
        log("Loading PTB-XL metadata and SCP statements...")
        df = pd.read_csv(config.CSV_PATH, index_col='ecg_id')
        agg_df = pd.read_csv(config.SCP_PATH, index_col=0)
        
        # FIX: Dynamically find the diagnostic column (it varies by version)
        diag_col = None
        for col in ['diagnostic_class', 'diagnostic_superclass', 'diagnostic']:
            if col in agg_df.columns:
                diag_col = col
                break
        
        if diag_col is None:
            raise ValueError(f"Could not find diagnostic column in {config.SCP_PATH}. Columns found: {agg_df.columns}")

        log(f"Using diagnostic column: {diag_col}")
        agg_df = agg_df[agg_df[diag_col].notnull()]

        def aggregate_diagnostic(y_dic):
            if isinstance(y_dic, str):
                y_dic = eval(y_dic)
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key][diag_col])
            tmp = list(set(tmp))
            return tmp[0] if len(tmp) > 0 else None

        df['label'] = df.scp_codes.apply(aggregate_diagnostic)
        df = df.dropna(subset=['label'])
        df = df[df['label'].isin(config.LABEL_MAP.keys())]
        df['label_id'] = df['label'].map(config.LABEL_MAP)

        log(f"Total valid records: {len(df)}. Loading 12-lead signals...")
        
        # Limit for prototype speed
        df = df.sample(n=min(config.TOTAL_RECORDS_TO_LOAD, len(df)), random_state=config.RANDOM_STATE)

        # Pre-allocate to avoid double memory spike from list->array conversion
        n = len(df)
        X = np.zeros((n, config.MAX_SAMPLES, config.CHANNELS), dtype=np.float32)
        y = np.zeros(n, dtype=np.int32)
        count = 0
        for _, row in df.iterrows():
            try:
                record_path = os.path.join(config.DATA_DIR, row["filename_lr"])
                signal, _ = wfdb.rdsamp(record_path)
                if signal.shape[1] < 12:
                    continue
                ecg = signal[:config.MAX_SAMPLES, :]
                if len(ecg) == config.MAX_SAMPLES:
                    X[count] = self._apply_filter(ecg).astype(np.float32)
                    y[count] = row["label_id"]
                    count += 1
            except Exception:
                continue

        X = X[:count]
        y = y[:count]

        if len(X) == 0:
            raise ValueError("No signals were loaded. Check your data paths.")

        log(f"Data shape: {X.shape}. Performing stratified split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )

        # Scale 3D array
        X_train_shape = X_train.shape
        X_test_shape = X_test.shape
        
        X_train_flat = X_train.reshape(-1, config.CHANNELS)
        X_test_flat = X_test.reshape(-1, config.CHANNELS)
        
        X_train_scaled = self.scaler.fit_transform(X_train_flat).reshape(X_train_shape)
        X_test_scaled = self.scaler.transform(X_test_flat).reshape(X_test_shape)

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

        return self