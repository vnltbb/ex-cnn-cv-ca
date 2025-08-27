import os
import time
import csv
import pandas as pd
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from typing import Optional

class EpochTimeLogger(Callback):
    """
    - 각 epoch의 경과 시간을 기록하고 학습 종료 시 csv로 저장
    - history csv가 있으면 'epoch_time_s' 컬럼으로 병합
    """
    def __init__(self, save_dir: str, history_csv: Optional[str] = None):
        super().__init__()
        self.save_dir = save_dir
        self.history_csv = history_csv
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self._epoch_start)

    def on_train_end(self, logs=None):
        total = time.time() - self.train_start

        # 1) epoch_times.csv 별도 저장
        out_path = os.path.join(self.save_dir, 'epoch_times.csv')
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'seconds'])
            for i, s in enumerate(self.epoch_times):
                w.writerow([i, s])

        # 2) history csv에 병합(있을 때만)
        if self.history_csv is not None and os.path.exists(self.history_csv):
            try:
                df = pd.read_csv(self.history_csv)
                # CSVLogger는 기본적으로 'epoch' 컬럼이 존재
                # 길이가 맞으면 그대로 병합
                if len(df) == len(self.epoch_times):
                    df['epoch_time_s'] = self.epoch_times
                else:
                    # 조기 종료 등으로 길이가 다르면 epoch 기준 안전 병합
                    df_times = pd.DataFrame({'epoch': list(range(len(self.epoch_times))),
                                            'epoch_time_s': self.epoch_times})
                    df = df.merge(df_times, on='epoch', how='left')
                df.to_csv(self.history_csv, index=False)
            except Exception as e:
                print(f'[EpochTimeLogger] history 병합 생략(에러): {e}')

        # 콘솔 출력
        avg = sum(self.epoch_times) / max(1, len(self.epoch_times))
        print(f'[Time] total={total:.1f}s, per-epoch avg={avg:.2f}s')


def get_callbacks(model_name, save_dir, patience):
    
    early = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'{save_dir}/{model_name}.keras', save_best_only=True)
    redu = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=patience//2, min_lr=1e-6)
    
    history_csv_path = f'{save_dir}/{model_name}.csv'
    log = CSVLogger(history_csv_path, append=True)
    
    time_logger = EpochTimeLogger(save_dir=save_dir, history_csv=history_csv_path)
    
    return [checkpoint, early, redu, log, time_logger]
