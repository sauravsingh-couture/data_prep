import warnings
warnings.filterwarnings("ignore")
from typing import Callable, Iterator, Optional, Sequence, List, TypeVar, Generic, Sized, Any, Dict, Tuple, Union
from darts.models import NaiveMean, TSMixerModel
import torch

from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers.scaler import Scaler
from darts.metrics.metrics import rmse
from darts.models import TSMixerModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.data.inference_dataset import GenericInferenceDataset
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.model_selection import train_test_split

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import root_mean_squared_error

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path 
import pickle
import shutil
import pyarrow as pa
import pyarrow.hdfs as hdfs
import pyarrow.parquet as pq
fs = pa.hdfs.connect()
from darts import TimeSeries
from tqdm.notebook import tqdm

def give_previous_date(date, week_delta):
    from datetime import datetime, timedelta
    current_date = datetime.strptime(date, '%Y-%m-%d')
    prev_date = current_date - timedelta(weeks=week_delta)
    prev_date_str = prev_date.strftime('%Y-%m-%d')
    return prev_date

def cross_join(df1, df2):
    df1["key"] = 1
    df2["key"] = 1

    result = pd.merge(df1, df2, on="key").drop("key", axis=1)
    return result

def load_data_parquet(path):
    data = pq.ParquetDataset(path, fs).read().to_pandas()
    return data

def split_series(series, train_start: str, train_end: str, test_start: str, test_end: str):
    train, test = [], []
    for series_ in tqdm(series, "Splitting Series"):
        train.append(series_.slice(pd.Timestamp(train_start), pd.Timestamp(train_end)))
        test.append(series_.slice(pd.Timestamp(test_start), pd.Timestamp(test_end)))
        
    return train, test

input_chunk_length = 52
output_chunk_length = 13
use_static_covariates = False
# max_samples_per_ts = 14
num_loader_workers = 32
full_training = True

model_name="TSMixer"
data_version = "3.0"
exp_name = f"TSMixerModel_icl={input_chunk_length}"

base_input_dir = Path("/app/notebooks/Saurav/electronics/DL_exp/")
input_dir = base_input_dir / f"dataset/version={data_version}"
output_dir = input_dir / "experiments" / exp_name

output_dir.mkdir(parents=True, exist_ok=True)
shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=False)
work_dir = output_dir.as_posix()

df_sales_rd = load_data_parquet(f"/data/Archive/Saurav/digital/phase1/forecasting_data/version={data_version}/")
print(df_sales_rd.shape)
print(df_sales_rd.columns)

df_sales_rd_c1 = df_sales_rd.copy()
df_sales_rd_c1.head()
df_sales_rd_c1['date'] = pd.to_datetime(df_sales_rd_c1['date'])
print(df_sales_rd_c1.date.min(), df_sales_rd_c1.date.max())

df_sales_rd_c1['year'] = df_sales_rd_c1['date'].dt.year
df_sales_rd_c1['month'] = df_sales_rd_c1['date'].dt.month

past_covariates_columns = ['total_avg_discount']
future_covariates_columns = ['year', 'month']

ts = TimeSeries.from_group_dataframe(df_sales_rd_c1, time_col='date', value_cols=['total_quantity'], freq='W-MON', fill_missing_dates=True,
                                     group_cols=['article_id','pin_code', 'channel'], n_jobs=-1,verbose=True)

past_covariates_ts = TimeSeries.from_group_dataframe(df_sales_rd_c1, time_col='date', value_cols=past_covariates_columns, freq='W-MON', fill_missing_dates=True,
                                     group_cols=['article_id','pin_code', 'channel'], n_jobs=-1,verbose=True)

future_covariates_ts = TimeSeries.from_group_dataframe(df_sales_rd_c1, time_col='date', value_cols=future_covariates_columns, freq='W-MON', fill_missing_dates=True,
                                     group_cols=['article_id','pin_code', 'channel'], n_jobs=-1,verbose=True)

train_start, train_end, test_start, test_end = "2022-08-08", "2024-03-31", "2024-04-01", "2024-06-30"

train, val = split_series(ts, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,)

past_covariates_train, past_covariates_val = split_series(past_covariates_ts, train_start=train_start, train_end=train_end, 
                                                          test_start=test_start, test_end=test_end,)

future_covariates_train, future_covariates_val = split_series(future_covariates_ts, train_start=train_start, train_end=train_end, 
                                                              test_start=test_start, test_end=test_end,)
future_covariates_train_prediction_time, future_covariates_val_prediction_time = split_series(future_covariates_ts, train_start=train_start, train_end=train_end, 
                                                              test_start=give_previous_date(test_start, input_chunk_length), test_end=test_end,)

print(train[1].time_index, val[1].time_index)

def create_model_params(input_chunk_length: int, output_chunk_length: int, full_training: bool = False, 
                        work_dir=None,) -> Dict: 
    
    
    early_stopper = EarlyStopping(monitor="train_loss", patience=5, min_delta=1e-3, mode="min",)
    if full_training:
        limit_train_batches = None
        limit_val_batches = None
        max_epochs = 20
        batch_size = 128
    else:
        limit_train_batches = 1
        limit_val_batches = 1
        max_epochs = 1
        batch_size = 512
        
    progress_bar = TFMProgressBar(
        enable_sanity_check_bar=True, 
        enable_validation_bar=True,
    )
    
    pl_trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_epochs": max_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "accelerator": "auto",
        "callbacks": [early_stopper, progress_bar],
    }
    
    optimizer_cls = torch.optim.AdamW
    optimizer_kwargs = {
        "lr": 0.0030280145724043794,  # 3e-3
        "weight_decay": 6.194971275031679e-05,  # 3e-3
    }
    
    lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    lr_scheduler_kwargs = {"gamma": 0.8}  # 0.9
    # lr_scheduler_cls = torch.optim.lr_scheduler.OneCycleLR
    # lr_scheduler_kwargs = {
    #     "epochs": max_epochs,
    #     "steps_per_epoch": 100, ## TODO: Drive this
    #     "max_lr": 1e-3,
    # }
    
    likelihood = None  # None, QuantileRegression()
    loss_fn = torch.nn.MSELoss(reduction="mean")  # None, torch.nn.MSELoss(reduction="mean") (mean, sum, none)
    
    add_encoders = {
        "cyclic": {"past": ["month"], "future": ["month"]},
        "position": {"past": ["relative"], "future": ["relative"]},
    }
    
    model_params = {
        "ff_size": 32,
        "hidden_size": 128,
        "num_blocks": 4,
        "activation": "ReLU",
        "dropout": 0.25,
        "normalize_before": False,
    }
    
    return {
        **model_params,
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "use_reversible_instance_norm": True,
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "likelihood": likelihood,
        "loss_fn": loss_fn,
        "save_checkpoints": True,
        "force_reset": True,
        "batch_size": batch_size,
        "random_state": 42,
        "add_encoders": add_encoders,
        "work_dir": work_dir,
    }

model = TSMixerModel(**create_model_params(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, 
                                           full_training=full_training, work_dir=output_dir.as_posix(),),
                     use_static_covariates=use_static_covariates, model_name=model_name,)

model.fit(
    series=train,
    past_covariates=past_covariates_train,
    future_covariates=future_covariates_train,
    # val_series=val,
    # val_past_covariates=past_covariates_val,
    # val_future_covariates=future_covariates_val,
    verbose=True,
    # max_samples_per_ts=max_samples_per_ts,
    num_loader_workers=num_loader_workers,
)
# model = model.load_from_checkpoint(model.model_name, model.work_dir)

val_preds = model.predict(
    n=13,
    series=train,
    past_covariates=past_covariates_train,
    future_covariates=future_covariates_val_prediction_time,
    batch_size=1024,
    n_jobs=-1,
    num_samples=50,
    num_loader_workers=8,
    mc_dropout=True,
    verbose=True,
)

date_range = pd.date_range(start=test_start, end=test_end, freq='W-MON')

date = pd.DataFrame({'date': date_range, 
                     'step': range(1, len(date_range) + 1)
                    })

predicted_df = []
for series_ in tqdm(val_preds, desc="Processing Prediction"):
    # print(series_.static_covariates.reset_index(drop=True), series_.with_values)
    # pred = pd.DataFrame(series_.quantile_df(0.5).reset_index(), columns=['prediction'])
    pred = pd.DataFrame(list(series_.quantile_df(.5).reset_index()['total_quantity_0.5']), columns=['prediction'])
    prediction_ = cross_join(series_.static_covariates.reset_index(drop=True), pred)
    prediction_ = pd.concat([prediction_, date], axis = 1)
    predicted_df.append(prediction_)
                

predicted_df = pd.concat(predicted_df)
predicted_df["prediction"] = predicted_df["prediction"].clip(lower=0)
predicted_df["prediction"] =np.ceil(predicted_df["prediction"])
predicted_df["algorithm"] = model_name
predicted_df["date"] = test_start

predicted_df.to_csv(f"{work_dir}/result.csv", index=False)