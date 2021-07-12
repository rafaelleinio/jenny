
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Set

import pandas as pd
from prophet import Prophet
import numpy as np


def _make_ts_dataframe(periods: int, time_granularity: timedelta, include_history: bool) -> pd.DataFrame:
    self_train_metrics_start_ts = datetime(year=2021, month=7, day=10)
    self_train_metrics_end_ts = datetime(year=2021, month=7, day=11)
    self_ts_column = "ts"

    start_ts = self_train_metrics_start_ts if include_history else (self_train_metrics_end_ts + time_granularity)
    periods = (
        (self_train_metrics_start_ts - self_train_metrics_end_ts).days
        if include_history
        else periods
    )
    dates = pd.date_range(start=start_ts, periods=periods)
    return pd.DataFrame({self_ts_column: dates})


print(_make_ts_dataframe(periods=1, time_granularity=timedelta(days=1), include_history=False))
