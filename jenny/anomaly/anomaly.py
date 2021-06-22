from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Optional, List, Dict

import pandas as pd
from prophet import Prophet
import numpy as np


# Errors
def calculate_ape(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(true - predicted) / true


@dataclass
class CrossValidationMetrics:
    pass


class CrossValidation:
    def __init__(
            self,
            model_kwargs_list: List[Dict[str, Any]],
            start_proportion: float = 0.25,
            slide: int = 1
    ):
        self.models = self._create_models(model_kwargs_list)
        self.start_proportion = start_proportion
        self.slide = slide

    def _create_models(self, model_kwargs_list):
        pass

    def run(self):
        pass


@dataclass
class TrainMetrics:
    train_time: int
    errors: np.ndarray
    start_ts: datetime
    end_ts: datetime


class ForecastModel(ABC):
    def __init__(self, model_kwargs: Dict[str, Any], ts_column: str, feature_column: str):
        self.model_kwargs = model_kwargs
        self.model = self._create_model()
        self.is_trained = False
        self.ts_column = ts_column
        self.feature_column = feature_column
        self.train_metrics: Optional[TrainMetrics] = None

    @abstractmethod
    def _create_model(self) -> Any:
        pass

    @abstractmethod
    def _train(self, train_pdf: pd.DataFrame):
        pass

    def _calculate_train_errors(self, train_pdf) -> np.ndarray:
        predicted = self.predict(periods=0, include_history=True)[self.feature_column].to_numpy()
        true = train_pdf[self.feature_column].to_numpy()
        return calculate_ape(true=true, predicted=predicted)

    def train(self, train_pdf: pd.DataFrame):
        train_start_ts = datetime.now()
        self._train(train_pdf)
        train_end_ts = datetime.now()
        self.is_trained = True
        return TrainMetrics(
            train_time=int((train_end_ts - train_start_ts).total_seconds() * 1000),
            errors=self._calculate_train_errors(train_pdf),
            start_ts=train_pdf[self.ts_column].min().to_pydatetime(),
            end_ts=train_pdf[self.ts_column].min().to_pydatetime()
        )

    @abstractmethod
    def _predict(self, ts_pdf: pd.DataFrame):
        pass

    def _make_ts_dataframe(self, periods: int, include_history: bool):
        start_ts = self.train_metrics.start_ts if include_history else self.train_metrics.end_ts
        periods = (
                self.train_metrics.start_ts - self.train_metrics.end_ts
        ).days if include_history else periods
        dates = pd.date_range(start=start_ts, periods=periods + 1)
        return pd.DataFrame({self.ts_column: dates})

    def predict(self, periods: int = 1, include_history=False):
        if not self.is_trained:
            raise RuntimeError("Model needs to be created before.")
        ts_pdf = self._make_ts_dataframe(periods, include_history)
        return self._predict(ts_pdf)


class ProphetForecastModel(ForecastModel):
    def __init__(self, model_kwargs: Dict[str, Any], ts_column: str, feature_column: str):
        super().__init__(model_kwargs, ts_column, feature_column)

    def _create_model(self) -> Prophet:
        return Prophet(**self.model_kwargs)

    def _train(self, train_pdf: pd.DataFrame):
        model: Prophet = self.model
        model.fit(train_pdf[[self.ts_column, self.feature_column]].rename(
            columns={self.ts_column: "ds", self.feature_column: "y"}
        ))

    def _predict(self, periods: int = 1):
        model: Prophet = self.model
        input_pdf = model.make_future_dataframe(periods=periods, include_history=False)
        output_pdf: pd.DataFrame = model.predict(input_pdf)
        return output_pdf[["ds", "yhat"]].rename(
            columns={"ds": self.ts_column, "yhat": self.feature_column}
        )
