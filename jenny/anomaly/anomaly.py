from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Set

import pandas as pd
from prophet import Prophet
import numpy as np


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

    def train(self, train_pdf: pd.DataFrame) -> TrainMetrics:
        train_start_ts = datetime.now()
        self._train(train_pdf)
        train_end_ts = datetime.now()
        self.is_trained = True
        return TrainMetrics(
            train_time=int((train_end_ts - train_start_ts).total_seconds() * 1000),
            errors=self._calculate_train_errors(train_pdf),
            start_ts=train_pdf[self.ts_column].min().to_pydatetime(),
            end_ts=train_pdf[self.ts_column].min().to_pydatetime(),
        )

    @abstractmethod
    def _predict(self, ts_pdf: pd.DataFrame) -> pd.DateFrame:
        pass

    def _make_ts_dataframe(self, periods: int, include_history: bool) -> pd.DataFrame:
        start_ts = self.train_metrics.start_ts if include_history else self.train_metrics.end_ts
        periods = (
            (self.train_metrics.start_ts - self.train_metrics.end_ts).days
            if include_history
            else periods
        )
        dates = pd.date_range(start=start_ts, periods=periods + 1)
        return pd.DataFrame({self.ts_column: dates})

    def predict(self, periods: int = 1, include_history=False) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model needs to be trained before.")
        ts_pdf = self._make_ts_dataframe(periods, include_history)
        return self._predict(ts_pdf)


class ProphetForecastModel(ForecastModel):
    def __init__(self, model_kwargs: Dict[str, Any], ts_column: str, feature_column: str):
        super().__init__(model_kwargs, ts_column, feature_column)

    def _create_model(self) -> Prophet:
        return Prophet(**self.model_kwargs)

    def _train(self, train_pdf: pd.DataFrame) -> None:
        model: Prophet = self.model
        model.fit(
            train_pdf[[self.ts_column, self.feature_column]].rename(
                columns={self.ts_column: "ds", self.feature_column: "y"}
            )
        )

    def _predict(self, periods: int = 1) -> pd.DataFrame:
        model: Prophet = self.model
        input_pdf = model.make_future_dataframe(periods=periods, include_history=False)
        output_pdf: pd.DataFrame = model.predict(input_pdf)
        return output_pdf[["ds", "yhat"]].rename(
            columns={"ds": self.ts_column, "yhat": self.feature_column}
        )


class ModelsConstructor:
    MODELS = {"prophet": ProphetForecastModel}

    def list_models(self) -> List[str]:
        return list(self.MODELS.keys())

    def create_model(
        self, name: str, model_kwargs: Dict[str, Any], ts_column: str, feature_column: str
    ) -> ForecastModel:
        if name not in self.list_models:
            raise ValueError(f"Model not defined, list of available models: {self.list_models}")
        model_class = self.MODELS[name]
        return model_class(
            model_kwargs=model_kwargs, ts_column=ts_column, feature_column=feature_column
        )


@dataclass
class ModelValidationSpec:
    name: str
    model_kwargs_list: List[Dict[str, Any]]

    def __repr__(self):
        return f"<{self.name} ModelValidationSpec>"

    def __eq__(self, other):
        if not isinstance(other, ModelValidationSpec):
            return False
        return other.name == self.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class CrossValidationMetrics:
    name: str
    timestamps: np.ndarray
    errors: np.ndarray
    errors_avg: float
    errors_stddev: float
    errors_max: float
    errors_min: float


@dataclass
class ModelEvaluationResult:
    name: str
    timestamp: datetime
    model_kwargs: Dict[str, Any]
    error: float


@dataclass
class ModelValidationResult:
    model_kwargs: Dict[str, Any]
    error: float
    # TODO eq and sort


# Errors
def calculate_ape(true: float, predicted: float) -> float:
    return np.abs(true - predicted) / true


class CrossValidation:
    def __init__(
        self,
        validation_specs: Set[ModelValidationSpec],
        ts_column: str,
        feature_column: str,
        start_proportion: float = 0.25,
        slide: int = 1,
        time_granularity: timedelta = timedelta(days=1)
    ):
        self.validation_specs = validation_specs
        self.ts_column = ts_column
        self.feature_column = feature_column
        self.start_proportion = start_proportion
        self.slide = slide
        self.time_granularity = time_granularity

    def _define_step_dates(self):
        pass

    def _create_model(self, name: str, model_kwargs: Dict[str, Any]) ->  ForecastModel:
        return ModelsConstructor().create_model(
            name=name,
            model_kwargs=model_kwargs,
            ts_column=self.ts_column,
            feature_column=self.feature_column,
        )

    def _filter_pdf(
            self, data_pdf, end_ts: datetime, start_ts: Optional[datetime] = None
    ) -> pd.DataFrame:
        pass

    def _calculate_error(self, predicted: pd.DataFrame, real: pd.DataFrame) -> float:
        pass

    def _validate_ts(self, name: str, data_pdf: pd.DataFrame, timestamp: datetime, validation_spec: ModelValidationSpec):
        validation_ts = timestamp - (self.time_granularity * 2)
        validation_train_pdf = self._filter_pdf(data_pdf, validation_ts)

        validation_results = []
        for model_kwargs in validation_spec.model_kwargs_list:
            model = self._create_model(name, model_kwargs)
            model.train(validation_train_pdf)
            predict_target_ts_pdf = model.predict()
            real_target_ts_pdf = self._filter_pdf(data_pdf, start_ts=validation_ts, end_ts=validation_ts)
            error = self._calculate_error(predicted=predict_target_ts_pdf, real=real_target_ts_pdf)
            validation_results.append(ModelValidationResult(error=error, model_kwargs=model_kwargs)
        return validation_results

    def _evaluate_ts(
            self,
            name: str,
            data_pdf: pd.DataFrame,
            timestamp: datetime,
            validation_spec: ModelValidationSpec
    ) -> ModelEvaluationResult:
        evaluation_ts = timestamp - self.time_granularity
        evaluation_train_pdf = self._filter_pdf(data_pdf, evaluation_ts)

        validation_results = self._validate_ts(name, data_pdf, timestamp, validation_spec)
        best_validation_result = sorted(validation_results).pop()


        return ModelEvaluationResult(name=, timestamp=, model_kwargs=, error=)

    def run(self, data_pdf: pd.DataFrame):
        sorted_data_pdf = data_pdf.sort_values(by=[self.ts_column])

        for validation_spec in self.validation_specs:
            pass
