# dataset builder
from jenny.dataset_generation import BatchDatasetGenerator

# feature generators
from jenny.dataset_generation import (
    IdFeatureGenerator,
    TimestampFeatureGenerator,
    NumericFeatureGenerator,
    TimeSensitiveNumericFeatureGenerator,
    TextFeatureGenerator,
    CategoryFeatureGenerator,
)

# time series modeling
from jenny.dataset_generation import (
    Trend,
    Seasonality,
    Noise,
    NormalPercentageDeviation,
)

# profiling
from jenny.profiling import (
    ProfilingPipeline,
    GlobalProfiler,
    ColumnStatsProfiler,
)


dataset_generator = BatchDatasetGenerator(
    events_generator=TimeSensitiveNumericFeatureGenerator(
        name="events",
        trend=Trend(
            base_value=100,
            slope=1,
        ),
        seasonality=Seasonality(
            week_days=[0.85, 0.9, 1, 1, 0.9, 0.8, 0.8],
            month_period=[1, 0.9, 1],
            year_months=[1, 0.95, 0.9, 0.9, 0.85, 0.85, 0.85, 0.85, 0.9, 0.9, 0.95, 1],
        ),
        noise=Noise(
            var=0.05,
            seed=123,
        ),
    ),
    features=[
        IdFeatureGenerator(name="id", min_id=1000, monotonically_increase=True),
        TimestampFeatureGenerator(name="ts"),
        NumericFeatureGenerator(
            name="numeric_feature",
            base_value=500,
            percentage_deviation_generator=NormalPercentageDeviation(
                var=0.8,
                seed=123,
            ),
        ),
        TimeSensitiveNumericFeatureGenerator(
            name="numeric_feature_with_trend",
            trend=Trend(
                base_value=500,
                slope=10,
            ),
            noise=Noise(
                var=0.05,
                seed=123,
            ),
        ),
        TimeSensitiveNumericFeatureGenerator(
            name="numeric_feature_with_seasonality",
            trend=Trend(
                base_value=500,
                slope=10,
            ),
            noise=Noise(
                var=0.05,
                seed=123,
            ),
            seasonality=Seasonality(
                week_days=[0.9, 1.1, 1.1, 1, 0.9, 0.9, 0.9],
                year_months=[1.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2],
            ),
        ),
        TextFeatureGenerator(
            name="description",
            max_base_length=120,
        ),
        CategoryFeatureGenerator(name="category", categories=["A", "B", "C", "D", "E"]),
    ],
)

from datetime import datetime

start_ts = datetime(year=2018, month=1, day=1)
df = dataset_generator.generate(start_ts=start_ts, n=800)

metrics = ["max", "min", "sum", "avg"]
profiling_pipeline = ProfilingPipeline(
    profilers=[
        GlobalProfiler(),
        ColumnStatsProfiler(from_column="id", metrics=metrics),
        ColumnStatsProfiler(from_column="numeric_feature", metrics=metrics),
        ColumnStatsProfiler(from_column="numeric_feature_with_trend", metrics=metrics),
        ColumnStatsProfiler(from_column="numeric_feature_with_seasonality", metrics=metrics),
        ColumnStatsProfiler(from_column="description", metrics=metrics, not_numeric=True),
        ColumnStatsProfiler(from_column="category", metrics=metrics, not_numeric=True),
    ]
)
