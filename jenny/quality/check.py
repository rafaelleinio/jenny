from pyspark.sql import Column

from typing import Callable


class Check:
    def __init__(self, metric: Callable, *args, **kwargs):
        self.metric = metric
        self.args = args
        self.kwargs = kwargs

    def apply(self, column: str) -> Column:
        column = self.metric(column, *self.args, **self.kwargs)

        if not isinstance(column, Column):
            raise ValueError("Metric should return a Column instance.")

        return column.cast("boolean")
