from pyspark.sql import Column
from pyspark.sql.functions import col, min, isnull


def is_in_range(column: str, lower_bound: float, upper_bound: float) -> Column:
    return min(
        (col(column) >= lower_bound) & (col(column) <= upper_bound)
    )

def is_not_null(column: str):
    return min(~isnull(col(column)))
