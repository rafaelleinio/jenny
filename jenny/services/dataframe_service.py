from typing import List

from pyspark.sql import DataFrame


def match_column_pattern(column_name: str, pattern: str) -> bool:
    """Verify if the column name matches the pattern.

    Args:
        pattern: string pattern to use.
        column: column names to try match with the pattern.

    Returns:
        True for a column that matches the pattern, False otherwise.

    """
    negate = False
    if pattern.startswith("!"):
        negate = True
        pattern = pattern[1:]
    split = pattern.split("*")
    matches_pattern = column_name.startswith(split[0]) and column_name.endswith(split[-1])
    return matches_pattern if not negate else not matches_pattern


def get_matching_columns(df: DataFrame, pattern: str) -> List[str]:
    matching_columns = [
        column_name
        for column_name in df.columns
        if match_column_pattern(column_name, pattern)
    ]
    if not matching_columns:
        raise ValueError(
            f'The name pattern "{pattern}" don\'t match any columns in df: '
            f"{df.columns}"
        )
    return matching_columns
