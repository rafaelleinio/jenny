from typing import List

from pyspark.sql import DataFrame


class Report:
    def __init__(self, report_df):
        self.report_df = report_df
        self.report_df.cache()

    @property
    def as_df(self) -> DataFrame:
        return self.report_df

    @property
    def as_json(self) -> List[dict]:
        return [row.asDict() for row in self.as_df.collect()]

    @property
    def coverage(self) -> float:
        report = self.as_json
        successes = [True for validation in report if validation["is_success"]]
        return len(successes) / len(report)

    def has_full_coverage(self) -> bool:
        return True if self.coverage == 1 else False
