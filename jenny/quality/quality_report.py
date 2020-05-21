from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.functions import array, col, explode

from jenny.quality.validation import Validation


class QualityReport:
    def __init__(self, validations, dataframe=None):
        self.validations = validations
        self.dataframe = dataframe

    def _create_validations_df(self, input_df) -> DataFrame:
        all_validations = sum([v.validate() for v in self.validations], [])
        validations_df = input_df.agg(*all_validations)
        return validations_df

    def _create_report_df(self, validations_df) -> DataFrame:
        transposed_df = validations_df.withColumn("validations", array(*validations_df)).select(
            explode(col("validations")).alias("validations")
        )
        return transposed_df.select(
            transposed_df.validations.id.alias("id"),
            transposed_df.validations.description.alias("description"),
            transposed_df.validations.from_column.alias("from_column"),
            transposed_df.validations.metric.alias("metric"),
            transposed_df.validations.is_success.alias("is_success"),
        )

    def _check_dataframe(self, dataframe):
        if not dataframe:
            raise ValueError("Please input the Spark dataframe to be validated.")
        if not isinstance(dataframe, DataFrame):
            raise ValueError("input_df should be an instance of Spark dataFrame.")

    @property
    def validations(self) -> List[Validation]:
        return self._validations

    @validations.setter
    def validations(self, value: List[Validation]):
        if (not isinstance(value, List)) and not (
            all(isinstance(item, Validation) for item in value)
        ):
            raise ValueError("Validations should be a list of Validation objects.")
        self._validations = value

    @property
    def dataframe(self) -> DataFrame:
        return self._dataframe

    @dataframe.setter
    def dataframe(self, value: DataFrame):
        self._dataframe = value
        if not self._dataframe:
            return
        self._check_dataframe(value)
        validations_df = self._create_validations_df(value)
        self._report_df = self._create_report_df(validations_df)
        self._report_df.cache()

    def input(self, dataframe):
        self._check_dataframe(dataframe)
        return QualityReport(validations=self.validations, dataframe=dataframe)

    def get_dataframe_report(self) -> DataFrame:
        self._check_dataframe(self.dataframe)
        return self._report_df

    def get_dict_report(self) -> List[dict]:
        return [row.asDict() for row in self.get_dataframe_report().collect()]

    def get_quality_coverage(self) -> float:
        report = self.get_dict_report()
        successes = [
            True for validation in self.get_dict_report() if validation["is_success"]
        ]
        return len(successes) / len(report)

    def has_full_coverage(self) -> bool:
        return True if self.get_quality_coverage() == 1 else False
