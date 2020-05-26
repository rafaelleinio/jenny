from collections import namedtuple
from typing import List, Union

from pyspark.sql.functions import array, col, explode, lit, struct
from pyspark.sql import DataFrame

from jenny.services import dataframe_service
from jenny.quality.report import Report
from jenny.quality.check import Check

Validation = namedtuple("Validation", ["id", "description", "column_name", "check"])


class Validator:
    def __init__(self, validations: List[Validation] = None):
        self.validations = validations or []

    def add_validation(
        self, id: str, description: str, apply_on: Union[List[str], str], check: Check
    ):
        if not isinstance(check, Check):
            raise ValueError("check need to be an instance of Check.")

        new_column_names = (
            [apply_on] if isinstance(apply_on, str) else apply_on
        )
        new_validations = [
            Validation(id, description, column_name, check)
            for column_name in new_column_names
        ]
        return Validator(validations=self.validations + new_validations)

    def validate(self, input_df: DataFrame) -> Report:
        if not isinstance(input_df, DataFrame):
            raise ValueError("input_df should be an instance of Spark DataFrame.")

        self.converted_validations = self._convert_validations_columns(input_df)
        self.validations_df = self._create_validations_df(input_df)
        report_df = self._create_report_df()
        return Report(report_df)

    def _convert_validations_columns(self, input_df) -> List[Validation]:
        return list(
            set(
                Validation(
                    validation.id, validation.description, column_name, validation.check
                )
                for validation in self.validations
                for column_name in dataframe_service.get_matching_columns(
                    input_df, pattern=validation.column_name
                )
            )
        )

    def _create_validations_df(self, input_df: DataFrame) -> DataFrame:
        validation_columns = [
            struct(
                lit(validation.id).alias("id"),
                lit(validation.description).alias("description"),
                lit(validation.column_name).alias("from_column"),
                lit(
                    validation.check.apply(validation.column_name)._jc.toString()
                ).alias("metric"),
                validation.check.apply(validation.column_name).alias("is_success"),
            ).alias(f"{validation.id}_on_{validation.column_name}")
            for validation in self.converted_validations
        ]
        validations_df = input_df.agg(*validation_columns)
        return validations_df

    def _create_report_df(self) -> DataFrame:
        transposed_df = self.validations_df.withColumn(
            "validations", array(*self.validations_df)
        ).select(explode(col("validations")).alias("validations"))
        return transposed_df.select(
            transposed_df.validations.id.alias("id"),
            transposed_df.validations.description.alias("description"),
            transposed_df.validations.from_column.alias("from_column"),
            transposed_df.validations.metric.alias("metric"),
            transposed_df.validations.is_success.alias("is_success"),
        )
