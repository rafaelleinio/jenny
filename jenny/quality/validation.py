from pyspark.sql.functions import lit, struct

class Validation:
    def __init__(self, id, description, columns, check):
        self.id = id
        self.description = description
        self.columns = columns
        self.check = check

    def _create_check_struct_column(self, column_name):
        check_column = self.check.apply(column_name)
        return struct(
            lit(self.id).alias("id"),
            lit(self.description).alias("description"),
            lit(column_name).alias("from_column"),
            lit(check_column._jc.toString()).alias("metric"),
            check_column.alias("is_success"),
        ).alias(f"{self.id}_on_{column_name}")

    def validate(self):
        return [
            self._create_check_struct_column(column_name)
            for column_name in self.columns
        ]
