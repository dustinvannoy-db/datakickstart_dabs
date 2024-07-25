import pytest
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.testing.utils import assertSchemaEqual

import os
import sys
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
parent_parent_dir =  os.path.dirname(parentdir)
print(parent_parent_dir)
sys.path.insert(0,parent_parent_dir)

from src.transforms_spark import remove_non_word_characters

spark = SparkSession.builder.getOrCreate()

def test_remove_non_word_characters_schema_check():
    source_data = [
        ("jo&&se",),
        ("**li**",),
        ("#::luisa",),
        (None,)
    ]
    source_df = spark.createDataFrame(source_data, ["name"])

    actual_df = source_df.withColumn(
        "clean_name",
        remove_non_word_characters(F.col("name"))
    )

    expected_data = [
        ("jo&&se", "jose"),
        ("**li**", "li"),
        ("#::luisa", "luisa"),
        (None, None)
    ]
    expected_df = spark.createDataFrame(expected_data, ["name", "clean_name"])

    assertSchemaEqual(actual_df.schema, expected_df.schema)

def test_df_col_order_schema_check():
  data = [
        (9, 9, 7),
        (-1, -2, -3)
    ]
  data2 = [
        (-1, -2, -3),
        (9, 9, 7)
    ]
  df1 = spark.createDataFrame(data, ["some_num", "neg_num"])
  df2 = spark.createDataFrame(data2, ["some_num", "neg_num"])
  assertSchemaEqual(df1.schema, df2.schema)

# # DataFrame equality messages peform schema comparisons before analyzing the actual content of the DataFrames. DataFrames that don't have the same schemas should error out as fast as possible. Let's compare a DataFrame that has a string column an integer column with a DataFrame that has two integer columns to observe the schema mismatch message.
# def test_datatype_mismatch():
#     data1 = [
#         (1, "a"),
#         (2, "b"),
#         (3, "c"),
#         (None, None)
#     ]
#     df1 = spark.createDataFrame(data1, ["num", "letter"])

#     data2 = [
#         (1, 6),
#         (2, 7),
#         (3, 8),
#         (None, None)
#     ]
#     df2 = spark.createDataFrame(data2, ["num", "num2"])

#     assertSchemaEqual(df1.schema, df2.schema)