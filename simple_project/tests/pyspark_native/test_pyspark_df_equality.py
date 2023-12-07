import pytest
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.testing.utils import assertDataFrameEqual

import os
import sys
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
parent_parent_dir =  os.path.dirname(parentdir)
print(parent_parent_dir)
sys.path.insert(0,parent_parent_dir)

from src.transforms_spark import remove_non_word_characters


spark = SparkSession.builder.getOrCreate()

def test_remove_non_word_characters_long():
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

    assertDataFrameEqual(actual_df, expected_df)

# def test_remove_non_word_characters_long_error():
#     source_data = [
#         ("matt7",),
#         ("bill&",),
#         ("isabela*",),
#         (None,)
#     ]
#     source_df = spark.createDataFrame(source_data, ["name"])

#     actual_df = source_df.withColumn(
#         "clean_name",
#         remove_non_word_characters(F.col("name"))
#     )

#     expected_data = [
#         ("matt7", "matt"),
#         ("bill&", "bill"),
#         ("isabela*", "isabela"),
#         (None, None)
#     ]
#     expected_df = spark.createDataFrame(expected_data, ["name", "clean_name"])

#     assertDataFrameEqual(actual_df, expected_df)

def test_df_equality_no_row_order():
  data = [[1], [2], [3]]
  df1 = spark.createDataFrame(data, ["some_num"])
  data2 = [[2], [1], [3]]
  df2 = spark.createDataFrame(data2, ["some_num"])
  assertDataFrameEqual(df1, df2, checkRowOrder=False)

def test_df_equality_with_row_order():
  data = [
        (2, 1, 3)
    ]
  df1 = spark.createDataFrame(data, ["some_num"])
         
  data2 = [
        (2, 1, 3)
    ]
  df2 = spark.createDataFrame(data2, ["some_num"])
  assertDataFrameEqual(df1, df2)

# def test_df_col_order():
#   data = [
#         (9, 9, 7),
#         (-1, -2, -3)
#     ]
#   data2 = [
#         (-1, -2, -3),
#         (9, 9, 7)
#     ]
#   df1 = spark.createDataFrame(data, ["some_num", "neg_num"])
#   df2 = spark.createDataFrame(data2, ["neg_num", "some_num"])
#   assertDataFrameEqual(df1, df2, checkRowOrder=True)

# # DataFrame equality messages peform schema comparisons before analyzing the actual content of the DataFrames. DataFrames that don't have the same schemas should error out as fast as possible. Let's compare a DataFrame that has a string column an integer column with a DataFrame that has two integer columns to observe the schema mismatch message.
# def test_schema_mismatch_message():
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

#     assertDataFrameEqual(df1, df2)