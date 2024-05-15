# Databricks notebook source
# MAGIC %md
# MAGIC # Unit tests in PySpark 
# MAGIC
# MAGIC ---
# MAGIC <img align="right" src="https://media3.giphy.com/media/fRYeEj3DtgrW9VpVWs/giphy.gif?cid=ecf05e47cdpuxkms9stljb9g0axvahcqt1rr5ewhzfrpxvsm&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="250" height="300"> 
# MAGIC
# MAGIC ___
# MAGIC
# MAGIC #### Learning objectives:
# MAGIC * Understand the importance of unit tests  
# MAGIC * How to structure a Python project to use `pytest`
# MAGIC * How to test PySpark code with native PySpark testing functions
# MAGIC
# MAGIC ___
# MAGIC
# MAGIC Testing is a crucial part of software development that helps in identifying issues with your code early and often, thereby ensuring the reliability and robustness of the application. Unit tests are designed to test the independent pieces of logic that comprise your application. In general, tests look to validate that your logic is functioning as intended. **By asserting that the actual output of our logic is identical to the expected output, we can determine if the logic has been implemented correctly.** Ideally each test will cover exactly one piece of functionality, e.g., a specific data transformation or helper function.
# MAGIC
# MAGIC ___
# MAGIC
# MAGIC  
# MAGIC  In the context of PySpark, tests are usually centered around **comparing DataFrames** for expected output.  There are several dimensions by which DataFrames can be compared:
# MAGIC  <br><br>
# MAGIC  
# MAGIC  * Schemas
# MAGIC  * Columns
# MAGIC  * Rows 
# MAGIC  * Entire DataFrames
# MAGIC
# MAGIC In this notebook, we are going to learn about unit testing PySpark applications using the Python `pytest` framework. Tests will be written using the popular [`chispa`](https://github.com/MrPowers/chispa) library as well as the new [PySpark native testing functions available in Spark 3.5](https://issues.apache.org/jira/browse/SPARK-44042). This tutorial is designed to keep it simple, so if you want to learn more about how `pytest` works we recommend taking a closer look at [the official documentation](https://docs.pytest.org/en/7.4.x/).
# MAGIC
# MAGIC Let's get started!
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Working with `pytest`

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to structure your project with pytest
# MAGIC Consider the following hypothetical project structure for a simple reporting use case:
# MAGIC
# MAGIC ```
# MAGIC ├── notebooks/
# MAGIC │   ├── analysis.ipynb   # report with visualizations
# MAGIC ├── src/
# MAGIC │   ├── load_config.py   # helper functions
# MAGIC │   └── cleaning.py
# MAGIC ├── tests/
# MAGIC │   ├── main_test.py     # unit tests
# MAGIC ├── requirements.txt     # dependencies
# MAGIC └── test-requirements.txt
# MAGIC ```
# MAGIC
# MAGIC The `notebooks/` folder contains `analysis.ipynb`, which reports on and visualizes some business data.  Assume that this notebook imports custom Python functions from both files in `src/` to help load and clean data like so:
# MAGIC
# MAGIC ```
# MAGIC from src.load_config import db_loader, config_handler
# MAGIC from src.cleaning import *
# MAGIC ```
# MAGIC
# MAGIC Since our report depends on these functions to get and prepare the data correctly, we want to write tests that validate our functions are behaving as intended.  To so do, we create a `tests/` folder and include our tests there.  **`pytest` is designed to look for folders and files with "test" as a prefix or suffix.**  In this example our test script is called `main_test.py`, but later on you will see examples like `test_pyspark_column_equality.py`.  Both are supported and will be picked up by `pytest`. 
# MAGIC
# MAGIC The last two files in our project are specify any dependencies for all of our code.  It is a best practice to separate testing dependencies, since we only need `pytest` to run the testing scripts.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Installing dependencies
# MAGIC To use `pytest` we will need to install it alongside any other dependencies, then restart the Python interpreter to make them available in our environment.

# COMMAND ----------

!cp ../requirements.txt ~/.
%pip install -r ~/requirements.txt
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Invoking `pytest`
# MAGIC
# MAGIC `pytest` is usually run from the system command line, but it can also be executed from the context of a notebook or Python REPL.  We'll be using the latter method to invoke our tests from the Databricks editor.  This lets us make use of Spark and other configuration variables in Databricks Runtime. 
# MAGIC
# MAGIC One limitation of this approach is that changes to the test will be cached by Python's import caching mechanism.  If we wanted to iterate on tests during a development scenario, we would need to use `dbutils.library.restartPython()` to clear the cache and pick up changes to our tests.  This tutorial has been structured to render this unnecessary, but it is important to note!
# MAGIC
# MAGIC In the following cell, we first make sure that all tests will run relative to our repository root directory.  Then we define `run_pytest`, a helper function to invoke a specific test file in our project.  Importantly, **this function also fails the Databricks notebook cell execution if tests fail.**  This ensures we surface errors whether we run these unit tests in an interactive session or as part of a Databricks Workflow.

# COMMAND ----------

import pytest
import os
import sys

# Run all tests in the repository root.
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = os.path.dirname(os.path.dirname(notebook_path))
os.chdir(f'/Workspace/{repo_root}')
# %pwd

def run_pytest(pytest_path):
  # Skip writing pyc files on a readonly filesystem.
  sys.dont_write_bytecode = True

  retcode = pytest.main([pytest_path, "-p", "no:cacheprovider"])

  # Fail the cell execution if we have any test failures.
  assert retcode == 0, 'The pytest invocation failed. See the log above for details.'

# COMMAND ----------

# MAGIC %md
# MAGIC # Test scenarios with PySpark native tests
# MAGIC Before we use `run_pytest()`, let's take a closer look at how `chispa` works.

# COMMAND ----------

# MAGIC %md
# MAGIC ### DataFrame equality
# MAGIC
# MAGIC A common data transformation task is to add or remove columns from DataFrames.  To validate that our transformations are working as intended, we can assert that the actual output DataFrame of a function is equivalent to an expected output DataFrame.  We will use this approach to test DataFrame equality for schemas, columns, as well as row or column orders.
# MAGIC
# MAGIC `assertDataFrameEqual` is one of the new testing utility functions in Spark 3.5. It includes arguments to control whether row order should be compared and approximate comparisons for floating point values. For details see the [documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.testing.assertDataFrameEqual.html).
# MAGIC
# MAGIC

# COMMAND ----------

import pyspark.sql.functions as F
from src.transforms_spark import remove_non_word_characters
from pyspark.testing.utils import assertDataFrameEqual

# Dirty rows
dirty_rows = [
      ("jo&&se",),
      ("**li**",),
      ("#::luisa",),
      (None,)
  ]
source_df = spark.createDataFrame(dirty_rows, ["name"])

# Cleaned rows using function
clean_df = source_df.withColumn(
    "clean_name",
    remove_non_word_characters(F.col("name"))
)

# Expected output, should be identical to clean_df
expected_data = [
      ("jo&&se", "jose"),
      ("**li**", "li"),
      ("#::luisa", "luisa"),
      (None, None)
  ]
expected_df = spark.createDataFrame(expected_data, ["name", "clean_name"])

assertDataFrameEqual(clean_df, expected_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Formatted output with PyTest
# MAGIC Let's use the `assertDataFrameEqual` function to run the a test that will fail because it has an integer mixed with the strings. By running using pytest we get a formatted output showing passed and failed tests.

# COMMAND ----------

run_pytest("tests/pyspark_native/test_df_equality.py")

# COMMAND ----------

# MAGIC %md
# MAGIC The native PySpark testing functions provide clear text and visual explanations of why the test failed.  This makes it quick and easy to go back and iterate on our logic.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Approximate column equality
# MAGIC
# MAGIC For example, assume we have a DataFrame with a floating point column where values aren't exact. By setting `rtol` or `atol` to a less strict value than the defaults the test will accept some variances.
# MAGIC
# MAGIC From the documentation:
# MAGIC ```
# MAGIC         rtol : float, optional
# MAGIC             The relative tolerance, used in asserting approximate equality for float values in actual
# MAGIC             and expected. Set to 1e-5 by default. (See Notes)
# MAGIC         atol : float, optional
# MAGIC             The absolute tolerance, used in asserting approximate equality for float values in actual
# MAGIC             and expected. Set to 1e-8 by default. (See Notes)
# MAGIC
# MAGIC     For DataFrames with float/decimal values, assertDataFrame asserts approximate equality.
# MAGIC     Two float/decimal values a and b are approximately equal if the following equation is True:
# MAGIC         
# MAGIC         ``absolute(a - b) <= (atol + rtol * absolute(b))``.
# MAGIC ```
# MAGIC
# MAGIC A precision of 0.1 will determine these columns to be identical.  If we set precision to 0.01, the test would fail the assertion.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import StructType
from src.transforms_spark import remove_non_word_characters
from pyspark.testing.utils import assertDataFrameEqual

schema = StructType().add("num1", "float")

data1 = [
    [1.1],
    [2.2],
    [3.3],
    [None]
  ]
df = spark.createDataFrame(data1, schema)

data2 = [
    [1.1],
    [2.15],
    [3.37],
    [None]
  ]
df2 = spark.createDataFrame(data2, schema)


assertDataFrameEqual(df, df2, rtol=.04)
assertDataFrameEqual(df, df2, atol=.07)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Schema equality
# MAGIC
# MAGIC PySpark contains a specific function for testing schema equality called `assertSchemaEqual`.  It accepts an actual and expected **schema** as input, and returns a helpful error message in case the test fails:

# COMMAND ----------

from pyspark.testing.utils import assertSchemaEqual

# DF with numeric and string columns
data1 = [
    (1, "a"),
    (2, "b"),
    (3, "c"),
    (None, None)
    ]
df1 = spark.createDataFrame(data1, ["num", "letter"])
    
# DF with only numeric columns
data2 = [
    (1, 88.8),
    (2, 99.9),
    (3, 1000.1),
    (None, None)
    ]

df2 = spark.createDataFrame(data2, ["num", "double"])

# Compare them
assertSchemaEqual(df1.schema, df2.schema)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's run that with `pytest` to see what the test report looks like with PySpark native testing functions.

# COMMAND ----------

run_pytest("tests/pyspark_native/test_schema_equality.py")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Row and column order
# MAGIC Sometimes pipelines or analytics depend on row or column order.  `assertDataFrameEqual()` assumes that equality extends to row and column order.  The row order can be ignored by setting `ignoreRowOrder` to True. When checking schema equality with `assertSchemaEqual`, the column order can be ignored by setting `ignoreColumnOrder` to True. Both default to False.  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparing pandas on Spark DataFrames
# MAGIC
# MAGIC PySpark 3.5 also includes `assertPandasOnSparkEqual` for testing DataFrames created with pandas on Spark.  This function includes optional arguments to check approximate equality of the `ps.Series` and `ps.Index` attributes, similar to the approximate comparisons in `chispa`.  Note that DBR 14+ is required to run this cell.

# COMMAND ----------

import pyspark.pandas as ps
from pyspark.testing.pandasutils import assertPandasOnSparkEqual

psdf1 = ps.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
psdf2 = ps.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
assertPandasOnSparkEqual(psdf1, psdf2)  # pass, ps.DataFrames are equal

s1 = ps.Series([212.32, 100.0001])
s2 = ps.Series([212.32, 100.0])
assertPandasOnSparkEqual(s1, s2, checkExact=False)  # pass, ps.Series are approx equal

s1 = ps.Index([212.300001, 100.000])
s2 = ps.Index([212.3, 100.0001])
assertPandasOnSparkEqual(s1, s2, almost=True)  # pass, ps.Index obj are almost equal

# COMMAND ----------

# import pyspark.testing.utils as t
# help(t)
