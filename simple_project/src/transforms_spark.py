import pyspark.sql.functions as F

# Removing all non-word characters in PySpark
def remove_non_word_characters(col):
    return F.regexp_replace(col, "[^\\w\\s]+", "")