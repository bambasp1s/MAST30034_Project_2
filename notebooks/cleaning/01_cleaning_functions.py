from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

VALID_STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
VALID_REVENUE_LEVELS = ["a", "b", "c", "d", "e"]
 
# aus post code ranges per state
POSTCODE_RANGES = {
    "NSW": [(1000, 2599), (2619, 2899),(2921, 2999)],
    "ACT": [(200, 299), (2600, 2618), (2900, 2920)],
    "VIC": [(3000, 3999), (8000, 8999)],
    "QLD": [(4000, 4999), (9000, 9999)],
    "SA": [(5000, 5999)],
    "WA": [(6000, 6999)],
    "TAS": [(7000, 7999)],
    "NT": [(800, 999)],
}


#--------------------------------------#

# general helper functions

# trim whitespace on string columns, turns empty strings into NULL
def trim_strings(df: DataFrame) -> DataFrame:
    for c, t in df.dtypes:
        if t == "string":
            df = df.withColumn(c, F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c))))
    return df


# null counts per column
def report_nulls(df: DataFrame) -> DataFrame:
    exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    return df.select(F.count(F.lit(1)).alias("_total_rows"), *exprs)


# returns True if a key appears more than once
def flag_duplicates(df: DataFrame, key: str, flag_name: str = None) -> DataFrame:
    flag_name = flag_name or f"dup_{key}"
    w = Window.partitionBy(key)
    return df.withColumn(flag_name, F.count(F.lit(1)).over(w) > 1)


# returns outliers with 1.5x iqr rule
def outliers_by_iqr(df: DataFrame, col: str, k: float = 1.5):
    q1, q3 = df.approxQuantile(col, [0.25, 0.75], 0.001)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    flagged = df.withColumn(f"is_outlier_{col}", (F.col(col) < lo) | (F.col(col) > hi))
    return flagged, lo, hi

# change column types, returns nulls for values that cannot be converted
def safe_cast(col, dtype):
    return F.expr(f"try_cast(`{col}` as {dtype})")

#--------------------------------------#
# merchant cleaning

_TAG_PATTERN = r"^\(\((.*)\),\s*\(([a-z])\),\s*\(take rate:\s*([0-9.]+)\)\)$"
 
 
def clean_merchants(df: DataFrame) -> DataFrame:

    df = trim_strings(df).dropDuplicates()
 
    norm = F.lower(F.col("tags"))
    norm = F.translate(norm, "[]", "()")
    norm = F.regexp_replace(norm, r"\s+", " ")
    norm = F.regexp_replace(norm, r"\(\s+", "(")
    norm = F.regexp_replace(norm, r"\s+\)", ")")
 
    df = (
        df.withColumn("merchant_abn", safe_cast("merchant_abn", "long"))
          .withColumn("_tags_norm", norm)
          .withColumn("category", F.regexp_extract("_tags_norm", _TAG_PATTERN, 1))
          .withColumn("revenue_level", F.regexp_extract("_tags_norm", _TAG_PATTERN, 2))
          .withColumn("take_rate", F.regexp_extract("_tags_norm", _TAG_PATTERN, 3))
    )

    # regexp_extract returns "" on no match
    for c in ["category", "revenue_level", "take_rate"]:
        df = df.withColumn(c, F.when(F.col(c) == "", None).otherwise(F.col(c)))
    df = df.withColumn("category", F.regexp_replace("category", r"\s*,\s*", ", "))
    df = df.withColumn("take_rate", safe_cast("take_rate", "double"))
 
    df = (
        df.withColumn("tags_unparsed", F.col("category").isNull())
          .withColumn("invalid_revenue_level", ~F.col("revenue_level").isin(VALID_REVENUE_LEVELS)
                      | F.col("revenue_level").isNull())
          .withColumn("invalid_take_rate", F.col("take_rate").isNull()
                      | ~F.col("take_rate").between(0, 100))
          .withColumn("invalid_abn", F.length(F.col("merchant_abn").cast("string")) != 11)
          .withColumn("missing_name", F.col("name").isNull())
    )
    df = flag_duplicates(df, "merchant_abn")
    return df.drop("_tags_norm")