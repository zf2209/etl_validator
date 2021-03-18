import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from data_utils.load_env import is_spark_installed
from data_utils.preprocessing.column_imputer import ColumnImputer

if is_spark_installed:
    import databricks.koalas as ks


def test_pandas():
    df = pd.DataFrame(columns=["col1", "col2", "col3"], data=[[1, 2, None], [None, 4, 5], [None, 6, 6]])
    _test_column_imputer_pipeline(df.copy())
    _test_column_imputer_pipeline_multiple(df.copy())


@pytest.mark.skipif(not is_spark_installed, reason="requires spark is installed")
def test_koalas():
    df = ks.DataFrame(columns=["col1", "col2", "col3"], data=[[1, 2, None], [None, 4, 5], [None, 6, 6]])
    _test_column_imputer_pipeline(df.copy())
    _test_column_imputer_pipeline_multiple(df.copy())


def _test_column_imputer_pipeline(df):
    pipe = Pipeline([("impute", ColumnImputer(to_column="col1", from_column="col2"))])
    transformed_df = pipe.transform(df)

    assert transformed_df.at[1, "col1"] == 4
    assert transformed_df.at[2, "col1"] == 6


def _test_column_imputer_pipeline_multiple(df):
    pipe = Pipeline([("impute", ColumnImputer(to_column=["col1", "col3"], from_column=["col2", "col2"]))])
    transformed_df = pipe.transform(df)

    assert transformed_df.at[0, "col3"] == 2
    assert transformed_df.at[1, "col1"] == 4
    assert transformed_df.at[2, "col1"] == 6
