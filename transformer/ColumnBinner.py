from typing import List, Optional

import databricks.koalas as ks
import numpy as np
import pandas as pd
from pyspark.ml.feature import Bucketizer
from sklearn.base import BaseEstimator, TransformerMixin

from data_utils.preprocessing.base import set_df_library


class ColumnBinner(BaseEstimator, TransformerMixin):
    """
    Bucketize a column in Pandas/Koalas dataframe with given split points and labels.

    :param Optional[str] from_column: column from which to bucket,
      default None
    :param Optional[str] to_column: column to which to store the labels
      default None
    :param Optional[List[float]] bins: numerical split points with n+1 elements,
      default None
    :param Optional[List[str]] labels: labels for each bin with n elements,
      default None
    """

    def __init__(
        self,
        from_column: Optional[str] = None,
        to_column: Optional[str] = None,
        bins: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        self.from_column = from_column
        self.to_column = to_column
        self.bins = bins
        self.labels = labels
        self.dflib_ = None

    def fit(self, X, y=None):
        # pylint: disable=unused-argument
        return self

    def transform(self, X):
        """Bucket from_column and assign to
        to_column in X.

        :param Union[pd.DataFrame, ks.DataFrame] X: input pandas/koalas dataframe
        :return: transformed dataframe
        """
        self.dflib_ = set_df_library(X)
        if self.dflib_ == ks:
            # rename "index" column to "index_place_holder" if already exist
            if "index" in X.columns:
                X = X.rename(columns={"index": "index_place_holder"})
            # drop to_column if already exist
            if self.to_column in X.columns:
                X = X.drop(self.to_column)
            # Bucketizer's right range point is inclusive, warning: 0 will be converted to negative
            sdf = X.to_spark(index_col="index_")
            bucketizer = Bucketizer(
                splits=self.bins, inputCol=self.from_column, outputCol=self.to_column, handleInvalid="keep"
            )
            sdf = bucketizer.transform(sdf)
            X = sdf.to_koalas(index_col="index_")
            # X = X.rename(columns={"index": "index_"})
            # X.set_index("index_", inplace=True)
            if "index_place_holder" in X.columns:
                X = X.rename(columns={"index_place_holder": "index"})
            # ks doesn't support multi-dtype repalcement, e.g. {1.0: 'a'},
            # but NaN is still kept null after astype('str')
            X[self.to_column] = X[self.to_column].astype("str")
            X = X.replace(dict(zip(np.arange(0.0, len(self.bins) - 1).astype("str").tolist(), self.labels)))
        elif self.dflib_ == pd:
            # pd.cut will return category dtype, which is unrecognizable for
            # spark, ks, Alteryx, so let's convert to str,
            # warning: 0.0 will be converted to '0.0'
            null_filter = X[self.from_column].isnull()
            X[self.to_column] = pd.cut(x=X[self.from_column], bins=self.bins, labels=self.labels).astype("str")
            X.loc[null_filter, self.to_column] = np.nan
        return X
