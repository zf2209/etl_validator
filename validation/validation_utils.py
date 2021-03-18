from typing import Dict, List, Tuple, Union, Optional, Callable

import numpy as np
import pandas as pd
import pycountry as pc


def validate_iso2(
    df: pd.DataFrame,
    rule: Dict[str, int],
    result_df: Optional[pd.DataFrame] = None,
    result_index: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate if values in a country column are in or convertable to ISO2 standard.
    Returns result_df and result_index.

    >>> result[0]
        column_name validation_type  validation_rule  pass   count  error_rate  distinct_values
    0   all_columns	columns_complete [a, b]           False  5      0.5         'd'
    1   all_columns	columns_order    [a, b]           False  5      0.5         'd'
    3   b           non_nullable     non_nullable     False  3      0.3         nan

    >>> result[1]
         a  a_non_nullable  b_non_nullable  a_allowed_values  c_min  c_max  b_minlen  b_maxlen  a_type  b_type  c_type
    0    1  True            False           True              False  True   True      False     True    True    True
    1    2  True            True            True              False  True   True      True      True    True    True

    :param pd.DataFrame df: pandas df to validate
    :param List[str] rule: a list of columns to validate, e.g. {'domicile', 'country_client'}
    :param Optional[pd.DataFrame] result_df: validation results per col/rule, if not None will append result
    :param Optional[pd.DataFrame] result_index: hash validation results for each cell, if not None will append result
    :return Tuple[pd.DataFrame, pd.DataFrame]: the function returns 1st pd.DataFrame -> validation results per col/rule,
      2nd pd.DataFrame -> hash validation results for each cell
    """
    # pylint: disable=too-many-nested-blocks
    # this is a growing list of unallowed values containing continents and world
    unallowed_values = ["aisa", "africa", "europe", "north america", "south america", "australia/oceania", "antarctica"]
    # get latest lowercase ISO2 country code from pycountry
    iso_2 = [country_dict.alpha_2.lower() for country_dict in pc.countries.search_fuzzy("")]
    if rule:
        for col in rule:
            if col in df.columns:
                country_distinct = list(df[col].unique())
                country_invalid = []
                for country in country_distinct:
                    if isinstance(country, str):
                        country_clean = country.lower()
                        # country should not exist in unallowed_values and have "world" in it
                        if (country_clean in unallowed_values) or ("world" in country_clean):
                            country_invalid.append(country)
                        else:
                            # if 2-digit, must be in ISO2 standard
                            if len(country_clean) == 2:
                                if not country_clean in iso_2:
                                    country_invalid.append(country)
                            else:
                                try:
                                    _ = pc.countries.search_fuzzy(country_clean)
                                # if unable to find a fuzzy match
                                except LookupError:
                                    country_invalid.append(country)
                val_result = ~df[col].isin(country_invalid)
                result_df, result_index = _append_result(
                    col=col,
                    validation_type="iso2",
                    validation_rule="iso2",
                    validation_result=val_result,
                    df=df,
                    result_df=result_df,
                    result_index=result_index,
                )
    return result_df, result_index
