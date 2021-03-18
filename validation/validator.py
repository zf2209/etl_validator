import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yaml
from data_utils.typing import coerce_type_df
from gc_ontology.core.manager import OntologyManager

from client_data.schema_inference.utils import get_package_data_files_path
from client_data.validation.validation_utils import validate_iso2


hard_validation = ["non_nullable", "allowed_values", "type"]


def ontology_validator(
    df: pd.DataFrame,
    error_tolerance: Optional[float] = 0.05,
    base_path: Optional[str] = "DEFAULT",
    view_path: Optional[str] = None,
    validation_config: Optional[Dict] = None,
    hash_col: Optional[List[str]] = None,
    disable_validation: Optional[List[str]] = None,
    display_name: Optional[bool] = False,
) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
    """ Validate dataframe with view schema from Ontology and user specified validation config,
      and return validation results for overall dataframe, each column/rule and cell.

    >>> result = ontology_validator(df, view)
    >>> result[0]
    False
    >>> result[1]
        column_name validation_type  validation_rule  pass   count  error_rate  distinct_values
    0   all_columns	columns_complete [a, b]           False  5      0.5         'd'
    1   all_columns	columns_order    [a, b]           False  5      0.5         'd'
    3   b           non_nullable     non_nullable     False  3      0.3         nan
    4   a           allowed_values   [1, 2, 4]        False  1      0.1         5
    5   c           min              3                False  1      0.1         2
    8   b           maxlen           2                False  1      0.1         'abc'
    >>> result[2]
         a  a_non_nullable  b_non_nullable  a_allowed_values  c_min  c_max  b_minlen  b_maxlen  a_type  b_type  c_type
    0    1  True            False           True              False  True   True      False     True    True    True
    1    2  True            True            True              False  True   True      True      True    True    True
    2    3  True            True            False             True   True   True      False     True    True    True

    :param pd.DataFrame df: input dataframe to validate
    :param int error_tolerance: error rate tolerance to pass soft validation, default 0.05
    :param str base_path: base path for ontology_manager
    :param str view_path: view dict to retrieve data schema from ontology, e.g. "gold/cybercube_input.yaml"
    :param Dict validation_config: a dict of format {valudation: rule} for validation rules manual specification,
      and it will be excuted in complementary to ontology config
    :param List[str] hash_col: ["col1", "col2"] you want to use as hash index in addition to df index
    :param List[str] disable_validation: a list of validation rules you want to skip for ontology config
    :param bool display_name: whether to validate on display names, default False

    :return Tuple[bool, pd.DataFrame, bool, pd.DataFrame, bool, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            1st bool -> whether the df passes all the validations,
            1st pd.DataFrame -> validation results per col/rule,
            2nd bool -> whether the df passes all the hard validations,
            2nd pd.DataFrame -> hard validation results per col/rule,
            3rd bool -> whether the df passes all the soft validations with error_tolerance,
            3rd pd.DataFrame -> soft validation results per col/rule,
            4th pd.DataFrame -> a hash df containing errors per col/rule,
            5th pd.DataFrame -> a clean subset (row filter) that passes all the validations
            6th pd.DataFrame -> a message df containing validation errors for each row
                with comma separated col_error encoding string
    """
    # pylint: disable=too-many-arguments
    df_original = df.copy()
    validation_config = validation_config or {}
    hash_col = hash_col or []
    disable_validation = disable_validation or []
    result_df = pd.DataFrame(
        data={
            "column_name": [],
            "validation_type": [],
            "validation_rule": [],
            "pass": [],
            "count": [],
            "error_rate": [],
            "distinct_values": [],
        }
    )
    # typecast dataframes
    result_df, _ = coerce_type_df(
        result_df, {"column_name": "str", "validation_type": "str", "validation_rule": "str", "pass": "bool"}
    )

    if not set(hash_col).issubset(set(df.columns)):
        raise KeyError(f"{set(hash_col) - set(df.columns)} not in input df columns: {list(df.columns)}")
    result_index = df[hash_col].copy()

    if view_path:
        view = yaml.load(open(f"{get_package_data_files_path()}/config/views/{view_path}"))
        ontology = OntologyManager(base_path=base_path, validate=False).load_view(view)
        ontology_validations = {
            "columns_complete": [ontology.column_mapping_display, validate_col_complete],
            "columns_order": [ontology.column_mapping_display, validate_col_order],
            "non_nullable": [ontology.non_nullable_fields, validate_non_nullable_fields],
            "allowed_values": [ontology.allowed_values_fields, validate_allowed_values],
            "min_value": [ontology.min_value_fields, validate_min_value],
            "max_value": [ontology.max_value_fields, validate_max_value],
            "minlen": [ontology.min_len_fields, validate_minlen],
            "maxlen": [ontology.max_len_fields, validate_maxlen],
            "type": [ontology.field_types, validate_type],
        }
        for k, v in ontology_validations.items():
            if k in ["columns_complete", "columns_order"]:
                sub_validation_config = (
                    list(ontology.column_mapping_display.values())
                    if display_name
                    else list(ontology.column_mapping.keys())
                )
            else:
                sub_validation_config = v[0](display_name=display_name)
            if not k in disable_validation:
                result_df, result_index = v[1](
                    df=df, rule=sub_validation_config, result_df=result_df, result_index=result_index
                )

    validations = {
        "columns_complete": validate_col_complete,
        "columns_order": validate_col_complete,
        "non_nullable": validate_non_nullable_fields,
        "allowed_values": validate_allowed_values,
        "min_value": validate_min_value,
        "max_value": validate_max_value,
        "minlen": validate_minlen,
        "maxlen": validate_maxlen,
        "type": validate_type,
        "duplicates": validate_duplicates,
        "iso2": validate_iso2,
        "condition": validate_condition,
    }
    for k, v in validations.items():
        if k in validation_config:
            result_df, result_index = v(
                df=df, rule=validation_config[k], result_df=result_df, result_index=result_index
            )
    result_df = result_df.drop_duplicates()
    # type clean up and add time stamp
    result_df["pass"] = result_df["pass"].astype("bool")
    time_stamp = datetime.datetime.now()
    result_df["time_stamp"] = time_stamp
    result_index["time_stamp"] = time_stamp

    error_df = result_df.loc[~result_df["pass"]]
    hard_error_df = error_df.loc[error_df["validation_rule"].isin(hard_validation)]
    soft_error_df = error_df.loc[~error_df["validation_rule"].isin(hard_validation)]
    soft_error_df = soft_error_df.loc[soft_error_df["error_rate"] >= error_tolerance]
    pass_validation = error_df.empty
    pass_hard_validation = hard_error_df.empty
    pass_soft_validation = soft_error_df.empty
    error_index_df = (
        result_index.reset_index()
        .rename({"index": "row_number"})
        .iloc[np.where(~np.all(result_index[(set(result_index.columns) - set(hash_col))], axis=1))[0]]
    )
    clean_df = df_original.loc[set(df_original.index) - set(error_index_df.index)]
    message_df = pd.DataFrame(data={"validation_error": [""]*len(error_index_df)}, index=error_index_df["index"].to_list())
    columns = list(set(error_index_df.columns) - set(["index", "time_stamp"]))
    if not pass_validation:
        error_df.loc[error_df["validation_rule"].isin(hard_validation), "hard_soft_validation"] = "hard"
        error_df["hard_soft_validation"].fillna("soft", inplace=True)
        for col in columns:
            message_df.loc[~error_index_df[col], "validation_error"] += col + ","
        message_df["validation_error"] = message_df["validation_error"].str[:-1]

    return (
        pass_validation,
        error_df,
        pass_hard_validation,
        hard_error_df,
        pass_soft_validation,
        soft_error_df,
        error_index_df,
        clean_df,
        message_df,
    )
