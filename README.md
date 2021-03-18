# work_sample
A major part of my work is to construct ETL pipeline to collect clients' data. The data, usually in Excel, will be updated to Databricks, and the ETL pipeline will transform, validate and save it to Delta table if it meets certain criteria.

In transformer folder, you can find a scikit-learn pipeline style transformer designed for bucketizing column in Pandas or Koalas dataframe and its test file.

In validation folder, you can find a generalized validation module utilzied in ETL pipeline, and utility function to validate whether country is convertible to ISO2 standard.
