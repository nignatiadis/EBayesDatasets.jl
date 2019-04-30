using DataDeps
using CSV
using DataFrames

register(DataDep(
  "uci_communities-and-crime",
  "https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized",
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
  "383a9530b2802b4457986095b13af8f02a802a2504feaef45783f9c129e003f1"))


crime_path = @datadep_str "uci_communities-and-crime"
df = CSV.File("$crime_path/CommViolPredUnnormalizedData.txt", header=false) |>
     DataFrame


crime_colnames = CSV.File("rawdata/crime_and_communities/column_names.txt",
                     header=false, allowmissing=:none) |>
          DataFrame

names!(df, Symbol.(crime_colnames[:,1]))
