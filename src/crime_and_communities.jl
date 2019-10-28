function __init__()
     register(DataDep(
       "communities-and-crime",
       "https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized",
       "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
       "383a9530b2802b4457986095b13af8f02a802a2504feaef45783f9c129e003f1"
    ))
end

const crime_colnames_path = joinpath(dirname(@__FILE__), "..", "rawdata",
                              "crime_and_communities","column_names.txt")


