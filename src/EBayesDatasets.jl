module EBayesDatasets

    using CSV
    using DataDeps
    using DataFrames
    using Query
    using EBayesCore

    import EBayesCore:transform
    import Base.Broadcast: broadcastable

    abstract type EBayesDataset end
    broadcastable(eb_dataset::EBayesDataset) = Ref(eb_dataset)

    abstract type EBayesBenchmark end

    #brown_batting
    include("brown_batting.jl")



    function __init__()
        register(DataDep(
          "communities-and-crime",
          "https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized",
          "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
          "383a9530b2802b4457986095b13af8f02a802a2504feaef45783f9c129e003f1"
        ))
    end
end # module
