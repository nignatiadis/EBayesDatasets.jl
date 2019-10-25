module EBayesDatasets

    # datasets I used for brown batting
    using CSV
    using DataDeps
    using DataFrames
    using Query
    using EBayes

    import Base.Broadcast: broadcastable

    # Datasets I used for movie lens (split up into individual modules later)

    using Dates, GroupLens, JuliaDB
    using Random
    using IterTools
    using MLDataPattern
    using StatsBase
    import StatsBase:std

    select = JuliaDB.select
    table = JuliaDB.table
    groupby = JuliaDB.groupby

    abstract type EBayesDataset end
    broadcastable(eb_dataset::EBayesDataset) = Ref(eb_dataset)


    #Brown Batting dataset -> do not load this temporarily
    # include("brown_batting.jl")

    #Load MovieLens.jl
    include("movielens.jl")


    #function __init__()
    #    register(DataDep(
    #      "communities-and-crime",
    #      "https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized",
    #      "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
    #      "383a9530b2802b4457986095b13af8f02a802a2504feaef45783f9c129e003f1"
    #    ))
    #end
end # module
