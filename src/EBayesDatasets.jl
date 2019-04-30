module EBayesDatasets

    using CSV
    using DataDeps
    using DataFrames
    using Query
    using EBayesCore

    abstract type EBayesDataset end

    #brown_batting
    include("brown_batting.jl")



    function __init__()
        include("crime_and_communities.jl")
    end
end # module
