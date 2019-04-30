using CSV
using Query
using DataFrames
using StatsBase
using MLDataPattern
using Random


const brown_path = joinpath(dirname(@__FILE__),
                            "rawdata",
                            "Brown_batting_data.txt")


struct BrownBatting <: EBayesDataset
    n_min_train::Int
    n_min_test::Int
    raw_df::DataFrame
    df::DataFrame
end

BrownBatting() = BrownBatting(11, 11)

function BrownBatting(n_min_train, n_min_test)
    raw_df = CSV.File(brown_path,
                      normalizenames=true) |>
             DataFrame

    batting_clean = @from i in brown_batting begin
                    @let AB_train = i.AB_4_ + i.AB_5_ + i.AB_6_
                    @let AB_test = i.AB_7_ + i.AB_8_ + i.AB_9_10_
                    @let H_train = i.H_4_ + i.H_5_ + i.H_6_
                    @let H_test = i.H_7_ + i.H_8_ + i.H_9_10_
                    @where AB_train >= n_min_train
                    @select {First_Name=i.First_Name, Last_Name=i.Last_Name,
                             Pitcher = i.Pitcher_,
                             AB_train = AB_train, AB_test = AB_test,
                             H_train = H_train, H_test = H_test}
                    @collect DataFrame
             end

    BrownBatting(n_min_train, n_min_test, raw_df, batting_clean)

end
