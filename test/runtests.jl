using CSV
using Query
using DataFrames
using Distributions
using StatsBase

brown_batting = CSV.File("rawdata/Brown_batting_data.txt",
                          normalizenames=true) |>
        DataFrame

abstract type EBayesTransform end

struct ArcsineTransform <: EBayesTransform end

function transform_data(arc::ArcsineTransform, h, n)
        asin(sqrt( (h + 1/4)/(n+1/2)))
end

function transform_params(arc::ArcsineTransform, p)
        asin(p)
end


abstract type EmpiricalBayesDataset end

struct BrownBatting end

function train_data(brown_batting)
        brown_batting_train
end

n_min_train = 11
n_min_test = 11

batting_clean = @from i in brown_batting begin
                @let AB_train = i.AB_4_ + i.AB_5_ + i.AB_6_
                @let AB_test = i.AB_7_ + i.AB_8_ + i.AB_9_10_
                @let H_train = i.H_4_ + i.H_5_ + i.H_6_
                @let H_test = i.H_7_ + i.H_8_ + i.H_9_10_
                @where AB_train >= 11
                @select {First_Name=i.First_Name, Last_Name=i.Last_Name,
                         Pitcher = i.Pitcher_, AB_train = AB_train,
                         AB_test = AB_test, H_train = H_train, H_test = H_test}
                @collect DataFrame
end


function train(br::BrownBatting)
        (Binomial(), batting_clean[:AB_train], batting_clean[:AB_test])
end

# rename as predict functions...
function MLE(hs, abs)
    transform_data.(Ref(ArcsineTransform()), hs, abs)
end

function grand_mean(hs, abs)
    res_vec = zeros(Float64, length(hs))
    grand_mean = mean(transform_data.(Ref(ArcsineTransform()), hs, abs))
    fill!(res_vec, grand_mean)
end

batting_pred1 = MLE(batting_clean[:H_train], batting_clean[:AB_train] )

function eval_pred(pred_fun)
    batting_eval = deepcopy(batting_clean)
    batting_pred = pred_fun(batting_clean[:H_train], batting_clean[:AB_train] )
    AB_test = batting_clean[:AB_test]
    batting_test = MLE(batting_clean[:H_test], AB_test)
    error_vec = (batting_pred .- batting_test).^2 .- 1./(4.*AB_test)
    batting_eval[:error_vec] = error_vec
    batting_eval = filter(row -> row[:AB_test] >= 11, batting_eval)
    sum(batting_eval[:error_vec])
end
