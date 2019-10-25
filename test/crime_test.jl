using DataDeps
using CSV
using DataFrames
using Random
using Query
push!(LOAD_PATH, "../../EmpiricalBayesBase")
push!(LOAD_PATH, "..")

using EBayesCore
using EBayesDatasets

register(DataDep(
  "uci_communities-and-crime",
  "https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized",
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
  "383a9530b2802b4457986095b13af8f02a802a2504feaef45783f9c129e003f1"))


crime_path = @datadep_str "communities-and-crime"
df = CSV.File("$crime_path/CommViolPredUnnormalizedData.txt", header=false) |>
     DataFrame

first(df,6)
crime_colnames = CSV.File("../rawdata/crime_and_communities/column_names.txt",
                     header=false, allowmissing=:none) |>
          DataFrame

names!(df, Symbol.(crime_colnames[:,1]))



length(findall(true_prop .== "?"))
df_filt = @from i in df begin
            @where i.nonViolPerPop != "?"
            @select i
            @collect DataFrame
          end

df_filt = df_filt |>
         @mutate(nonViolPerPop = parse.(Float64, _.nonViolPerPop)) |>
         DataFrame

df_filt_stefan = @from i in df_filt begin
                      @where i.population >= 20000
                      @select i
                      @collect DataFrame
                    end

# sanity check:
size(df_filt_stefan, 1) == 1192 #1992 in Stefan's paper
mean(df_filt_stefan[:nonViolPerPop])/1000 #~6% as it should

names(df_filt)

extrema(df_filt[:population])


npopulation = df_filt[:population]
crime_rate = df_filt[:nonViolPerPop] ./ 100_000
findmax(crime_rate)
findmin(crime_rate)


df_filt[1817,:]
ncrimes = Int64.(round.(df_filt[:nonViolPerPop] ./ 100_000 .* npopulation))


using Distributions

B_subsample = 100

# Hypergeometric(s, f, n)  s successes, f failures, n trials
# (s k)(f n-k)/(s+f n)

# prob k incidents =  (from violent results choose k *  from non violent results choose n-k) /

# so here; Hypergeometric(Violent, Non-Violent, B_subsample)
Random.seed!(1)
subsampled_crimes = rand.( Hypergeometric.(ncrimes, npopulation .- ncrimes, B_subsample))


unbiased_est = subsampled_crimes ./ B_subsample


mean( sqrt.(unbiased_est) .- sqrt.(crime_rate ))
std( sqrt.(unbiased_est) .- sqrt.(crime_rate ))./ sqrt.(length(crime_rate))

sqrt(var( sqrt.(unbiased_est) .- sqrt.(crime_rate ))*4*B_subsample)


unbiased_errors = (unbiased_est .- crime_rate).^2
unbiased_mse = mean(unbiased_errors )*1e6
unbiased_mse_std = std(unbiased_errors ) ./ sqrt.(length(crime_rate))*1e6

unbiased_sqrt_errors = ( sqrt.(unbiased_est) .- sqrt.(crime_rate)) .^2
unbiased_sqrt_mse =  mean(unbiased_sqrt_errors )*1e5


transformed_crimes =  NormalSamples( sqrt.(unbiased_est),
                                     ones(length(unbiased_est)) .* sqrt(1/B_subsample/4))


1/sqrt(B_subsample)/2
sure_fit = fit(SURE(Normal()), transformed_crimes)

sure_sqrt_preds = predict(sure_fit)

sure_sqrt_errors = ( sure_sqrt_preds .-  sqrt.(crime_rate)) .^2
sure_sqrt_mse = mean(sure_sqrt_errors)*1e5


 # transform back as well
sure_errors =  (sure_sqrt_preds.^2 .- crime_rate).^2
sure_mse = mean(sure_errors )*1e6
sure_mse_std = std(sure_errors)/ sqrt.(length(crime_rate)) *1e6

names(df_filt)
names(df_filt)[130:140]
X = df_filt[:,6:129]
X_sub =X[:, findall( typeof.(eachcol(X)) .== Vector{Float64})]

using MLJ
MLJ.@load XGBoostRegressor

tree = XGBoostRegressor(max_depth=5)

r_num_round = range(tree, :num_round, lower=2, upper=100)
r_eta = range(tree, :eta, lower=0.01, upper=1.0)
#r_max_depth = range(tree, :adeta, lower=2, upper=5)
#r_gamma = range(tree, :gamma, lower=0, upper=1)
nested_ranges = (num_round = r_num_round, eta = r_eta)
                # gamma=r_gamma)
tuned_XGBoost = TunedModel(model=tree, #,
                          tuning=Grid(resolution=10),
                          resampling=CV(nfolds=5),
                          nested_ranges=nested_ranges,
                          measure=rms)

xgboost_fit = fit(tuned_XGBoost, 1, X_sub, transformed_crimes.Z)

xgboosed_fitted_params = xgboost_fit[3]
findmin(xgboosed_fitted_params.measurements)
xgboosed_fitted_params.parameter_values[15,:]


xgboost_fit_preds = StatsBase.predict(xgboost_fit[1], X_sub)

xgboost_errors =  (xgboost_fit_preds.^2 .- crime_rate).^2
xgboost_mse = mean(xgboost_errors)*1e6
std(xgboost_errors)/sqrt(length(crime_rate))*1e6

# not too bad actually




ebcf_method = EBayesCrossFit(SURE(Normal()),
                             RegressionLocation(tuned_XGBoost), 5)
ebcf_fit = fit(ebcf_method, X_sub, transformed_crimes)

ebcf_preds = predict(ebcf_fit)
ebcf_errors =  (ebcf_preds.^2 .- crime_rate).^2
mean(ebcf_errors)*1e6

std(ebcf_errors)/sqrt(length(crime_rate))*1e6

using JLD2
@save "xgboost_ebcf_preds.jld2" ebcf_preds

@save "xgboost_ebcf_preds_200.jld2" ebcf_preds

79.52 + 4.70 # need X2 multiplier      EBCF
122.42 + 7.57         # XGBOOST
85.59 + 3.58          # SURE
92.23 + 3.55          # Naive

idx_enough = findall(npopulation .>= 20000)


mean(crime_rate)*1000
mean(crime_rate)

mean(ebcf_errors)*1e6
std(ebcf_errors)*1e6/sqrt(length(crime_rate))

xgb 168 +- 10
sure 184 +- 9.46
ebcf 149 + 10
unbiased  224 +- 8.42

mean(xgboost_errors)*1e6
std(xgboost_errors)*1e6/sqrt(length(crime_rate))

mean(sure_errors)*1e6
std(sure_errors)*1e6/sqrt(length(crime_rate))

mean(unbiased_errors)*1e6
std(unbiased_errors)*1e6/sqrt(length(crime_rate))


ebcf_errors_sqrt = (ebcf_preds .- sqrt.(crime_rate).^2)
mean(ebcf_errors_sqrt)
std(ebcf_errors_sqrt)/sqrt(length(crime_rate))

xgboost_errors_sqrt = (xgboost_fit_preds .- sqrt.(crime_rate).^2)
mean(xgboost_errors_sqrt)
std(xgboost_errors_sqrt)/sqrt(length(crime_rate))
