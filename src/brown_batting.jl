const brown_path = joinpath(dirname(@__FILE__),
                            "..",
                            "rawdata",
                            "Brown_batting_data.txt")


struct BrownBatting <: EBayesDataset
    n_min_train::Int
    n_min_test::Int
    raw_df::DataFrame
    df::DataFrame
    players::Symbol
end

broadcastable(br::BrownBatting) = Ref(BrownBatting)


function BrownBatting(; n_min_train=11, n_min_test=11, players=:all)
    BrownBatting(n_min_train, n_min_test, players)
end

function BrownBatting(n_min_train, n_min_test, symbol)
    raw_df = CSV.File(brown_path,
                      normalizenames=true) |>
             DataFrame

    batting_clean = @from i in raw_df begin
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

    batting_clean[:H_train] = convert(Vector{Int64}, batting_clean[:H_train])
    batting_clean[:H_test] = convert(Vector{Int64}, batting_clean[:H_test])
    batting_clean[:AB_train] = convert(Vector{Int64}, batting_clean[:AB_train])
    batting_clean[:AB_test] = convert(Vector{Int64}, batting_clean[:AB_test])
    batting_clean[:Pitcher] = categorical(batting_clean[:Pitcher])

    if symbol == :pitchers
        batting_clean = DataFrames.filter(row -> row[:Pitcher] != 0, batting_clean)
    elseif symbol == :nonpitchers
        batting_clean = DataFrames.filter(row -> row[:Pitcher] == 0, batting_clean)
    end
    BrownBatting(n_min_train, n_min_test, raw_df, batting_clean, symbol)
end

#
function traindata(br::BrownBatting)
    batting_clean = br.df
    if br.players == :all
        X = batting_clean[[:AB_train, :Pitcher]]
    else
        X = batting_clean[[:AB_train]] #maybe expose vector later.
    end

    Y = HeteroskedasticBinomialSamples(batting_clean[:H_train],
                                       batting_clean[:AB_train])
    # figure out how to properly transform later.
    Y = transform(ArcsineTransform(), Y)
    (X,Y)
end

function testdata(br::BrownBatting)
    batting_clean = br.df
    Y = HeteroskedasticBinomialSamples(batting_clean[:H_test],
                                       batting_clean[:AB_test])
    transform(ArcsineTransform(), Y)
end

#function traindata(br::BrownBatting, HeteroskedasticNormalSamples)
#    EBayesCore.transform(ArcsineTransform, traindata(br))
#end


struct TSE <: EBayesBenchmark end

default_transform(tse::TSE) = ArcsineTransform()

struct NormalizedTSE <: EBayesBenchmark
    baseline_method
end


NormalizedTSE() = NormalizedTSE(EBayesCore.FlatPrior())

broadcastable(bench::EBayesBenchmark) = Ref(bench)

function benchmark(br::BrownBatting, tse::TSE, method)
    X, ss_train = traindata(br)
    ss_test = testdata(br)

    #transf = default_transform(tse)
    test_z_score = predict(EBayesCore.FlatPrior(), ss_test)

    pred_z_score = predict(method, X, ss_train)
    error_vec = (test_z_score .- pred_z_score).^2 .- ss_test.σ.^2
    #keep_idx = ss_test.Ns .>= br.n_min_test
    keep_idx = ss_test.σ.^2 .<= 1/(4*br.n_min_test)

    sum(error_vec[keep_idx])
end

#function benchmark(br::BrownBatting, tse::TSE, method)
#    benchmark(br, tse, (method,))
#end

function benchmark(br::BrownBatting, tse::NormalizedTSE, method)
    baseline_tse = benchmark(br, TSE(), tse.baseline_method)
    method_tse = benchmark(br, TSE(), method)
    method_tse/baseline_tse
end

benchmark(br::BrownBatting, method) = benchmark(br, NormalizedTSE(), method)
