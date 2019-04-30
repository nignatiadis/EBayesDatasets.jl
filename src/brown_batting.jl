const brown_path = joinpath(dirname(@__FILE__),
                            "..",
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

    BrownBatting(n_min_train, n_min_test, raw_df, batting_clean)

end

#
function traindata(br::BrownBatting)
    batting_clean = br.df
    EBayesCore.HeteroskedasticBinomialSamples(batting_clean[:H_train],
                                              batting_clean[:AB_train])
end

function testdata(br::BrownBatting)
    batting_clean = br.df
    EBayesCore.HeteroskedasticBinomialSamples(batting_clean[:H_test],
                                              batting_clean[:AB_test])
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



function benchmark(br::BrownBatting, tse::TSE, method::Tuple)
    ss_train = traindata(br)
    ss_test = testdata(br)

    transf = default_transform(tse)
    test_z_score = predict(EBayesCore.FlatPrior(), transf, ss_test)

    pred_z_score = predict(method..., transf, ss_train)
    error_vec = (test_z_score .- pred_z_score).^2 .- 1 ./ (4 .* ss_test.Ns)
    keep_idx = ss_test.Ns .>= br.n_min_test

    sum(error_vec[keep_idx])
end

function benchmark(br::BrownBatting, tse::TSE, method)
    benchmark(br, tse, (method,))
end

function benchmark(br::BrownBatting, tse::NormalizedTSE, method)
    baseline_tse = benchmark(br, TSE(), tse.baseline_method)
    method_tse = benchmark(br, TSE(), method)
    method_tse/baseline_tse
end
