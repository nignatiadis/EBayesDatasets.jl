using Revise
using EBayesCore
using EBayesDatasets


br = EBayesDatasets.BrownBatting()
br_pitcher = EBayesDatasets.BrownBatting(players=:pitchers)
br_nonpitcher = EBayesDatasets.BrownBatting(players=:nonpitchers)

test_br = EBayesDatasets.testdata(br)
X, train_br = EBayesDatasets.traindata(br)
X_pitchers, train_pitchers = EBayesDatasets.traindata(br_pitcher)


predict(EBayesCore.FlatPrior(), ArcsineTransform(), test_br)

predict(GrandMeanLocation(), ArcsineTransform(), train_br)

benchmark_mse = EBayesDatasets.benchmark(br, EBayesCore.FlatPrior())

benchmark_mse_norm = EBayesDatasets.benchmark(br,
                                      EBayesDatasets.NormalizedTSE(),
                                      EBayesCore.FlatPrior())

EBayesDatasets.benchmark(br, SURE(Normal()))
EBayesDatasets.benchmark(br_pitcher, SURE(Normal()))
EBayesDatasets.benchmark(br_nonpitcher, SURE(Normal()))


benchmark_mse_norm = EBayesDatasets.benchmark(br,
                                              EBayesDatasets.NormalizedTSE(),
                                              SURE(Normal()))
