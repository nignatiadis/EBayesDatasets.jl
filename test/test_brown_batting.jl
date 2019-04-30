using EBayesDatasets
using EBayesCore

br = EBayesDatasets.BrownBatting()

test_br = EBayesDatasets.testdata(br)
train_br = EBayesDatasets.traindata(br)


predict(EBayesCore.FlatPrior(), ArcsineTransform(), test_br)

predict(GrandMeanLocation(), ArcsineTransform(), train_br)

benchmark_mse = EBayesDatasets.benchmark(br, EBayesDatasets.TSE(), EBayesCore.FlatPrior())

benchmark_mse_norm = EBayesDatasets.benchmark(br,
                                      EBayesDatasets.NormalizedTSE(),
                                      EBayesCore.FlatPrior())

benchmark_mse_norm = EBayesDatasets.benchmark(br,
                                              EBayesDatasets.NormalizedTSE(),
                                              GrandMeanLocation())

                                            
