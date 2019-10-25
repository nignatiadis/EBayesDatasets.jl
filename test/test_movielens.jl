using Revise

using EBayesCore
using EBayesDatasets

activate
tst = EBayesDatasets.MovieLens(;seed_id=2)

