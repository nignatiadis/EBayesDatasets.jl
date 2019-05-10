struct MovieLens <: EBayesDataset
     n_min_train::Int
     n_min_test::Int
     seed_id::Int
     prop_test::Float64
     movie_df #JuliaDB object
     X_df::DataFrame
     Zs_train::NormalSamples
     Zs_test::NormalSamples
     train_sd::Float64
     test_sd::Float64
end

const movie_genres = [("Action", "Action"),
                ("Adventure","Adventure"),
                ("Animation","Animation"),
                ("Children","Children"),
                ("Comedy","Comedy"),
                ("Crime", "Crime"),
                ("Documentary","Documentary"),
                ("Drama","Drama"),
                ("Fantasy","Fantasy"),
                ("Film-Noir","FilmNoir"),
                ("Horror","Horror"),
                ("Musical","Musical"),
                ("Mystery","Mystery"),
                ("Romance","Romance"),
                ("Sci-Fi","SciFi"),
                ("Thriller","Thriller"),
                ("War","War"),
                ("Western","Western"),
                ("(no genres listed)", "NoGenre")]



function MovieLens(;n_min_train=6, n_min_test= 11, seed_id=1, prop_test = 0.1)
        MovieLens(n_min_train, n_min_test, seed_id, prop_test)
end

function MovieLens(n_min_train, n_min_test, seed_id, prop_test)
        movies = loadtable(datadep"MovieLens-20m/ml-20m/movies.csv", indexcols=[1],
                           colparsers=[Int32,String,String])

        movie_titles = JuliaDB.select(movies, :title)
        movie_year_regex = r".*\((\d{4})\)"

        mm = match.(movie_year_regex, movie_titles)

        movies_f = movies[findall( .! isnothing.(mm))]
        movie_f_titles = JuliaDB.select(movies_f, :title)
        movie_f_year = map( x-> parse(Int64, match(movie_year_regex, x).captures[1]),
                            movie_f_titles)

        movies_f = JuliaDB.setcol(movies_f, :year => movie_f_year)


        ratings = loadtable(datadep"MovieLens-20m/ml-20m/ratings.csv", indexcols=[1,2],
                           colparsers=[Int32,Int32,Float32,Int]);

        ratings = JuliaDB.table(ratings, pkey=:userId)

        users = unique(JuliaDB.select(ratings, :userId))

        Random.seed!(seed_id)

        # TODO: Make sure the unit test checks if split at correct prop more or less
        train, test = splitobs(shuffleobs(users), at = prop_test)

        train_users = table(users[train], names=[:userId], pkey=:userId)
        test_users = table(users[test], names=[:userId], pkey=:userId)

        train_ratings = JuliaDB.join(ratings, train_users;
                             how=:inner, rselect=:userId)
        test_ratings = JuliaDB.join(ratings, test_users;
                             how=:inner, rselect=:userId)

        test_movie_mean = JuliaDB.groupby((:test_mean => mean,
                                           :test_n => length,
                                           :test_sd => std),
                           test_ratings, :movieId; select=:rating)
        train_movie_mean = JuliaDB.groupby((:train_mean => mean,
                                            :train_n => length,
                                            :train_sd => std),
                           train_ratings, :movieId; select=:rating)

        movie_df = JuliaDB.join(movies_f, test_movie_mean; lkey=:movieId, rkey=:movieId)
        movie_df = JuliaDB.join(movie_df, train_movie_mean; lkey=:movieId, rkey=:movieId)

        #TODO: Handling of n_min_test, n_min_train incosistent compared to Batting
        movie_df= JuliaDB.filter( row -> (row.train_n >= n_min_train) && (row.test_n >= n_min_test), movie_df)
        # step above reduces movies to ~10000



        # oh issue because it is an array now..

        train_sd = mean(JuliaDB.select(movie_df, :train_sd))
        test_sd = mean(JuliaDB.select(movie_df, :test_sd))

        train_sds = train_sd./sqrt.(JuliaDB.select(movie_df, :train_n))
        test_sds = test_sd./sqrt.(JuliaDB.select(movie_df, :test_n))


        train_Zs = NormalSamples(Float64.(JuliaDB.select(movie_df, :train_mean)),
                                 train_sds)

        test_Zs = NormalSamples(Float64.(JuliaDB.select(movie_df, :test_mean)),
                                 test_sds)


        df_X = DataFrame(year = JuliaDB.select(movie_df, :year),
                         n = JuliaDB.select(movie_df, :train_n))



        #mle_error = mean((truth .- preds_mle).^2 .- test_sds.^2)


        # construct Genre features...
        genres = JuliaDB.select(movie_df, :genres)
        movie_genre_dict = Dict(first.(movie_genres) .=> 1:19)

        feature_mat = zeros(Int64, size(genres,1), length(movie_genre_dict))

        for (i, genre_i) in enumerate(genres)
            split_genre = split(genre_i, "|")
            for spl in split_genre
                idx = get(movie_genre_dict, spl, nothing)
                if !isnothing(idx)
                    feature_mat[i, idx] = 1
                end
            end
        end

        # some properties of the feature matrix...
        #feature_sum = vec(mean(feature_mat, dims=1))

        #by_genre = DataFrame(genre=movie_genres , feature_sum = feature_sum[1:end-1])
        #sort!(by_genre, :feature_sum)

        for (i, genre) in enumerate(last.(movie_genres))
            df_X[Symbol(genre)] = feature_mat[:, i]
        end

        MovieLens( n_min_train, n_min_test, seed_id, prop_test,
                   movie_df,
                   df_X,
                   train_Zs, test_Zs,
                   train_sd, test_sd)
end

