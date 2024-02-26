import os
import surprise

import pandas as pd
import numpy as np


sigmoid = lambda x: 1. / (1. + np.exp(-(x - 3) / 0.1))

def define_genre(meta_data):
    genres = []
    for ge in meta_data["genres"]:
        for temp in eval(ge):
            if temp not in genres:
                genres.append(temp)
    g_idx = [g["id"] for g in genres]
    meta_data["genres"] = meta_data["genres"].map(lambda xx: [(xxx["id"], xxx["name"]) for xxx in eval(xx)])
    return meta_data, g_idx

def define_comp(meta_data):
    comp = []
    for ge in meta_data["production_companies"]:
        for temp in eval(ge):
            if temp not in comp:
                comp.append(temp)
    g_idx = [g["id"] for g in comp]
    meta_data["production_companies"] = meta_data["production_companies"].map(lambda xx: [ (xxx["id"], xxx["name"]) for xxx in eval(xx)])
    return meta_data, g_idx

def select_genres(meta_data):
    # Selecting Companies
    # MGM, Warner Bros, Paramount, 20th Fox, Columbia (x2)
    # 5 Movie Companies with most user ratings
    # selected_companies = [1, 2, 3, 4, 7, 8]

    # 10 Movie Companies with most user ratings
    selected_genres = list(range(2,11))
    gen_to_group = list(range(9))

    gen = meta_data["genres"].value_counts().index[selected_genres]
    gen_dict = dict([(x, gen_to_group[i]) for i, x in enumerate(gen)])
    meta_data = meta_data.astype({"id": "int"})
    meta_data = meta_data[meta_data["genres"].isin(gen)]

    return meta_data, gen_dict

def select_companies(meta_data):
    # Selecting Companies
    # MGM, Warner Bros, Paramount, 20th Fox, Columbia (x2)
    # 5 Movie Companies with most user ratings
    # selected_companies = [1, 2, 3, 4, 7, 8]

    # 10 Movie Companies with most user ratings
    # selected_companies = [1, 2, 3, 4, 5, 7, 6, 8, 9, 10]
    # comp_to_group = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8]

    selected_companies = [1, 2, 3, 4, 7, 8]
    comp_to_group = [0, 1, 2, 3, 4, 4]

    comp = meta_data["production_companies"].value_counts().index[selected_companies]
    comp_dict = dict([(x, comp_to_group[i]) for i, x in enumerate(comp)])
    meta_data = meta_data.astype({"id": "int"})
    meta_data = meta_data[meta_data["production_companies"].isin(comp)]

    return meta_data, comp_dict

def select_movies(ratings, meta_data, n_movies = 100, n_user= 10000):

    # Use the 100 Movies with the most ratings
    po2 = ratings["movieId"].value_counts()

    #Select the n_movies with highest variance
    var_scores = [np.std(ratings[ratings["movieId"].isin([x])]["rating"]) for x in po2.index[:(n_movies*3)]]
    var_sort = np.argsort(var_scores)[::-1]

    selected_movies = po2.index[var_sort[:n_movies]]
    ratings = ratings[ratings["movieId"].isin(selected_movies)]
    meta_data = meta_data[(meta_data["id"].isin(selected_movies))]

    po = ratings["userId"].value_counts()

    ratings = ratings[ratings["userId"].isin(po.index[:n_user])] # remove users with less than 10 votes
    meta_data = meta_data[meta_data["id"].isin(ratings["movieId"].value_counts().index[:])]

    selected_movies_ids = meta_data[['id', 'production_companies']].drop_duplicates()

    return ratings, meta_data, selected_movies_ids

def get_ranking_matrix_incomplete(ratings, meta_data, n_user):

    user_id_to_idx = dict(zip(sorted(ratings["userId"].unique()), np.arange(n_user)))
    # Create a single Ranking Matrix, only relevance for rated movies
    # Leave it incomplete
    ranking_matrix = np.zeros((n_user, len(meta_data["id"])))
    movie_id_to_idx = {}
    movie_idx_to_id = []
    print(np.shape(ranking_matrix))
    for i, movie in enumerate(meta_data["id"]):
        movie_id_to_idx[movie] = i
        movie_idx_to_id.append(movie)
        single_movie_ratings = ratings[ratings["movieId"].isin([movie])]
        ranking_matrix[[user_id_to_idx[x] for x in single_movie_ratings["userId"]], i] = single_movie_ratings[
            "rating"]

    return ranking_matrix, np.array(list(user_id_to_idx.keys()))

def get_matrix_factorization(ratings, meta_data, n_user, n_movies):
    # Matrix Faktorization
    algo = surprise.SVD(n_factors=50, biased=False)
    reader = surprise.Reader(rating_scale=(0.5, 5))
    surprise_data = surprise.Dataset.load_from_df(ratings[["userId", "movieId", "rating"]],
                                                  reader).build_full_trainset()
    algo.fit(surprise_data)

    pred = algo.test(surprise_data.build_testset())
    print("MSE: ", surprise.accuracy.mse(pred))
    print("RMSE: ", surprise.accuracy.rmse(pred))

    ranking_matrix = np.dot(algo.pu, algo.qi.T)
    movie_idx_to_id = [surprise_data.to_raw_iid(x) for x in range(n_movies)]
    features_matrix_factorization = algo.pu
    print("Means: ", np.mean(features_matrix_factorization), np.mean(algo.qi.T))
    print("Feature STD:", np.std(features_matrix_factorization), np.std(algo.qi))
    print("Full Matrix Shape", np.shape(ranking_matrix), "rankinG_shape", np.shape(ranking_matrix))

    return ranking_matrix, features_matrix_factorization, movie_idx_to_id


def load_movie_data(n_user, n_movies, docs_ids, movie_ranking_sample_file=None, group_by='comp'):
    meta_data = pd.read_csv("data/movies_metadata.csv")[["production_companies", "id", "genres"]]

    # meta_data = meta_data.drop([19730, 29503, 35587]) # No int id
    # Delete Movies with non-int ID
    id_list = meta_data['id']
    for idx, i in enumerate(id_list):
        try:
            idd = int(i.strip())
        except:
            print(idx)
            meta_data = meta_data.drop([idx])

    meta_data = meta_data.loc[meta_data['id'].isin(docs_ids.astype(str))]

    meta_data, g_idx = define_genre(meta_data)
    if group_by == 'comp':
        meta_data, G_dict = select_companies(meta_data)
    else:
        meta_data = meta_data.explode('genres')
        meta_data, G_dict = select_genres(meta_data)

    ratings_full = pd.read_csv("data/ratings.csv")
    ratings = ratings_full[ratings_full["movieId"].isin(meta_data["id"])]
    ratings, meta_data, selected_movies_ids = select_movies(ratings, meta_data, n_movies=n_movies, n_user=n_user)

    ranking_matrix, user_idx_to_id = get_ranking_matrix_incomplete(ratings, selected_movies_ids, n_user)
    full_matrix, features_matrix_factorization, movie_idx_to_id = get_matrix_factorization(ratings, meta_data, n_user, n_movies)

    full_matrix[np.nonzero(ranking_matrix)] = ranking_matrix[np.nonzero(ranking_matrix)]
    user_features = features_matrix_factorization
    prob_matrix = sigmoid(full_matrix)
    groups = []
    for x in movie_idx_to_id:
        gen_for_single_movie = []
        for genre in meta_data[meta_data["id"].isin([x])]['genres' if group_by == 'genres' else 'production_companies'].to_list():
            gen_for_single_movie.append(G_dict[genre])
        groups.append(gen_for_single_movie)

    po = ratings["userId"].value_counts()
    po2 = ratings["movieId"].value_counts()
    print("Number of Users", len(po.index), "Number of Movies", len(po2.index))
    print("the Dataset before completion is", len(ratings) / float(n_user * n_movies), " filled")
    print("The most rated movie has {} votes, the least {} votes; mean {}".format(po2.max(), po2.min(), po2.mean()))
    print("The most rating user rated {} movies, the least {} movies; mean {}".format(po.max(), po.min(), po.mean()))

    print(n_user, n_movies)
    if movie_ranking_sample_file:
        sample_dir = '/'.join(movie_ranking_sample_file.split('/')[:2])
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        for i in range(10):
            random_matrix = np.random.rand(n_user, n_movies)
            np.save(movie_ranking_sample_file+"{}.npy".format(i), [np.asarray(prob_matrix > random_matrix, dtype=np.float16), user_features, groups, user_idx_to_id, movie_idx_to_id])

    return prob_matrix, user_features, groups, user_idx_to_id, movie_idx_to_id

def load_movie_data_saved(filename ="data/movie_data_prepared.npy"):
    """
    Load an already created Movie Rating Matrix
    """
    full_matrix, user_features, groups, user_idx_to_id, movie_idx_to_id = np.load(filename, allow_pickle=True)
    return full_matrix, user_features, groups, user_idx_to_id, movie_idx_to_id

def sample_user_movie(MOVIE_RATING_FILE):
    """
    Yielding a Movie
    """
    ranking, features, _, user_idx_to_id, movie_idx_to_id = load_movie_data_saved(MOVIE_RATING_FILE)
    print(np.shape(ranking))
    while True:
        random_order = np.random.permutation(np.shape(ranking)[0])
        for i in random_order:
            yield (ranking[i,:], features[i,:])

def sample_user_base(distribution, alpha =0.5, beta = 0.5, u_std=0.3, BI_LEFT = 0.5):
    """
    Returns a User of the News Platform
    A user cosists of is Polarity and his Openness
    """
    if(distribution == "beta"):
        u_polarity = np.random.beta(alpha, beta)
        u_polarity *= 2
        u_polarity -= 1
        openness = u_std
    elif(distribution == "discrete"):
        #3 Types of user -1,0,1. The neutral ones are more open
        u_polarity = np.random.choice([-1,0,1])
        if(u_polarity == 0):
            openness = 0.85
        else:
            openness = 0.1
    elif(distribution == "bimodal"):
        if np.random.rand() < BI_LEFT:
            u_polarity = np.clip(np.random.normal(0.5,0.2,1),-1,1)
        else:
            u_polarity = np.clip(np.random.normal(-0.5, 0.2, 1), -1, 1)
        openness = np.random.rand()/2 + 0.05 #Openness uniform Distributed between 0.05 and 0.55
    else:
        print("please specify a distribution for the user")
        return (0,1)
    return np.asarray([u_polarity, openness])

def assign_groups(items):
    n_groups = max([i.g for i in items])+1
    G = [ [] for i in range(n_groups)]
    for i, item in enumerate(items):
        G[item.g].append(i)
    return G