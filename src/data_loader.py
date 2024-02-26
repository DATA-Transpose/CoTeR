import numpy as np

import documents as DOC
import data_utils as DU
import simulation as DSIM

def get_data_loader(MOVIE_RATING_FILE, iterations, start_doc_num):

    _, _, groups, user_idx_to_id, movie_idx_to_id = DU.load_movie_data_saved(MOVIE_RATING_FILE)

    all_docs = []
    for i, g in enumerate(groups):
        all_docs.append(DOC.Movie(i, g))

    numerical_relevances = DSIM.get_numerical_relevances(MOVIE_RATING_FILE)

    selected_start_doc_ids = np.random.choice(len(all_docs), start_doc_num, replace=False)
    future_doc_ids = np.setdiff1d(np.array(range(len(all_docs))), selected_start_doc_ids)
    docs = np.array(all_docs)[selected_start_doc_ids].tolist()
    future_docs = np.array(all_docs)[future_doc_ids].tolist()
    start_popularity = np.ones(len(docs))
    G = DSIM.assign_groups(all_docs)

    return docs, future_docs, all_docs, start_popularity, numerical_relevances, iterations, G, movie_idx_to_id
