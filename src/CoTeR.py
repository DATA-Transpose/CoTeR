import math
import numpy as np

from tqdm import tqdm

import nueral_coteaching as NUCOT
import data_utils as DU
import simulation as DSIM
import data_managers as DM
from unfairness import get_unfairness


def sample_round_training_data(users, docs, train_x_map, w_pophist, start_user_num, start_doc_num, movie_idx_to_id, iter):
    sampled_users = users[-start_user_num:]
    sampled_docs = docs[-start_doc_num:]
    train_x = [train_x_map[(u[1], movie_idx_to_id[d.id])] for u in sampled_users for d in sampled_docs]
    train_x = np.array(train_x)

    train_relevance = w_pophist[:iter + 1] - np.concatenate((np.zeros((1, len(docs))), w_pophist[:iter]))
    train_y = [train_relevance[ui, di] for ui in range(len(users) - start_user_num, len(users)) for di in range(len(docs) - start_doc_num, len(docs))]
    train_y = np.array(train_y)

    sampled_users = users[:-start_user_num]
    sampled_docs = docs[:-start_doc_num]
    old_train_x = [train_x_map[(u[1], movie_idx_to_id[d.id])] for u in sampled_users for d in sampled_docs]
    old_train_x = np.array(old_train_x)

    old_train_y = [train_relevance[ui, di] for ui in range(len(users) - start_user_num) for di in range(len(docs) - start_doc_num)]
    old_train_y = np.array(old_train_y)

    old_sample_index_list = np.arange(len(old_train_y))
    np.random.shuffle(old_sample_index_list)
    old_sample_index_list = old_sample_index_list[:len(train_y)]
    old_train_x = old_train_x[old_sample_index_list]
    old_train_y = old_train_y[old_sample_index_list]

    if len(old_train_x) > 0:
        train_x = np.concatenate((old_train_x, train_x))
        train_y = np.concatenate((old_train_y, train_y))

    return train_x, train_y

def next_doc_in_at(iterations, iter, future_docs):
    new_doc_in_before = iterations - (iterations / 10)
    interval = math.floor((new_doc_in_before - iter) / len(future_docs))
    return iter + interval - 1

def uneven_next_doc_in_at(iterations, iter, future_docs):
    new_doc_in_before = iterations - (iterations / 10)
    interval = math.ceil((new_doc_in_before - iter) / len(future_docs))
    if iter >= new_doc_in_before:
        return -1
    return iter + interval

def main(
    docs: list,
    popularity,
    iterations,
    future_docs: list,
    numerical_relevances,
    movie_idx_to_id,
    ranking_method='',
    top_k=[100],
    user_emb_file='',
    movie_emb_file='',
    MOVIE_RATING_FILE='',
    mask_rate=0,
    all_docs=[],
    doc_entry_type='',
    user_emb_dim=0,
    top_rate=0.5,
    even=1):

    mask_rate = float(mask_rate)
    if mask_rate > 0:
        forget_rate = mask_rate
    else:
        forget_rate = 0.4
        
    users_manager = DM.UserSampler(user_emb_file, MOVIE_RATING_FILE=MOVIE_RATING_FILE, mask_rate=mask_rate)
    docs_manager = DM.DocumentSampler(movie_emb_file)
    docs_manager.set_doc_generator(all_docs, docs, future_docs, doc_entry_type)

    train_x_map = {}
    for i in tqdm(range(10000)):
        user = users_manager.get_next()
        for doc in all_docs:
            if user_emb_dim == 50:
                train_x_map[(user[1], movie_idx_to_id[doc.id])] = np.concatenate((user[2], docs_manager[movie_idx_to_id[doc.id]]))
            if user_emb_dim == 100:
                train_x_map[(user[1], movie_idx_to_id[doc.id])] = np.concatenate((users_manager[user[1]], docs_manager[movie_idx_to_id[doc.id]]))

    users = []
    G = DSIM.assign_groups(docs)
    cum_exposure = np.zeros(len(docs))
    runtime_relevances = np.asarray(popularity, dtype=np.float32)
    popularity = np.asarray(popularity)
    aff_scores = np.zeros((iterations, len(docs))) # for skyline

    pophist = np.zeros((iterations, len(docs)))
    p_pophist = np.zeros((iterations, len(docs)))
    w_pophist = np.zeros((iterations, len(docs)))
    realtime_ranking_hist = np.zeros((iterations, len(docs)))
    real_relevance_hist = np.zeros((iterations, len(docs)))
    NDCG = np.zeros((iterations, len(top_k)))

    group_prop = np.zeros((iterations, len(G)))
    group_clicks = np.zeros((iterations, len(G)))
    group_rel = np.zeros((iterations, len(G)))
    true_group_rel = np.zeros((iterations, len(G)))
    group_prop_k = np.zeros((iterations, len(top_k), len(G)))
    group_clicks_k = np.zeros((iterations, len(top_k), len(G)))
    group_rel_k = np.zeros((iterations, len(top_k), len(G)))
    true_group_rel_k = np.zeros((iterations, len(top_k), len(G)))

    count = np.zeros(len(docs))
    rel_count = np.zeros(len(docs))
    relevances = np.zeros(len(docs))
    nn_errors = np.zeros(iterations)
    net = None

    new_doc_in_at = -1
    new_doc_cache = []
    new_event_count = 0 
    delta_r = np.zeros(iterations)

    for iter in tqdm(range(iterations)):
        in_iter_doc_list = np.array([d.id for d in docs])
        def DatasetId_to_InIterId(dataset_ids):
            return np.where(in_iter_doc_list == np.array(dataset_ids)[:, None])[-1]

        count += 1
        rel_count += 1
        user = users_manager.get_next()
        users.append(user)
        ground_truth_probs = DSIM.affinity_score(user, docs)
        aff_probs = np.asarray([user[3][x.id] for x in docs])
        relevances += aff_probs

        if iter < 100:
            ranking = DSIM.IPS_rank(runtime_relevances / count)
        else:
            G_in_iter = [DatasetId_to_InIterId(G[i]) for i in range(len(G))]
            predict_x = [train_x_map[(user[1], movie_idx_to_id[d.id])] for d in docs]
            predict_x = np.array(predict_x)
            if 'CoTeR' in ranking_method:
                fairness_error = DSIM.get_unfairness(cum_exposure, runtime_relevances / count, G_in_iter, error=True)
                ranking = DSIM.neural_rank(net, predict_x, fairness_error)
            if ('LTR_Skyline' in ranking_method) or ('LTR_Pers' in ranking_method):
                ranking = DSIM.neural_rank(net, predict_x)

        propensities = DSIM.position_bias(ranking)
        popularity, runtime_relevances = DSIM.click(popularity, runtime_relevances, aff_probs, propensities)

        cum_exposure += propensities
        pophist[iter, :] = popularity
        w_pophist[iter, :] = runtime_relevances
        if ('LTR_Skyline' in ranking_method):
            aff_scores[iter] = ground_truth_probs

        if iter == 99:
            start_doc_num = len(docs)
            start_user_num = len(users)
            train_x = [train_x_map[(u[1], movie_idx_to_id[d.id])] for u in users for d in docs]
            train_x = np.array(train_x)
            
            if ('LTR_Skyline' in ranking_method):
                train_relevance = aff_scores[:iter + 1]
                train_y = [train_relevance[ui, di] for ui in range(len(users)) for di in range(len(docs))]
                train_y = np.array(train_y)
                net = NUCOT.COTNET(np.shape(train_x)[1], hidden_dim=64, out_dim=1)
            else:
                train_relevance = w_pophist[:iter + 1] - np.concatenate((np.zeros((1, len(docs))), w_pophist[:iter]))
                train_y = [train_relevance[ui, di] for ui in range(len(users)) for di in range(len(docs))]
                train_y = np.array(train_y)
                net = NUCOT.COTNET(np.shape(train_x)[1], hidden_dim=64, out_dim=1)
            net.train(train_x, train_y, epochs=1000, trial=iter, doc_num=len(docs), top_rate=top_rate, forget_rate=forget_rate)

        elif (iter > 99 and iter % 100 == 99):
            train_x, train_y = sample_round_training_data(users, docs, train_x_map, w_pophist, start_user_num, start_doc_num, movie_idx_to_id, iter)
            net.train(train_x, train_y, epochs=20, trial=iter, doc_num=len(docs), top_rate=top_rate, forget_rate=forget_rate)

        if iter >= 99:
            predict_x = [train_x_map[(user[1], movie_idx_to_id[d.id])] for d in docs]
            predict_x = np.array(predict_x)
            predicted_relevances = net.predict(predict_x)
        
        if iter >= 99:
            p_pophist[iter, :] = predicted_relevances
            nn_errors[iter] = np.mean((predicted_relevances - ground_truth_probs) ** 2)
            delta_r[iter] = DSIM.sMAPE(relevances / count, predicted_relevances)
        else:
            p_pophist[iter, :] = runtime_relevances
            delta_r[iter] = DSIM.sMAPE(relevances / count, runtime_relevances / count)

        NDCG[iter] = DSIM.get_ndcg_score_top_k(ranking, ground_truth_probs, top_k)[0]
        realtime_ranking_hist[iter, :] = ranking
        real_relevance_hist[iter, :] = ground_truth_probs

        for k_idx, k in enumerate(top_k):
            top_k_ranking = ranking[:k]
            group_prop_k[iter, k_idx, :] = [np.sum(cum_exposure[np.intersect1d(top_k_ranking, DatasetId_to_InIterId(G[i]))]) for i in range(len(G))]
            group_clicks_k[iter, k_idx, :] = [np.sum(popularity[np.intersect1d(top_k_ranking, DatasetId_to_InIterId(G[i]))]) for i in range(len(G))]
            group_rel_k[iter, k_idx, :] = [np.sum(pophist[iter, np.intersect1d(top_k_ranking, DatasetId_to_InIterId(G[g]))]) for g in range(len(G))]
            top_k_ranking = [docs[d].id for d in top_k_ranking]
            true_group_rel_k[iter, k_idx, :] = [np.sum(numerical_relevances[np.intersect1d(top_k_ranking, G[g])]) * np.max(count) for g in range(len(G))]        


        group_prop[iter, :] = [np.sum(cum_exposure[DatasetId_to_InIterId(G[i])]) for i in range(len(G))]
        group_clicks[iter, :] = [np.sum(popularity[DatasetId_to_InIterId(G[i])]) for i in range(len(G))]
        group_rel[iter, :] = [np.sum(p_pophist[iter, DatasetId_to_InIterId(G[g])]) for g in range(len(G))]
        true_group_rel[iter, :] = [np.sum(numerical_relevances[G[g]]) * np.max(count) for g in range(len(G))]
        

        # new doc event
        if even:
            if len(future_docs) > 0:
                if new_doc_in_at < 0:
                    new_doc_in_at = next_doc_in_at(iterations, iter, future_docs)
                if new_doc_in_at == iter:
                    new_doc = docs_manager.get_next()

                    count = np.append(count, np.max(count))
                    rel_count = np.append(rel_count, 0)
                    relevances = np.append(relevances, 0)
                    docs_id_in_same_group = []
                    for g in new_doc.g:
                        for d in G[g]:
                            docs_id_in_same_group.append(d)
                    docs_id_in_same_group = np.array(list(set(docs_id_in_same_group)))
                    
                    in_iter_docs_in_same_group = np.where(in_iter_doc_list == docs_id_in_same_group[:, None])[-1]
                    cum_exposure = np.append(cum_exposure, np.mean(cum_exposure[in_iter_docs_in_same_group]))
                    popularity = np.append(popularity, np.mean(popularity[in_iter_docs_in_same_group]))
                    runtime_relevances = np.append(runtime_relevances, np.mean(runtime_relevances[in_iter_docs_in_same_group]))
                    p_pophist = np.column_stack((p_pophist, np.mean(p_pophist[:, in_iter_docs_in_same_group], axis=1)))
                    pophist = np.column_stack((pophist, np.mean(pophist[:, in_iter_docs_in_same_group], axis=1)))
                    w_pophist = np.column_stack((w_pophist, np.mean(w_pophist[:, in_iter_docs_in_same_group], axis=1)))
                    realtime_ranking_hist = np.column_stack((realtime_ranking_hist, np.array([-1 for i in range(iterations)])))
                    real_relevance_hist = np.column_stack((real_relevance_hist, np.array([-1 for i in range(iterations)])))
                    if ('LTR_Skyline' in ranking_method):   
                        padding = np.array([u[0][new_doc.id] for u in users] + [0 for i in range(iterations - len(users))])
                        aff_scores = np.column_stack((aff_scores, padding.T))

                    docs.append(new_doc)
                    new_doc_cache.append(new_doc)
                    future_docs.remove(new_doc)
                    new_doc_in_at = next_doc_in_at(iterations, iter, future_docs) if len(future_docs) > 0 else 0
                    G = DSIM.assign_groups(docs)

                    if len(new_doc_cache) >= 10:
                        train_x, train_y = sample_round_training_data(users, docs, train_x_map, w_pophist, start_user_num, start_doc_num, movie_idx_to_id, iter)
                        net.train(train_x, train_y, epochs=20, trial=iter, doc_num=len(docs), top_rate=top_rate, forget_rate=forget_rate)
                        new_doc_cache = []
                        
                    new_event_count += 1
        else:
            if len(future_docs) > 0:
                if new_doc_in_at < 0:
                    new_doc_in_at = uneven_next_doc_in_at(iterations, iter, future_docs)
                if new_doc_in_at == iter or new_doc_in_at == -1:
                    if new_doc_in_at == -1:
                        new_doc_num = len(future_docs)
                    else:
                        new_doc_num = np.random.choice(5, 1)[0]
                        new_doc_num = new_doc_num if (new_doc_num <= len(future_docs)) else len(future_docs)
                    if new_doc_num > 0:
                        new_docs = []
                        for nd_idx in range(new_doc_num):
                            new_docs.append(docs_manager.get_next())

                        count = np.append(count, np.array([np.max(count)] * len(new_docs)))
                        rel_count = np.append(rel_count, np.array([np.max(count)] * len(new_docs)))
                        relevances = np.append(relevances, np.array([np.max(count)] * len(new_docs)))
                        docs_id_in_same_group = []
                        for nd in new_docs:
                            docs_id_in_same_group_for_single = []
                            for g in nd.g:
                                for d in G[g]:
                                    docs_id_in_same_group_for_single.append(d)
                            docs_id_in_same_group.append(np.array(list(set(docs_id_in_same_group_for_single))))
                        docs_id_in_same_group = np.array(docs_id_in_same_group)

                        for docs_id_in_same_group_for_single in docs_id_in_same_group:
                            in_iter_docs_in_same_group = np.where(in_iter_doc_list == docs_id_in_same_group_for_single[:, None])[-1]
                            cum_exposure = np.append(cum_exposure, np.mean(cum_exposure[in_iter_docs_in_same_group]))
                            popularity = np.append(popularity, np.mean(popularity[in_iter_docs_in_same_group]))
                            runtime_relevances = np.append(runtime_relevances, np.mean(runtime_relevances[in_iter_docs_in_same_group]))
                            p_pophist = np.column_stack((p_pophist, np.mean(p_pophist[:, in_iter_docs_in_same_group], axis=1)))
                            pophist = np.column_stack((pophist, np.mean(pophist[:, in_iter_docs_in_same_group], axis=1)))
                            w_pophist = np.column_stack((w_pophist, np.mean(w_pophist[:, in_iter_docs_in_same_group], axis=1)))
                            realtime_ranking_hist = np.column_stack((realtime_ranking_hist, np.array([-1 for i in range(iterations)])))
                            real_relevance_hist = np.column_stack((real_relevance_hist, np.array([-1 for i in range(iterations)])))
                        
                        for new_doc in new_docs:
                            if ('LTR_Skyline' in ranking_method):   
                                padding = np.array([u[0][new_doc.id] for u in users] + [0 for i in range(iterations - len(users))])
                                aff_scores = np.column_stack((aff_scores, padding.T))

                            docs.append(new_doc)
                            new_doc_cache.append(new_doc)
                            future_docs.remove(new_doc)
                        G = DSIM.assign_groups(docs)

                        if len(new_doc_cache) >= 10 and net != None:
                            train_x, train_y = sample_round_training_data(users, docs, train_x_map, w_pophist, start_user_num, start_doc_num, movie_idx_to_id, iter)
                            net.train(train_x, train_y, epochs=20, trial=iter, doc_num=len(docs), top_rate=top_rate, forget_rate=forget_rate)
                            new_doc_cache = []
                                
                        new_event_count += 1
                    new_doc_in_at = uneven_next_doc_in_at(iterations, iter, future_docs) if len(future_docs) > 0 else 0

    fairness_hist = {
        "prop": group_prop, 
        "clicks": group_clicks, 
        "rel": group_rel, 
        "true_rel": true_group_rel, 
        "NDCG": NDCG, 
        "realtime_ranking_hist": realtime_ranking_hist, 
        "real_relevance_hist": real_relevance_hist, 
        "delta_r": delta_r,
        "group_prop_k": group_prop_k[:, 0, :],
        "group_clicks_k": group_clicks_k[:, 0, :],
        "group_rel_k": group_rel_k[:, 0, :],
        "true_group_rel_k": true_group_rel_k[:, 0, :]
    }
    
    fairness_hist['overall_fairness'] = get_unfairness(fairness_hist, G)
    return fairness_hist