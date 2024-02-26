import copy
import math
import random
# import torch
import numpy as np

import data_utils as DU
import data_managers as DM


def sMAPE(y, yp):
    return np.mean(np.abs(y - yp) / ((y + yp) / 2))

def mask_interactions(interaction, rate=0.2):
    # the function to mask interaction vector
    # to simulate unseen interest of user
    if rate == 0:
        return interaction, interaction
    n = len(interaction)
    mask_1 = np.ones(n)
    mask_2 = np.ones(n)
    mask_count = int(n * rate)
    perm = np.random.permutation(n)
    mask_1[perm[:mask_count]] = 0
    mask_2[perm[mask_count:2 * mask_count]] = 0
    mask_1 *= interaction
    mask_2 *= interaction
    return mask_1, mask_2

def sigmoid(x):
    return (x / (1 + np.abs(x)))

def minmax(v):
    return (v - v.min()) / ((v.max() - v.min()) + 1e-6)

def get_numerical_relevances(MOVIE_RATING_FILE=""):
    ranking, _, _, _, _ = DU.load_movie_data_saved(MOVIE_RATING_FILE)
    return np.mean(ranking, axis=0)  # Mean over all users

def affinity_score(user, items):
    if (type(items) == list):
        return np.asarray([user[0][x.id] for x in items])
    else:
        return user[0][items.id]

def IPS_rank(weighted_popularity):
    return np.argsort(weighted_popularity)[::-1]

def neural_rank(net, user_doc, e_p=0, KP=0.01):
    relevances = net.predict(user_doc)
    return np.argsort(relevances + KP * e_p)[::-1]

def get_maximu_G_id(sorted_G,personal_relevance):
    max_G_ind=-1000
    max_i_value=-1000
    for ind,i in enumerate(sorted_G):
        if len(i)>0:
            if max_i_value<personal_relevance[i[0]]:
                max_i_value=personal_relevance[i[0]] 
                max_G_ind=ind
    return max_G_ind

def max_min_loss(a):
    return np.max(a) - np.min(a)

def neural_rank_top_down(net, user_doc, weighted_popularity, G, rank_group_split):
    personal_relevance = net.predict(user_doc)
    average_rele=[]
    pos_bias = 1 / (np.log2(2 + np.arange(weighted_popularity.shape[0]))) 
    pos_bias /= np.max(pos_bias)
    for i in G:
        average_rele.append(np.mean(weighted_popularity[i]))
    average_rele = np.array(average_rele)

    rank_group_split_copy = copy.deepcopy(rank_group_split)

    G_size=np.array([len(G_i) for G_i in G])
    sorted_G=[]
    def func(G):
        return personal_relevance[G]
    for i in G:
        sorted_G.append(sorted(i, key=func,reverse=True))

    def cum_id(rank_group_split_copy,k):
        product=rank_group_split_copy * pos_bias[np.newaxis,:]
        rank_group_split_cum_exposure=np.cumsum(product,axis=1)[:,k]/average_rele/G_size
        sorted_group_id=rank_group_split_cum_exposure.argsort(0)
        return sorted_group_id

    rangking=[]
    lambda_random_whole=2
    lambda_random_whole_part=0.6

    ranking_relevance=np.argsort(personal_relevance)[::-1]
    sorted_group_id_list=[]
    G_id_ranking_list=[]
    if random.random()>lambda_random_whole:
        return ranking_relevance
    else:
        for i in range(weighted_popularity.shape[0]):
            if random.random()>lambda_random_whole_part:
                G_id = get_maximu_G_id(sorted_G,personal_relevance)
                rangking.append(sorted_G[G_id].pop(0))
                rank_group_split_copy[G_id,i] += 1
            else:
                sorted_group_id = cum_id(rank_group_split_copy,i)
                sorted_group_id_list.append(sorted_group_id)
                for j in range(len(G)):
                    if sorted_G[sorted_group_id[j]]:
                        rangking.append(sorted_G[sorted_group_id[j]].pop(0))
                        rank_group_split_copy[sorted_group_id[j],i] += 1
                        break
        ranking = np.array(rangking).astype(int)
        return ranking


def position_bias(ranking):
    n = len(ranking)
    pos = 1/(np.log2(2+np.arange(n)))
    pos /= np.max(pos)
    pos_prob = np.zeros(n)
    pos_prob[ranking] = pos
    propensities = np.copy(pos_prob)
    return propensities

def click(popularity, runtime_relevances, ground_truth_probs, propensities):
    rand_var = np.random.rand(len(ground_truth_probs))
    rand_prop = np.random.rand(len(propensities))
    viewed = rand_prop < propensities
    clicks = np.logical_and(rand_var < ground_truth_probs, viewed)
    popularity += clicks
    runtime_relevances += clicks / (propensities + 1e-5)
    return popularity, runtime_relevances

def DatasetId_to_InIterId(in_iter_doc_list, dataset_ids):
    return np.where(in_iter_doc_list == np.array(dataset_ids)[:, None])[-1]

def get_unfairness(clicks, rel, G, error=False):
    """
    Get the Unfairess
    Input Clicks (Cum_Exposure for Exposure Unfairness, Clicks for Impact Unfairness)
    If Error, we return the difference to the best treated group,
    Otherwise just return the Exposure/Impact per Relevance
    """
    n = len(clicks)
    group_clicks = [sum(clicks[G[i]]) for i in range(len(G))]
    group_rel = [max(0.0001, sum(rel[G[i]])) for i in range(len(G))]
    group_fairness = [group_clicks[i] / group_rel[i] for i in range(len(G))]
    if (error):
        best = np.max(group_fairness)
        fairness_error = np.zeros(n)
        for i in range(len(G)):
            fairness_error[G[i]] = best - group_fairness[i]
        return fairness_error
    else:
        return group_fairness

def get_ndcg_score(ranking, true_relevances):
    dcg = np.sum(true_relevances[ranking] / np.log2(2+np.arange(len(ranking))))
    idcg = np.sum(np.sort(true_relevances)[::-1] / np.log2(2+np.arange(len(ranking))))
    if(idcg ==0):
        return 1
    return dcg / idcg

def get_ndcg_score_top_k(ranking, true_relevances,top_k_list=[1,3]):
    ndcg_top_k=[]
    for i in top_k_list:
        dcg = np.sum(true_relevances[ranking][:i] / np.log2(2+np.arange(len(ranking)))[:i])
        idcg = np.sum(np.sort(true_relevances)[::-1][:i] / np.log2(2+np.arange(len(ranking)))[:i])
        if(idcg == 0):
            ndcg_top_k.append(1)
        else:
            ndcg_top_k.append(dcg / idcg)
    ndcg_top_k = np.array(ndcg_top_k)
    return ndcg_top_k

def assign_groups(items):
    n_groups = max([g for i in items for g in i.g]) + 1
    G = [ [] for i in range(n_groups)]
    for i, item in enumerate(items):
        for g in item.g:
            G[g].append(item.id)
    for i, g in enumerate(G):
        G[i] = list(set(g))
    return G
