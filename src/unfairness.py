import numpy as np

def get_unfairness(data, G):
    #From Rundata containing Relevance, Exposure and Clicks(Impact), 
    #we compute the the overall unfairness (summed over all pairs of Groups)
    iterations, _ = np.shape(data["NDCG"])
    pair_group_combinations = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G))]
    overall_fairness = np.zeros((iterations, 4))
    for a, b in pair_group_combinations:
        overall_fairness[:, 0] += np.abs(
            data["prop"][:, a] / data["rel"][:, a] - data["prop"][:, b] / data["rel"][:, b])
        overall_fairness[:, 1] += np.abs(
            data["prop"][:, a] / data["true_rel"][:, a] - data["prop"][:, b] / data["true_rel"][:, b])
        overall_fairness[:, 2] += np.abs(
            data["clicks"][:, a] / data["rel"][:, a] - data["clicks"][:, b] / data["rel"][:, b])
        overall_fairness[:, 3] += np.abs(
            data["clicks"][:, a] / data["true_rel"][:, a] - data["clicks"][:, b] / data["true_rel"][:,b])
            
    return overall_fairness