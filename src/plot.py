import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

def select(query, src):
    res = []
    for r in src:
        if all(q in r for q in query) and 'Half' not in r and '0.0' not in r:
            res.append(r)
    return res

if __name__ == '__main__':
    trials = 5
    root = os.getcwd()
    results_path = os.path.join(root, 'output/results')
    plot_path = os.path.join(root, 'output/plots')
    results_list = os.listdir(results_path)

    metric = 'fair'

    nfd_res = select(
        [
            'START_100',
            'NFD'
        ],
        results_list
    )

    ltr_res = select(
        [
            'START_50',
            'random',
            'EVEN_True',
            'FD',
        ],
        results_list
    )

    methods = [
        'FairCo',
        'CoTeR'
    ]

    row_indc = ['FD']
    col_indc = ['0.2', '0.3', '0.4']

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    red_idx = 3
    highkight_idx = len(methods) - 1
    tmp = colors[red_idx].copy()
    colors[red_idx] = colors[highkight_idx]
    colors[highkight_idx] = tmp

    grid_data = []

    for r, row in enumerate(row_indc):
        r_data = []
        for c, col in enumerate(col_indc):
            grid_item_data = {}
            path = []
            for p in nfd_res + ltr_res:
                if row in p and col in p:
                    path.append(p)
            for m in methods:
                if m not in grid_item_data:
                    grid_item_data[m] = {}
                m_ndcg = []
                m_fair = []
                for p in path:
                    if m in p:
                        # print(p)
                        for i in range(trials):
                            # print(results_path, f'{p}/{i}.mat')
                            data = loadmat(os.path.join(results_path, f'{p}/{i}.mat'))
                            # print(data['NDCG'])
                            m_ndcg.append(data['NDCG'])
                            m_fair.append(data['overall_fairness'])
                m_ndcg = np.array(m_ndcg)
                m_fair = np.array(m_fair)
                grid_item_data[m]['ndcg'] = m_ndcg
                grid_item_data[m]['fair'] = m_fair
                # print(m_ndcg.shape)
                # print(m_fair.shape)
            r_data.append(grid_item_data)
        # print(len(r_data))

        grid_data.append(r_data)
    fontsize = 9
    fig, ax = plt.subplots(1, 3, figsize=(12, 2.5))
    # fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(12, 5))
    for r, row in enumerate(grid_data):
        for c, col in enumerate(row):
            for mid, method in enumerate(methods):
                if metric == 'ndcg':
                    print(col[method]['ndcg'])
                    data = col[method]['ndcg'][:,:,0].copy()
                else:
                    data = col[method]['fair'][:,:,1].copy()
                trials, n = np.shape(data)
                plot_data = np.cumsum(data, axis=1)/ np.arange(1,n+1)
                    
                mean = np.mean(plot_data, axis=0)
                std = np.std(plot_data, axis=0)

                if metric == 'ndcg':
                    ax[c].set_ylim([0.82,0.92])
                else:
                    ax[c].set_ylim([0.2, 1.2])
                ax[c].set_xlim([0, n])
                # ax.set_xlim([0,n])
                ax[c].set_xlabel("Users", fontsize=fontsize)
                # ax[r, c].set_ylabel(r"NDCG@10", fontsize=fontsize)

                linestyle='-'
                if 'FairCo' in method:
                    label = 'FairCo'
                if 'CoTeR' in method:
                    if row_indc[0] == 'NFD':
                        label = 'CoTeR-s'
                    else:
                        label = 'CoTeR-e'

                line = ax[c].plot(mean, label=label, linestyle=linestyle, c=colors[mid])
                line = line[0]
                # ax.plot(mean, label=method.split('?')[1])
                fill = ax[c].fill_between(
                    range(len(mean)),
                    mean - std,
                    mean + std,
                    alpha=0.1,
                    color=line.get_color()
                )

    # ax[0].legend()

    # fig.text(0.5, 0, 'Users', ha='center', fontsize=fontsize)
    if metric == 'ndcg':
        fig.text(0, 0.5, r"NDCG@10", va='center', rotation='vertical', fontsize=fontsize)
    else:
        fig.text(0, 0.5, r"Unfairness (UF)", va='center', rotation='vertical', fontsize=fontsize)
    
    fig_legend = ax[0].get_legend_handles_labels()[0]
    fig.legend(handles=fig_legend, ncol=len(methods), bbox_to_anchor=(0.5, 1.09), loc="upper center", fontsize=fontsize)
    # plt.legend(ncol=len(methods), loc="upper center", fontsize=fontsize)
    plt.tight_layout()
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(
        plot_path + f'/{metric}.pdf',
        bbox_inches='tight', 
        pad_inches=0,
        dpi=600
    )
    plt.close(fig)       