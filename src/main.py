import os
import scipy.io
import argparse

from data_loader import get_data_loader
from seed import set_seed

logger_name_config = [
    {
        'key': 'user_counts',
        'name': 'USER'
    },
    {
        'key': 'mask_rate',
        'name': 'MASK'
    },
    {
        'key': 'top_k',
        'name': 'TOPK'
    },
    {
        'key': 'start_doc_num',
        'name': 'START'
    },
    {
        'key': 'doc_entry_type',
        'name': 'ENTRY'
    },
    {
        'key': 'even',
        'name': 'EVEN'
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='exp', help="Experiment name")
    parser.add_argument("--seed", type=int, default=7, help="Seed")
    parser.add_argument("--trials", type=int, default=1, help="Trial number")
    parser.add_argument("--user_counts", type=int, default=4000, help="User counts")
    parser.add_argument("--methods", type=str, default=[], nargs='+', help="Methods")
    parser.add_argument("--mask_rate", type=float, default=0.0, help="Mask rate")
    parser.add_argument("--cot_top_rate", type=float, default=0.5, help="Cot top rate")
    parser.add_argument("--tradeoff", type=float, default=None, help="FairCo lambda")
    parser.add_argument("--top_k", type=int, default=[], nargs='+', help="Top k")
    parser.add_argument("--dataset", type=str, default='movie_comp_200', help="Dataset")
    parser.add_argument("--start_doc_num", type=int, default=100, help="Start doc num")
    parser.add_argument("--doc_num", type=int, default=200, help="Doc num")
    parser.add_argument("--user_emb", type=int, default=50, help="User emb")
    parser.add_argument("--doc_entry_type", type=str, default='random', help="Doc entry type")
    parser.add_argument("--movie_emb_file", type=str, default='data/movie_emb.csv', help="Movie emb file")
    parser.add_argument("--user_emb_file", type=str, default='data/user_emb.csv', help="User emb file")
    parser.add_argument("--even", type=bool, default=False, help="Even", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()


    for trial in range(args.trials):
        set_seed(args.seed + trial)

        for method in args.methods:
            user_counts = args.user_counts
            MOVIE_RATING_FILE = f"{args.dataset}/{trial}.npy"

            docs, future_docs, all_docs, start_popularity, numerical_relevances, user_counts, G, movie_idx_to_id = get_data_loader(
                MOVIE_RATING_FILE,
                user_counts, 
                args.start_doc_num
            )


            print(f'Running {method}...')
            # Prepare the settings and data for each model/method
            params = {
                'docs': docs,
                'future_docs': future_docs,
                'popularity': start_popularity,
                'numerical_relevances': numerical_relevances,
                'iterations': user_counts,
                'mask_rate': args.mask_rate, 
                'top_k': args.top_k,
                'user_emb_file': args.user_emb_file,
                'movie_emb_file': args.movie_emb_file, 
                'MOVIE_RATING_FILE': MOVIE_RATING_FILE,
                'all_docs': all_docs,
                'doc_entry_type': args.doc_entry_type,
                'even': args.even,
                'ranking_method': method,
                'user_emb_dim': args.user_emb,
                'movie_idx_to_id': movie_idx_to_id,
            }

            if 'FD_CoTeR' in method:
                from CoTeR import main
                params['top_rate'] = args.cot_top_rate
            elif 'FD_FairCo' in method:
                from FairCo import main
                
            # GO
            output = main(**params)

            # Save the output
            dir_name = [f'E_{args.exp_name}']
            for logger in logger_name_config:
                dir_name.append(f'{logger["name"]}_{str(getattr(args, logger["key"]))}')
            dir_name.append(f'M_{method}')
            dir_name = '_'.join(dir_name)
            dir_name = f'../output/results/{dir_name}'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            # save mat
            scipy.io.savemat(f'{dir_name}/{trial}.mat', output)
