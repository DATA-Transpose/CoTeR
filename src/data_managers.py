import pandas as pd
import numpy as np

import data_utils as DU
import simulation as DSIM

class EmbeddingManager(object):
    def __init__(self, filename) -> None:
        super().__init__()
        try:
            self.data = pd.read_csv(filename)
            if 'amazon' not in filename:
                self.ids = self.data[self.data.columns[0]].to_numpy().astype(np.int32)
                self.embs = self.data[self.data.columns[1:]].to_numpy()
            else:
                self.ids = self.data[self.data.columns[0]].to_numpy()
                self.embs = self.data[self.data.columns[1:]].to_numpy()
        except:
            self.ids = []
            self.embs = []
        
    
    def idx2itemid(self, idx):
        return self.ids[idx]

    def itemid2idx(self, itemid):
        return np.where(self.ids == itemid)[0][0]

    def get_emb_item_id(self, itemid):
        return self.embs[self.itemid2idx(itemid), :]

    def __getitem__(self, n):
        return self.get_emb_item_id(n)

class UserSampler(EmbeddingManager):
    def __init__(self, filename, MOVIE_RATING_FILE, mask_rate=0) -> None:
        super(UserSampler, self).__init__(filename)
        self.ranking, self.user_feature, _, self.user_idx_to_id, self.movie_idx_to_id = DU.load_movie_data_saved(MOVIE_RATING_FILE)
        self.random_order = np.random.permutation(np.shape(self.ranking)[0])
        self.current_id = 0

        self.ranking_1 = []
        self.ranking_2 = [] 

        for r in self.ranking:
            r1, r2 = DSIM.mask_interactions(r, mask_rate)
            self.ranking_1.append(r1)
            self.ranking_2.append(r2)
        self.ranking_1 = np.array(self.ranking_1)
        self.ranking_2 = np.array(self.ranking_2)

        self.sample_user_generator = self.sample()

    def reset(self):
        self.current_id = 0

    def sample(self):
        while True:
            if self.current_id >= len(self.random_order):
                self.current_id = 0
            yield (
                self.ranking[self.random_order[self.current_id],:], 
                self.user_idx_to_id[self.random_order[self.current_id]], 
                self.user_feature[self.random_order[self.current_id], :],
                self.ranking_1[self.random_order[self.current_id],:],
                self.ranking_2[self.random_order[self.current_id],:]
            )
            self.current_id += 1

    def get_next(self):
        return next(self.sample_user_generator)


class DocumentSampler(EmbeddingManager):
    def __init__(self, filename) -> None:
        super(DocumentSampler, self).__init__(filename)

    def set_doc_generator(self, all_docs, docs, future_docs, type):
        self.future_docs = future_docs.copy()
        self.current_id = 0
        if type == 'random':
            self.random_order = np.random.permutation(len(future_docs))
        if type == 'less':
            docG = DSIM.assign_groups(docs)
            futureG = DSIM.assign_groups(future_docs)
            in_iter_doc_list = np.array([d.id for d in future_docs])
            sort_by_count = np.argsort(np.array([len(g) for g in docG]))
            sorted_futrue_docs_by_group = []
            for gid in sort_by_count:
                for d in futureG[gid]:
                    if d not in sorted_futrue_docs_by_group:
                        sorted_futrue_docs_by_group.append(d)
            sorted_futrue_docs_by_group = np.array(sorted_futrue_docs_by_group)
            self.random_order = np.where(in_iter_doc_list == sorted_futrue_docs_by_group[:, None])[-1]
        if type == 'more':
            docG = DSIM.assign_groups(docs)
            futureG = DSIM.assign_groups(future_docs)
            in_iter_doc_list = np.array([d.id for d in future_docs])
            sort_by_count = np.argsort(np.array([len(g) for g in docG]))[::-1]
            sorted_futrue_docs_by_group = []
            for gid in sort_by_count:
                for d in futureG[gid]:
                    if d not in sorted_futrue_docs_by_group:
                        sorted_futrue_docs_by_group.append(d)
            sorted_futrue_docs_by_group = np.array(sorted_futrue_docs_by_group)
            self.random_order = np.where(in_iter_doc_list == sorted_futrue_docs_by_group[:, None])[-1]
        self.sample_doc_generator = self.sample()

    def reset(self):
        self.current_id = 0

    def sample(self):
        while True:
            if self.current_id >= len(self.random_order):
                self.current_id = 0
            yield self.future_docs[self.random_order[self.current_id]]
            self.current_id += 1

    def get_next(self):
        return next(self.sample_doc_generator)