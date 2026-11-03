import numpy as np
import torch
from recbole.data.interaction import cat_interactions, Interaction
from recbole.data.dataloader.general_dataloader import TrainDataLoader, NegSampleEvalDataLoader, FullSortEvalDataLoader

from recbole_gnn.data.transform import gnn_construct_transform

class MixTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        
    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            candidate_num = self.neg_sample_args["candidate_num"]
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_candidate_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num * candidate_num
            )
            self.model.eval()
            interaction = copy.deepcopy(inter_feat).to(self.model.device)
            interaction = interaction.repeat(self.neg_sample_num * candidate_num)
            neg_item_feat = Interaction(
                {self.iid_field: neg_candidate_ids.to(self.model.device)}
            )
            interaction.update(neg_item_feat)
            scores = self.model.predict(interaction).reshape(candidate_num, -1)
            indices = torch.max(scores, dim=0)[1].detach()
            neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
            neg_item_ids = neg_candidate_ids[
                indices, [i for i in range(neg_candidate_ids.shape[1])]
            ].view(-1)
            self.model.train()
            return self.sampling_func(inter_feat, neg_item_ids)
        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_item_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            )
            # breakpoint()
            all_item_ids=[]
            for user_id in user_ids:
                candidate = list(self._sampler.used_ids[user_id])
                uid_item_ids = np.random.choice(candidate, self.neg_sample_num)
                all_item_ids.extend(uid_item_ids)
            # breakpoint()
            all_item_ids.extend(neg_item_ids.tolist())
            # breakpoint()
            return self.sampling_func(inter_feat, all_item_ids)
        else:
            return inter_feat
        
    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids):
        inter_feat = inter_feat.repeat(self.times)
        # breakpoint()
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_feat = self._dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        # breakpoint()
        return inter_feat
    
class CustomizedTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['gnn_transform'] is not None:
            self.transform = gnn_construct_transform(config)


class CustomizedNegSampleEvalDataLoader(NegSampleEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['gnn_transform'] is not None:
            self.transform = gnn_construct_transform(config)

    def collate_fn(self, index):
        index = np.array(index)
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                data_list.append(self._neg_sampling(self._dataset[index]))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return self.transform(self._dataset, cur_data), idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None


class CustomizedFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['gnn_transform'] is not None:
            self.transform = gnn_construct_transform(config)
