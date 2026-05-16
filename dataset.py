"""Dataset class for the graph classification task."""

import os

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Sampler


class SequentialSampler(Sampler):
  def __init__(self,data_source):
      self.data_source = data_source
  def __iter__(self):
      return iter(range(len(self.data_source)))
  def __len__(self):
     return len(self.data_source)


class SubsetRandomSampler(Sampler):
   def __init__(self,indices):
       self.indices = indices
   def __iter__(self):
       return (self.indices[i] for i in torch.randperm(len(self.indices)))
   def __len__(self):
       return len(self.indices)
    

class GraphDataset(InMemoryDataset):
    def __init__(self, root, ids, train_val, transform=None, pre_transform=None, pre_filter=None):
        self.ids = ids
        self.train_val = train_val
        self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.process(self.ids)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.train_val + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def adj_matrix_to_adj_list(self, adj_matrix):
        edge_indices = torch.nonzero(adj_matrix)
        adj_list = torch.zeros(2, edge_indices.shape[0], dtype=torch.long)

        for i in range(edge_indices.shape[0]):
            adj_list[0, i] = edge_indices[i, 0]
            adj_list[1, i] = edge_indices[i, 1]

        return adj_list

    def process(self, ids):
        data_list = []
        for id in ids:
            info = id.replace('\n', '')
            try:
                graph_name = info.split('\t')[0].rsplit('.', 1)[0]
                site, graph_name = graph_name.split('/')
                label = info.split('\t')[1]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid id format: {info}. Expected format is 'site/filename\tlabel'") from exc

            if site in {'LUAD', 'LSCC'}:
                site = 'LUNG'
                graph_path = os.path.join(self.root, 'CPTAC_{}_features'.format(site))
            elif site == 'NLST':
                graph_path = os.path.join(self.root, 'NLST_Lung_features')
            elif site == 'TCGA':
                graph_name = info.split('\t')[0]
                _, graph_name = graph_name.split('/')
                graph_path = os.path.join(self.root, 'TCGA_LUNG_features')
            else:
                graph_path = os.path.join(self.root, f'{site}_features')
            graph_path = os.path.join(graph_path, 'simclr_files')

            feature_path = os.path.join(graph_path, graph_name, 'features.pt')
            if os.path.exists(feature_path):
                features = torch.load(feature_path, map_location='cpu')
            else:
                raise FileNotFoundError(f'features.pt for {graph_name} doesn\'t exist')

            adj_s_path = os.path.join(graph_path, graph_name, 'adj_s.pt')
            if os.path.exists(adj_s_path):
                adj_s = torch.load(adj_s_path, map_location='cpu')
            else:
                raise FileNotFoundError(f'adj_s.pt for {graph_name} doesn\'t exist')
            edge_index = self.adj_matrix_to_adj_list(adj_s)

            data = Data(
                x = torch.FloatTensor(features),
                edge_index = torch.LongTensor(edge_index),
                y = torch.tensor(self.classdict[label], dtype=torch.long).view(-1,1)
                )

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)