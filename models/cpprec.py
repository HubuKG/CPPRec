# coding: utf-8

import os
from collections import deque, defaultdict
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree
from utils.utils import build_non_zero_graph, build_knn_normalized_graph

from common.abstract_recommender import GeneralRecommender



class CueSlotInteraction(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, dropout: float):
        super().__init__()
        self.u = nn.Linear(in_dim, rank, bias=False)
        self.v = nn.Linear(in_dim, rank, bias=False)
        self.proj = nn.Identity() if rank == out_dim else nn.Linear(rank, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.u(x)
        b = self.v(x)
        z = a * b
        z = self.dropout(z)
        z = self.proj(z)
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-9)
        z = F.normalize(z, p=2, dim=-1)
        return z


class CueGate(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.mlp(h))
        return g * h


class CPPRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CPPRec, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.alpha_text_p = config['alpha_text_p']
        self.alpha_interest = config['alpha_interest']
        self.alpha_text_z = config['alpha_text_z']
        self.alpha_image = config['alpha_image']
        has_id = True
        self.config = config
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.cold_start = 0
        self.dataset = dataset
        # self.construction = 'weighted_max'
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.embedding_size = self.dim_latent * 3
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.num_of_session_k = config['num_of_session_k']
        self.gamma = config['gamma']
        self.theta = 1.0
        self.tau = config['tau']
        self.degree_n = config['degree_n']
        self.use_fai = config['use_fai']
        self.csi_rank = config['csi_rank']
        self.csi_dropout = config['csi_dropout']
        self.use_psh = config['use_psh']
        self.psh_hidden = config['psh_hidden']
        self.psh_tau = config['psh_tau']
        self.g_weight = config['g_weight']

        self.use_user_aug = config['use_user_aug']
        self.user_aug_item_source = config['user_aug_item_source']
        self.user_aug_item_k = config['user_aug_item_k']
        self.user_aug_user_k = config['user_aug_user_k']
        self.user_aug_remove_seen = config['user_aug_remove_seen']
        self.user_aug_apply_to_main = config['user_aug_apply_to_main']

        self.use_user_view_cl = config['use_user_view_cl']
        self.user_view_tau = config['user_view_tau']
        self.user_view_weight = config['user_view_weight']
        self.user_view_cache_every = config['user_view_cache_every']
        self.user_view_cache_on_cpu = config['user_view_cache_on_cpu']

        self._epoch_counter = 0
        self._user_rep_aug_cache = None  # [n_users, embedding_size]
        self.mm_adj = None
        # load data
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.data_path = dataset_path
        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),
                                       allow_pickle=True).item()


        self.co_adj = self.get_co_occurrence_item()

        # load inters
        self.train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(self.train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            image_indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            # store per-item neighbor indices (for USER-style edge expansion)
            self.image_knn = image_indices[1].view(self.num_item, self.knn_k).detach().cpu().numpy()
            self.image_adj = image_adj
            del image_adj, image_indices
        if self.p_feat is not None:
            self.p_embedding = nn.Embedding.from_pretrained(self.p_feat, freeze=False)
            p_indices, p_adj = self.get_knn_adj_mat(self.p_embedding.weight.detach())
            # store per-item neighbor indices (for USER-style edge expansion)
            self.p_knn = p_indices[1].view(self.num_item, self.knn_k).detach().cpu().numpy()
            self.p_adj = p_adj
            del p_adj, p_indices
        if self.z_feat is not None:
            self.z_embedding = nn.Embedding.from_pretrained(self.z_feat, freeze=False)
            z_indices, z_adj = self.get_knn_adj_mat(self.z_embedding.weight.detach())
            # store per-item neighbor indices (for USER-style edge expansion)
            self.z_knn = z_indices[1].view(self.num_item, self.knn_k).detach().cpu().numpy()
            self.z_adj = z_adj
            del z_adj, z_indices



        self.edge_index = torch.cat((self.edge_index, self.get_CF_edge()), dim=1)  # CF edge


        # keep a copy of the base graph (UI interaction edges + CF item-item edges)
        self.edge_index_base = self.edge_index

        # USER-style: build augmented user-item edges by folding 2-hop paths u->b->c into 1-hop u->c
        if self.use_user_aug:
            self.edge_index_aug = self._build_user_aug_edge_index()
        else:
            self.edge_index_aug = None
        if self.v_feat is not None and self.p_feat is not None:
            self.mm_adj = self.alpha_text_p * self.p_adj + self.alpha_image * self.image_adj + \
                          self.alpha_interest * self.co_adj + self.alpha_text_z * self.z_adj

        # weight
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 3, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        if self.use_psh:
            self.psh_gate_v = CueGate(self.dim_latent, self.psh_hidden)
            self.psh_gate_p = CueGate(self.dim_latent, self.psh_hidden)
            self.psh_gate_z = CueGate(self.dim_latent, self.psh_hidden)
        else:
            self.psh_gate_v = None
            self.psh_gate_p = None
            self.psh_gate_z = None

        self._user_v_raw = None
        self._user_p_raw = None
        self._user_z_raw = None
        self._user_v_hat = None
        self._user_p_hat = None
        self._user_z_hat = None

        if self.v_feat is not None:
            self.v_gcn = GCN(config, self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.v_feat)
        if self.p_feat is not None:
            self.p_gcn = GCN(config, self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.p_feat)

        if self.z_feat is not None:
            self.z_gcn = GCN(config, self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.z_feat)

    def build_session_tree(self, sim_matrix):
        session_enhanced_z = torch.zeros_like(sim_matrix).to('cpu')
        for i in range(self.num_item):
            n_order_relationships = self.find_weighted_n_order_relationships(self.weighted_binary_relations, idx=i,
                                                                             n=self.degree_n)
            _, cols3 = torch.topk(sim_matrix[i], self.knn_k)
            session_enhanced_z[i, cols3] += sim_matrix[i, cols3] / 2
            for order, relation in n_order_relationships.items():
                # current order clos
                enhanced_clos = []
                values = []
                for tail, value in relation:
                    enhanced_clos.append(tail)
                    values.append(value ** self.tau)
                order_coefficient = self.gamma * np.exp(-1 * (order - 1))
                coefficient = torch.tensor([order_coefficient * x for x in values],
                                           dtype=torch.float32).to(sim_matrix.device)
                session_enhanced_z[i, enhanced_clos] += coefficient * sim_matrix[i, enhanced_clos]
        return session_enhanced_z

    def get_co_occurrence_item(self):
        graph_co = self.item_graph_dict
        indices = []
        result = []

        for indx, v in graph_co.items():
            length = min(len(v[0]), self.num_of_session_k)
            if length == 0:
                continue
            indices.append(np.full(length, indx))
            result.append(v[0][:length])

        indices = torch.IntTensor(np.concatenate(indices)).to(self.device)
        result = torch.IntTensor(np.concatenate(result)).to(self.device)

        indices = torch.stack(
            (torch.flatten(indices), torch.flatten(result)), 0
        ).to(torch.int64).to(self.device)

        adj_size = torch.Size([len(graph_co), len(graph_co)])
        return self.compute_normalized_laplacian(indices, adj_size)

    def find_weighted_n_order_relationships(self, graph, idx, n):
        order_dict = defaultdict(list)
        visited = set()
        visited.add(idx)
        current_level_nodes = [(idx, 0)]
        for order in range(1, n + 1):
            next_level_nodes = []
            for node, _ in current_level_nodes:
                for neighbor, weight in graph.get(node, {}).items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level_nodes.append((neighbor, weight))
            if next_level_nodes:
                if order > 1:
                    '''
                    next_level_nodes = sorted(next_level_nodes, key=lambda x: x[1], reverse=True)[
                                       :int(len(next_level_nodes) / ((order - 1) * 2))
                                       ]
                    '''
                    next_level_nodes = sorted(next_level_nodes, key=lambda x: x[1], reverse=True)[
                        :int(len(current_level_nodes) / ((order - 1) * 2))
                    ]
                    # '''

                order_dict[order] = next_level_nodes
            current_level_nodes = next_level_nodes

        return dict(order_dict)

    def build_non_zero_graph(self, adj, is_sparse=True, norm_type='sym'):
        device = adj.device
        nonzero_indices = adj.nonzero()
        i = nonzero_indices.T
        v = adj[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        edge_index, edge_weight = self.get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)

    def build_knn_normalized_graph(self, adj, topk, is_sparse, norm_type):
        device = adj.device
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        if is_sparse:
            tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
            row = [i[0] for i in tuple_list]
            col = [i[1] for i in tuple_list]
            i = torch.LongTensor([row, col]).to(device)
            v = knn_val.flatten()
            edge_index, edge_weight = self.get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
            return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
        else:
            weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
            return self.get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def build_normalized_laplacian(self, vertex_count, hyperedges):
        num_edges = len(hyperedges)
        A = np.zeros((vertex_count, num_edges))
        for j, edge in enumerate(hyperedges):
            for v in edge:
                A[v, j] = 1
        vertex_degrees = np.sum(A, axis=1)
        D_v = np.diag(vertex_degrees)
        edge_degrees = np.sum(A, axis=0)
        D_e = np.diag(edge_degrees)
        if np.any(vertex_degrees == 0):
            raise ValueError("error")
        if np.any(edge_degrees == 0):
            raise ValueError("error")
        D_v_inv_sqrt = np.diag(1 / np.sqrt(vertex_degrees))
        D_e_inv = np.linalg.inv(D_e)
        I = np.eye(vertex_count)
        L = I - D_v_inv_sqrt @ A @ D_e_inv @ A.T @ D_v_inv_sqrt
        return L

    def get_sparse_laplacian(self, edge_index, edge_weight, num_nodes, normalization='none'):
        from torch_scatter import scatter_add
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg[deg <= 0] = 1e-7
        if normalization == 'sym':
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
            edge_weight.masked_fill_(edge_weight <= 1e-7, 1e-7)
        elif normalization == 'rw':
            deg_inv = 1.0 / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            edge_weight = deg_inv[row] * edge_weight
        return edge_index, edge_weight

    def get_dense_laplacian(self, adj, normalization='none'):
        if normalization == 'sym':
            rowsum = torch.sum(adj, -1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
            L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        elif normalization == 'rw':
            rowsum = torch.sum(adj, -1)
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diagflat(d_inv)
            L_norm = torch.mm(d_mat_inv, adj)
        elif normalization == 'none':
            L_norm = adj
        return L_norm

    def build_normalized_laplacian_with_zero_handling(self, vertex_count, hyperedges, epsilon=1e-6):
        num_edges = len(hyperedges)
        A = np.zeros((vertex_count, num_edges))
        for j, edge in enumerate(hyperedges):
            for v in edge:
                A[v, j] = 1
        vertex_degrees = np.sum(A, axis=1)
        edge_degrees = np.sum(A, axis=0)
        vertex_degrees[vertex_degrees == 0] = epsilon
        edge_degrees[edge_degrees == 0] = epsilon
        D_v = np.diag(vertex_degrees)
        D_e = np.diag(edge_degrees)
        D_v_inv_sqrt = np.diag(1 / np.sqrt(vertex_degrees))
        D_e_inv = np.linalg.inv(D_e)
        I = np.eye(vertex_count)
        L = I - D_v_inv_sqrt @ A @ D_e_inv @ A.T @ D_v_inv_sqrt
        return L

    def pre_epoch_processing(self):
        """Called by RecBole trainer at the beginning of each epoch."""
        self._epoch_counter += 1
        if self.use_user_aug and self.use_user_view_cl:
            if self.user_view_cache_every <= 0:
                raise ValueError("config['user_view_cache_every'] must be positive")
            if (self._epoch_counter % self.user_view_cache_every) == 0:
                self._refresh_user_aug_cache()

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))


    def _get_item_neighbors_matrix(self) -> np.ndarray:
        src = self.user_aug_item_source
        k = self.user_aug_item_k

        if src == 'image':
            if not hasattr(self, 'image_knn'):
                raise ValueError("image_knn not found; check v_feat and knn building.")
            return self.image_knn[:, :k]
        if src == 'p':
            if not hasattr(self, 'p_knn'):
                raise ValueError("p_knn not found; check p_feat and knn building.")
            return self.p_knn[:, :k]
        if src == 'z':
            if not hasattr(self, 'z_knn'):
                raise ValueError("z_knn not found; check z_feat and knn building.")
            return self.z_knn[:, :k]
        if src == 'co':
            neigh = np.full((self.num_item, k), -1, dtype=np.int64)
            for item, (nbrs, _) in self.item_graph_dict.items():
                if len(nbrs) == 0:
                    continue
                take = min(k, len(nbrs))
                neigh[item, :take] = np.asarray(nbrs[:take], dtype=np.int64)
            return neigh
        if src == 'mm':
            mats = []
            if hasattr(self, 'image_knn'):
                mats.append(self.image_knn[:, :k])
            if hasattr(self, 'p_knn'):
                mats.append(self.p_knn[:, :k])
            if hasattr(self, 'z_knn'):
                mats.append(self.z_knn[:, :k])
            if len(mats) == 0:
                raise ValueError("No KNN neighbor matrices available for 'mm' user_aug_item_source.")
            return np.concatenate(mats, axis=1)

        raise ValueError(f"Unknown config['user_aug_item_source']={src}")

    def _build_user_aug_edge_index(self) -> torch.Tensor:
        item_neighbors = self._get_item_neighbors_matrix()  # [n_items, L]
        csr = self.train_interactions.tocsr()

        aug_u, aug_i = [], []
        K_u = self.user_aug_user_k

        for u in range(self.num_user):
            items = csr[u].indices  # interacted item ids
            if items.size == 0:
                continue
            cand = item_neighbors[items].reshape(-1)
            cand = cand[cand >= 0]  # drop padding for 'co'
            if cand.size == 0:
                continue

            counts = np.bincount(cand, minlength=self.num_item).astype(np.int32)
            if self.user_aug_remove_seen:
                counts[items] = 0

            pos_mask = counts > 0
            if not np.any(pos_mask):
                continue

            k_take = min(K_u, int(pos_mask.sum()))
            idx = np.argpartition(-counts, k_take - 1)[:k_take]
            idx = idx[counts[idx] > 0]
            if idx.size == 0:
                continue
            idx = idx[np.argsort(-counts[idx])]

            aug_u.append(np.full(idx.shape[0], u, dtype=np.int64))
            aug_i.append(idx.astype(np.int64))

        if len(aug_u) == 0:
            return self.edge_index_base

        aug_u = np.concatenate(aug_u, axis=0)
        aug_i = np.concatenate(aug_i, axis=0) + self.n_users  # item node offset
        aug_edges = torch.from_numpy(np.stack([aug_u, aug_i], axis=0)).long().to(self.device)  # [2, E]
        aug_edges_undirected = torch.cat([aug_edges, aug_edges[[1, 0]]], dim=1)
        return torch.cat([self.edge_index_base, aug_edges_undirected], dim=1)

    def _compute_user_rep(self, edge_index: torch.Tensor, store_psh_buffers: bool = False) -> torch.Tensor:
        """Compute user representations under a given edge_index (no item-side mm_adj propagation)."""
        v_rep, _ = self.v_gcn(edge_index, self.v_feat)
        p_rep, _ = self.p_gcn(edge_index, self.p_feat)
        z_rep, _ = self.z_gcn(edge_index, self.z_feat)

        user_v = v_rep[:self.num_user]
        user_p = p_rep[:self.num_user]
        user_z = z_rep[:self.num_user]

        if store_psh_buffers:
            self._user_v_raw = user_v
            self._user_p_raw = user_p
            self._user_z_raw = user_z

        if self.use_psh:
            user_v_hat = self.psh_gate_v(user_v)
            user_p_hat = self.psh_gate_p(user_p)
            user_z_hat = self.psh_gate_z(user_z)
        else:
            user_v_hat, user_p_hat, user_z_hat = user_v, user_p, user_z

        if store_psh_buffers:
            self._user_v_hat = user_v_hat
            self._user_p_hat = user_p_hat
            self._user_z_hat = user_z_hat

        user_rep = torch.stack([user_v_hat, user_p_hat, user_z_hat], dim=2)  # [U, d, 3]
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)  # [U, 3d]
        return user_rep

    def _refresh_user_aug_cache(self):
        if not self.use_user_aug:
            self._user_rep_aug_cache = None
            return
        if self.edge_index_aug is None:
            self.edge_index_aug = self._build_user_aug_edge_index()

        was_training = self.training
        self.eval()
        with torch.no_grad():
            user_rep_aug = self._compute_user_rep(self.edge_index_aug, store_psh_buffers=False)  # [U, 3d]
            if self.user_view_cache_on_cpu:
                self._user_rep_aug_cache = user_rep_aug.detach().cpu()
            else:
                # keep on current device (usually GPU)
                self._user_rep_aug_cache = user_rep_aug.detach()
        if was_training:
            self.train()

    def _user_view_infonce(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """InfoNCE for view-level user alignment (base graph vs augmented graph)."""
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        logits = torch.matmul(anchor, positive.t()) / self.user_view_tau
        labels = torch.arange(anchor.size(0), device=anchor.device)
        return F.cross_entropy(logits, labels)


    def _psh_infonce(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        logits = torch.matmul(anchor, positive.t()) / self.psh_tau  # [B, B]
        labels = torch.arange(anchor.size(0), device=anchor.device)
        return F.cross_entropy(logits, labels)

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        # choose which graph to use for the main forward path
        if self.use_user_aug and self.user_aug_apply_to_main:
            edge_index_used = self.edge_index_aug
            if edge_index_used is None:
                edge_index_used = self._build_user_aug_edge_index()
                self.edge_index_aug = edge_index_used
        else:
            edge_index_used = self.edge_index_base

        # compute modality-specific node representations
        v_rep, self.v_preference = self.v_gcn(edge_index_used, self.v_feat)
        p_rep, self.p_preference = self.p_gcn(edge_index_used, self.p_feat)
        z_rep, self.z_preference = self.z_gcn(edge_index_used, self.z_feat)
        representation = torch.cat((v_rep, p_rep, z_rep), dim=1)

        user_rep = self._compute_user_rep(edge_index_used, store_psh_buffers=True)
        self.user_rep = user_rep
        item_rep = representation[self.num_user:]

        h = item_rep
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        item_rep = item_rep + h

        all_embedding = torch.cat((user_rep, item_rep), dim=0)
        self.result_embed = all_embedding

        user_tensor = all_embedding[user_nodes]
        pos_item_tensor = all_embedding[pos_item_nodes]
        neg_item_tensor = all_embedding[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_p = (self.p_preference[user] ** 2).mean() if self.p_preference is not None else 0.0
        reg_embedding_loss_z = (self.z_preference[user] ** 2).mean() if self.z_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_p + reg_embedding_loss_z)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        p_loss = 0.0
        if self.use_psh:
            u = user
            p_loss_v = self._psh_infonce(self._user_v_hat[u], self._user_v_raw[u])
            p_loss_p = self._psh_infonce(self._user_p_hat[u], self._user_p_raw[u])
            p_loss_z = self._psh_infonce(self._user_z_hat[u], self._user_z_raw[u])
            p_loss = (p_loss_v + p_loss_p + p_loss_z) / 3.0

        user_view_cl = 0.0
        if self.use_user_aug and self.use_user_view_cl:
            if self._user_rep_aug_cache is None:
                # build cache lazily for the first epoch
                self._refresh_user_aug_cache()
            if self._user_rep_aug_cache is not None:
                anchor = self.user_rep[user]
                if self.user_view_cache_on_cpu:
                    user_cpu = user.detach().to('cpu')
                    positive = self._user_rep_aug_cache[user_cpu].to(anchor.device)
                else:
                    positive = self._user_rep_aug_cache[user]
                user_view_cl = self._user_view_infonce(anchor, positive)

        return loss_value + reg_loss + self.g_weight * p_loss + self.user_view_weight * user_view_cl

    def full_sort_predict(self, interaction):

        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

    def get_CF_edge(self):
        edge_head = []
        edge_rear = []
        for key, value in self.item_graph_dict.items():
            neighbors = value[0][:10]
            for neighbor in neighbors:
                edge_head.append(key + self.n_users)
                edge_rear.append(neighbor + self.n_users)
                edge_head.append(neighbor + self.n_users)
                edge_rear.append(key + self.n_users)
        result = np.vstack([edge_head, edge_rear]).transpose()
        edge = torch.LongTensor(result).t().contiguous().to(self.device)
        return edge

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.train_interactions.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def user_graph_constract(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre

    def build_sim(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim


class Item_Graph_sample(torch.nn.Module):
    def __init__(self, num_item, aggr_mode, dim_latent):
        super(Item_Graph_sample, self).__init__()
        self.num_item = num_item
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, item_graph, item_matrix):
        index = item_graph
        u_features = features[index]
        user_matrix = item_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCN(torch.nn.Module):
    def __init__(self, config, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device
        self.config = config
        self.fW = nn.Parameter(torch.Tensor(3))
        self.use_fai = config['use_fai']
        self.csi_rank = config['csi_rank']
        self.csi_dropout = config['csi_dropout']

        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))
        if self.use_fai:
            self.fai = CueSlotInteraction(self.dim_feat, self.dim_latent, self.csi_rank, self.csi_dropout)
            self.MLP = None
            self.MLP_1 = None
        else:
            self.fai = None
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
        self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
        self.ID_embedding = nn.Embedding(self.num_item, 64)
        nn.init.xavier_uniform_(self.ID_embedding.weight)

    def forward(self, edge_index, features, item_edge=None):
        features = features.to(self.device)
        if self.use_fai:
            temp_features = self.fai(features)
        else:
            temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        temp_features = torch.multiply(self.ID_embedding.weight, temp_features)

        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)
        x_hat = h + x + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

