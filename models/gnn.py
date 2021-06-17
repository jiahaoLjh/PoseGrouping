import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoNet(nn.Module):

    def __init__(self, edge_channels, node_channels, cfg):
        super(GeoNet, self).__init__()

        self.edge_channels = edge_channels
        self.node_channels = node_channels
        self.cfg = cfg

        self.layers_aggregate_edges = nn.Sequential(
            nn.Conv1d(in_channels=self.edge_channels * 17, out_channels=1024, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=self.edge_channels, kernel_size=1, bias=True),
            nn.ReLU(),
        )
        self.layers_node = nn.Sequential(
            nn.Conv1d(in_channels=self.edge_channels * 17, out_channels=self.node_channels, kernel_size=1, bias=True),
            nn.ReLU(),
        )
        self.layers_node_edge_affinity = nn.Sequential(
            nn.Conv2d(in_channels=(self.node_channels * 2 + self.edge_channels), out_channels=256, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, bias=True),
        )

    def forward(self, edge_feat, edge_geo, jt_lists):
        batch_size = len(edge_feat)
        out_edge_geo = []

        for b in range(batch_size):
            ef = edge_feat[b]  # N x N x C_edge
            eg = edge_geo[b]  # N x N
            jl = jt_lists[b]
            n_node = ef.size(0)

            # === aggregate features from incoming edges
            split_size = []
            split_joint_idx = []
            for j in range(17):
                if len(jl[j]) > 0:
                    split_size.append(len(jl[j]))
                    split_joint_idx.append(j)

            features = eg.unsqueeze(2) * ef  # N x N x C_edge
            split_features = torch.split(features, split_size, dim=1)
            split_normalized_features = []
            idx = 0
            for j in range(17):
                if len(jl[j]) == 0:
                    split_normalized_features.append(torch.zeros([n_node, self.edge_channels]).float().cuda(self.cfg.get("gpu")))
                else:
                    split_normalized_features.append(split_features[idx].mean(dim=1))  # N x C_edge
                    idx += 1
            assert idx == len(split_size)
            normalized_features = torch.cat(split_normalized_features, dim=1)  # N x (17 x C_edge)
            normalized_features = normalized_features.transpose(1, 0).unsqueeze(0)  # 1 x (17 x C_edge) x N
            aggregated_features = self.layers_aggregate_edges(normalized_features)  # 1 x C_edge x N
            aggregated_features = aggregated_features.squeeze(0).transpose(1, 0)  # N x C_edge

            # === map features of different joint types to a canonical feature space
            full_vec_features = []
            split_agf = torch.split(aggregated_features, split_size, dim=0)
            for joint_idx, item in zip(split_joint_idx, split_agf):
                split_vec_features = torch.zeros([item.shape[0], self.edge_channels * 17]).float().cuda(self.cfg.get("gpu"))
                split_vec_features[:, joint_idx * self.edge_channels: (joint_idx + 1) * self.edge_channels] = item  # N_j x C_edge
                full_vec_features.append(split_vec_features)  # N_j x (17 x C_edge)
            full_vec_features = torch.cat(full_vec_features, dim=0)  # N x (17 x C_edge)

            full_vec_features = full_vec_features.transpose(1, 0).unsqueeze(0)  # 1 x (17 x C_edge) x N
            node_features = self.layers_node(full_vec_features)  # 1 x C_node x N

            node_features = node_features.squeeze(0).transpose(1, 0)  # N x C_node

            # === compute geometric affinity
            nf_i = node_features.unsqueeze(1)  # N x 1 x C_node
            nf_i = nf_i.repeat([1, n_node, 1])  # N x N x C_node
            nf_j = node_features.unsqueeze(0)  # 1 x N x C_node
            nf_j = nf_j.repeat([n_node, 1, 1])  # N x N x C_node
            cat_features = torch.cat([ef, nf_i, nf_j], dim=-1)  # N x N x (2C_node + C_edge)

            cat_features = cat_features.transpose(2, 0).unsqueeze(0)  # 1 x (2C_node + C_edge) x N x N
            sim = self.layers_node_edge_affinity(cat_features)  # 1 x 1 x N x N
            sim = sim.squeeze(0).transpose(2, 0).squeeze(2)  # N x N

            out_edge_geo.append(sim)

        return out_edge_geo


class VisualNet(nn.Module):

    def __init__(self, node_channels, cfg, output_node=True):
        super(VisualNet, self).__init__()

        self.node_channels = node_channels
        self.cfg = cfg
        self.output_node = output_node

        self.layers_node_affinity = nn.Sequential(
            nn.Conv2d(in_channels=self.node_channels, out_channels=256, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, bias=True),
        )
        if self.output_node:
            self.layers_node = nn.Sequential(
                nn.Conv1d(in_channels=self.node_channels * 2, out_channels=self.node_channels, kernel_size=1, bias=True),
                nn.ReLU(),
            )

    def forward(self, node_feat):
        batch_size = len(node_feat)
        out_edge_visual = []
        out_node_feat = []

        for b in range(batch_size):
            nf = node_feat[b]  # N x C_visual
            n_node = nf.size(0)

            # === compute visual affinity
            x_i = nf.unsqueeze(1)  # N x 1 x C_visual
            x_j = nf.unsqueeze(0)  # 1 x N x C_visual
            x_ij = torch.abs(x_i - x_j)  # N x N x C_visual

            x_ij = x_ij.transpose(2, 0).unsqueeze(0)  # 1 x C_visual x N x N
            e_visual = self.layers_node_affinity(x_ij)  # 1 x 1 x N x N
            e_visual = e_visual.squeeze(0).transpose(2, 0).squeeze(2)  # N x N

            out_edge_visual.append(e_visual)

            if self.output_node:
                # === update nodes
                e_visual = F.sigmoid(e_visual)

                diag_mask = 1.0 - torch.eye(n_node)  # N x N
                diag_mask = diag_mask.cuda(self.cfg.get("gpu"))

                e_visual = e_visual * diag_mask  # N x N, zero diagonals
                ne_visual = F.normalize(e_visual, p=1, dim=1)  # N x nN

                agg_nf = torch.mm(ne_visual, nf)  # N x C_visual
                agg_nf = torch.cat([agg_nf, nf], dim=1)  # N x 2C_visual
                agg_nf = agg_nf.transpose(1, 0).unsqueeze(0)  # 1 x 2C_visual x N

                out_nf = self.layers_node(agg_nf)  # 1 x C_visual x N
                out_nf = out_nf.squeeze(0).transpose(1, 0)  # N x C_visual

                out_node_feat.append(out_nf)

        return out_node_feat, out_edge_visual


def normalize_within_joint_type(edges, jt_lists, cfg=None):
    batch_size = len(edges)
    normalized_edges = []

    for b in range(batch_size):
        eg = edges[b]  # N x N
        jl = jt_lists[b]

        split_size = []
        split_joint_idx = []
        for j in range(17):
            if len(jl[j]) > 0:
                split_size.append(len(jl[j]))
                split_joint_idx.append(j)

        split_eg = torch.split(eg, split_size, dim=-1)
        split_weights = []
        for item in split_eg:
            weights = item / (item.max(dim=1, keepdim=True)[0] + 1e-8)  # N x N_j
            split_weights.append(weights)
        weights = torch.cat(split_weights, dim=1)  # N x N

        normalized_edges.append(eg * weights)

    return normalized_edges


class GNN(nn.Module):

    def __init__(self, in_channels_xy, in_channels_visual, cfg):
        super(GNN, self).__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.in_channels_xy = in_channels_xy
        self.in_channels_visual = in_channels_visual
        self.edge_channels_xy = 256
        self.node_channels_xy = 256
        self.node_channels_visual = 256
        self.n_iter_geo = cfg.get("gnn_n_layers_geometry")
        self.n_iter_vis = cfg.get("gnn_n_layers_visual")
        self.cfg = cfg

        self.embedding_visual = nn.Conv1d(in_channels=self.in_channels_visual, out_channels=self.node_channels_visual, kernel_size=1, bias=True)
        self.embedding_visual_full = nn.Conv1d(in_channels=self.node_channels_visual * 17, out_channels=self.node_channels_visual, kernel_size=1, bias=True)
        self.embedding_edge = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels_xy, out_channels=self.edge_channels_xy, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.edge_channels_xy, out_channels=self.edge_channels_xy, kernel_size=1, bias=True),
            nn.ReLU(),
        )
        self.layers_merge_affinity = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True),
        )

        for i in range(self.n_iter_geo):
            geo_net = GeoNet(edge_channels=self.edge_channels_xy, node_channels=self.node_channels_xy, cfg=cfg)
            self.add_module("geo_{}".format(i), geo_net)
        for i in range(self.n_iter_vis):
            visual_net = VisualNet(node_channels=self.node_channels_visual, cfg=cfg, output_node=(i + 1 != self.n_iter_vis))
            self.add_module("visual_{}".format(i), visual_net)

    def forward(self, node_feat_xy, node_feat_visual, edge_geo, jt_lists):
        pred_edges = []

        batch_size = len(node_feat_visual)
        node_feat = []
        edge_feat = []
        for b in range(batch_size):
            jl = jt_lists[b]

            # === embed visual features of different joint types to the same feature space
            inp = node_feat_visual[b]  # N x C_in
            inp = inp.transpose(1, 0).unsqueeze(0)  # 1 x C_in x N
            out = self.embedding_visual(inp)  # 1 x C_visual x N
            out = out.squeeze(0).transpose(1, 0)  # N x C_visual

            split_size = []
            split_joint_idx = []
            for j in range(17):
                if len(jl[j]) > 0:
                    split_size.append(len(jl[j]))
                    split_joint_idx.append(j)

            full_out = []
            split_out = torch.split(out, split_size, dim=0)
            for joint_idx, item in zip(split_joint_idx, split_out):
                split_out_features = torch.zeros([item.shape[0], self.node_channels_visual * 17]).float().cuda(self.cfg.get("gpu"))  # Nj x (17 x C_visual)
                split_out_features[:, joint_idx * self.node_channels_visual: (joint_idx + 1) * self.node_channels_visual] = item
                full_out.append(split_out_features)
            full_out = torch.cat(full_out, dim=0)  # N x (17 x C_visual)
            full_out = full_out.transpose(1, 0).unsqueeze(0)  # 1 x (17 x C_visual) x N
            full_out = self.embedding_visual_full(full_out)  # 1 x C_visual x N
            out = full_out.squeeze(0).transpose(1, 0)  # N x C_visual

            node_feat.append(out)

            # === embed edges
            inp = node_feat_xy[b]  # N x 34
            x_i = inp.unsqueeze(1)  # N x 1 x 34
            x_j = inp.unsqueeze(0)  # 1 x N x 34
            x_ij = x_i - x_j  # N x N x 34
            x_ij = x_ij.transpose(2, 0).unsqueeze(0)  # 1 x 34 x N x N
            out = self.embedding_edge(x_ij)  # 1 x C_edge x N x N
            out = out.squeeze(0).transpose(2, 0)  # N x N x C_edge

            edge_feat.append(out)

        for i in range(self.n_iter_geo):
            edge_geo_no_sigmoid = self._modules["geo_{}".format(i)](edge_feat, edge_geo, jt_lists)
            edge_geo = [F.sigmoid(item) for item in edge_geo_no_sigmoid]
            pred_edges.append(edge_geo)

        for i in range(self.n_iter_vis):
            node_feat, edge_visual_no_sigmoid = self._modules["visual_{}".format(i)](node_feat)
            edge_visual = [F.sigmoid(item) for item in edge_visual_no_sigmoid]
            pred_edges.append(edge_visual)

        edge_geo_visual = []
        for b in range(batch_size):
            eg = edge_geo_no_sigmoid[b]  # N x N
            ev = edge_visual_no_sigmoid[b]  # N x N
            cat = torch.cat([eg.unsqueeze(0).unsqueeze(1), ev.unsqueeze(0).unsqueeze(1)], dim=1)  # 1 x 2 x N x N

            sim = self.layers_merge_affinity(cat)  # 1 x 1 x N x N
            sim = sim.squeeze(1).squeeze(0)  # N x N
            sim = F.sigmoid(sim)

            edge_geo_visual.append(sim)

        edge_geo_visual = normalize_within_joint_type(edge_geo_visual, jt_lists, cfg=self.cfg)
        pred_edges.append(edge_geo_visual)

        return pred_edges

    @staticmethod
    def get_loss_edge(edge_pred, edge_gt, edge_mask):
        batch_size = len(edge_pred)
        loss_func = nn.BCELoss(reduction="none")
        loss_edge = 0.0
        gamma = 2.0

        for b in range(batch_size):
            eg_pred = edge_pred[b].unsqueeze(0)  # 1 x N x N
            eg_gt = edge_gt[b].unsqueeze(0)
            eg_mask = edge_mask[b].unsqueeze(0)

            if torch.sum(eg_mask) > 0:
                # === focal loss
                pt = eg_pred * eg_gt + (1 - eg_pred) * (1 - eg_gt)
                w = (1 - pt).pow(gamma)

                loss_edge += torch.sum(loss_func(eg_pred, eg_gt) * w * eg_mask) / torch.sum(eg_mask)

        if batch_size > 0:
            loss_edge /= batch_size

        return loss_edge

    @staticmethod
    def np_sample_accuracy(edge_pred, edge_gt, edge_mask):
        edge_mask = (edge_mask > 0).astype(np.float32)

        accurate_edges = 0
        false_positive_edges = 0
        false_negative_edges = 0
        tot_edges = 0

        if np.sum(edge_mask) == 0:
            return 0, 0, 0, 0

        accurate_edges = np.sum((np.abs(edge_pred - edge_gt) < 0.5) * edge_mask)
        false_positive_edges = np.sum(np.logical_and(edge_pred > 0.5, edge_gt == 0) * edge_mask)
        false_negative_edges = np.sum(np.logical_and(edge_pred < 0.5, edge_gt == 1) * edge_mask)
        tot_edges = np.sum(edge_mask)

        return accurate_edges, false_positive_edges, false_negative_edges, tot_edges
