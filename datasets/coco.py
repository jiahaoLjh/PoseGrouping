import os
import h5py
import logging

import numpy as np
import torch
import torch.utils.data

from pycocotools.coco import COCO

from graph_utils import spectral_clustering


class COCODataset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):
        super(COCODataset, self).__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        assert mode in ["train", "valid", "test"], mode

        self.cfg = cfg
        self.mode = mode

        self.anno_path = cfg.get("anno_file_{}".format(mode))

        self.coco = COCO(self.anno_path)

        if mode != "test":
            self.catIds = self.coco.getCatIds()
            self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        else:
            self.imgIds = self.coco.getImgIds()

        self.logger.info("{} images in {} split".format(self.__len__(), mode))

        cache_path = os.path.join(cfg.get("cache_dir"), mode)
        graph_file = os.path.join(cache_path, "graph_data.h5")
        feat_file = os.path.join(cache_path, "features.h5")

        assert os.path.isfile(graph_file), "No graph data found at {}, generate graph data first!".format(graph_file)
        assert os.path.isfile(feat_file), "No feature found at {}, generate feature first!".format(feat_file)
        self.logger.info("Load graph data at {}".format(graph_file))
        self.graph_data = h5py.File(graph_file, "r")
        self.logger.info("Load feature at {}".format(feat_file))
        self.features = h5py.File(feat_file, "r")

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        assert 0 <= idx < self.__len__(), "{}, {}".format(idx, self.__len__())

        if str(idx) not in self.graph_data:
            return dict()

        graph_data = self.graph_data[str(idx)]
        all_nodes = graph_data["nodes"][:]  # normalized x, normalized y, confidence
        all_node_info = graph_data["node_info"][:]  # x, y, confidence, joint type, is_inlier
        all_edge_masks = graph_data["edge_masks"][:]
        all_edge_labels = graph_data["edge_labels"][:]

        feature_data = self.features[str(idx)]
        all_node_features = feature_data["node_features"][:]
        assert all_node_features.shape[1] == 9 * 256, all_node_features.shape

        assert all_nodes.shape[0] == all_node_info.shape[0] == all_edge_masks.shape[0] == all_edge_labels.shape[0] == all_node_features.shape[0]

        # === sort by confidence and keep the top N detections
        inds = np.argsort(all_node_info[:, 2])[::-1]
        inds = inds[:self.cfg.get("max_n_det")]
        inds = sorted(inds)
        nodes = all_nodes[inds]
        node_info = all_node_info[inds]
        edge_masks = all_edge_masks[inds][:, inds]
        edge_labels = all_edge_labels[inds][:, inds]
        node_features = all_node_features[inds]

        # === reorder nodes in joint type order
        inds = np.argsort(node_info[:, 3])
        nodes = nodes[inds]
        node_info = node_info[inds]
        edge_masks = edge_masks[inds][:, inds]
        edge_labels = edge_labels[inds][:, inds]
        node_features = node_features[inds]

        # === encode joint type info
        node_xy = np.zeros([nodes.shape[0], 17, 2])
        for i in range(nodes.shape[0]):
            node_xy[i, int(node_info[i, 3]), :] = nodes[i, :2]
        node_xy = node_xy.reshape([nodes.shape[0], 34])

        # === mapping from joint type to nodes
        jt_list = [[] for _ in range(17)]
        for i in range(node_info.shape[0]):
            jt_list[int(node_info[i, 3])].append(i)

        assert node_info.shape[0] > 0, "{}: node_info shape {}".format(idx, node_info.shape)

        return {
            "node_xy": node_xy,
            "node_visual": node_features,
            "edge_initial": np.ones_like(edge_labels),
            "edge_masks": edge_masks,
            "edge_labels": edge_labels,
            "node_info": node_info,
            "ids": idx,
            "image_ids": self.imgIds[idx],
            "jt_lists": jt_list,
        }

    def cluster(self, predictions):
        results_json = []

        for idx, pred_edge, node_info, jt_list in predictions:
            persons, scores = spectral_clustering(node_info,
                                                  pred_edge,
                                                  jt_list,
                                                  self.imgIds[idx])
            assert len(persons) == len(scores)

            for person, score in zip(persons, scores):
                anno = {"image_id": self.imgIds[idx], "category_id": 1, "keypoints": [], "score": score}
                for j in range(17):
                    if person[j, 3] > -1:
                        anno["keypoints"] += [float(person[j, 0]), float(person[j, 1]), 1]
                    else:
                        anno["keypoints"] += [0, 0, 0]
                results_json.append(anno)

        return results_json

    @staticmethod
    def collate(batch):
        ret = dict()

        for k in [
            "node_xy",
            "node_visual",
            "edge_initial",
            "edge_masks",
            "edge_labels",
            "node_info",
            "ids",
            "image_ids",
            "jt_lists",
        ]:
            ret[k] = [b[k] for b in batch if k in b]

        return ret
