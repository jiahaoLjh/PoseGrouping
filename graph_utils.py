import numpy as np
from sklearn.cluster import KMeans


class GTNode(object):

    def __init__(self, x, y, nx, ny, visibility, joint_type, area, person_id):
        self.x = x
        self.y = y
        self.nx = nx
        self.ny = ny
        self.visibility = int(visibility)
        self.joint_type = int(joint_type)
        self.area = area
        self.person_id = int(person_id)

        self.det_id = []


class DETNode(object):

    def __init__(self, x, y, nx, ny, confidence, joint_type, feature):
        self.x = x
        self.y = y
        self.nx = nx
        self.ny = ny
        self.confidence = confidence
        self.joint_type = int(joint_type)
        self.feature = feature

        self.gt_id = -1
        self.gt_id_score = 0.0


class GraphNode(DETNode):
    IN_CHANNEL = 3

    def __init__(self, x, y, nx, ny, confidence, joint_type, feature, is_inlier=True):
        super(GraphNode, self).__init__(x, y, nx, ny, confidence, joint_type, feature)

        self.is_inlier = is_inlier

    def get_feature(self):
        fs = [self.nx, self.ny, self.confidence]
        fs = np.array(fs)
        assert len(fs) == GraphNode.IN_CHANNEL

        return fs

    def get_info(self):
        info = [self.x, self.y, self.confidence, self.joint_type, 1.0 if self.is_inlier else 0.0]
        info = np.array(info)

        return info


def spectral_clustering(node_info, edges, jt_list, image_id=-1):
    sym_edges = (edges + edges.transpose([1, 0])) * 0.5
    sym_edges[range(node_info.shape[0]), range(node_info.shape[0])] = 0.0

    sym_edges_with_node_confidence = sym_edges * (node_info[:, 2].reshape([-1, 1]) * node_info[:, 2].reshape([1, -1])) ** 0.5

    AA = (sym_edges > 0.5).astype(np.float32)
    remaining_nodes = np.array(range(node_info.shape[0]), dtype=np.int32)

    persons = []
    scores = []
    pools = set()
    masks = set()

    iterations = 0
    while len(remaining_nodes) > 0:
        if iterations > 5:
            print("More than 5 iterations (image id: {}), remaining nodes: {}".format(image_id, len(remaining_nodes)))
            break
        A = AA[remaining_nodes][:, remaining_nodes]
        D = np.diag(A.sum(axis=1))
        L = D - A
        vals, vecs = np.linalg.eigh(L)

        vecs = vecs[:, np.argsort(vals)]
        vals = vals[np.argsort(vals)]

        for n_c in range(len(remaining_nodes)):
            # === search for the approximate number of clusters
            #     s.t. either vals[i] < 0.5 and vals[i+1] > 0.5 or vals[i] > 0.5 and vals[i+1] > 2 * vals[i]
            if n_c < len(remaining_nodes) - 1:
                if vals[n_c + 1] < 0.5:
                    continue

                if vals[n_c] > 0.5 and vals[n_c + 1] < 2 * vals[n_c]:
                    break

            kmeans = KMeans(n_clusters=n_c + 1)
            kmeans.fit(vecs[:, :n_c + 1])
            labels = kmeans.labels_
            assert labels.min() >= 0 and labels.max() < n_c + 1, "min = {}, max = {}".format(labels.min(), labels.max())

            for c_id in range(n_c + 1):
                indices = np.where(labels == c_id)[0]

                # === single node cluster
                if len(indices) == 1:
                    is_new = True
                    for p in pools:
                        if indices[0] in p:
                            is_new = False
                            break
                    if is_new:
                        pools.add(tuple(indices))
                        person = np.zeros([17, node_info.shape[1] + 1]) - 1
                        person[int(node_info[remaining_nodes[indices[0]], 3]), :-1] = node_info[remaining_nodes[indices[0]]]
                        person[int(node_info[remaining_nodes[indices[0]], 3]), -1] = remaining_nodes[indices[0]]
                        persons.append(person)
                        scores.append(node_info[remaining_nodes[indices[0]], 2])
                    masks |= set(remaining_nodes[indices])

                elif len(indices) > 1:
                    all_nodes = []
                    for i in range(len(indices)):
                        all_nodes.append(node_info[remaining_nodes[indices[i]]])
                    all_nodes = np.array(all_nodes)

                    score_confidence = all_nodes[:, 2].mean()

                    score_intra = sym_edges[remaining_nodes[indices]][:, remaining_nodes[indices]].sum()
                    cnt_intra = len(indices) ** 2

                    avg_score_intra = score_intra / cnt_intra if cnt_intra > 0 else 0.0

                    # === prune to single person skeleton
                    person = np.zeros([17, node_info.shape[1] + 1]) - 1

                    # === no duplicate joint types
                    if len(indices) <= 17 and len(node_info[remaining_nodes[indices], 3]) == len(np.unique(node_info[remaining_nodes[indices], 3])):
                        for i in range(len(indices)):
                            person[int(node_info[remaining_nodes[indices[i]], 3]), :-1] = node_info[remaining_nodes[indices[i]]]
                            person[int(node_info[remaining_nodes[indices[i]], 3]), -1] = remaining_nodes[indices[i]]
                        selected_indices = remaining_nodes[indices]
                        selected_score_confidence = score_confidence
                        selected_avg_score_intra = avg_score_intra

                    else:
                        affinity_score = np.zeros([len(indices)])
                        for i in range(len(indices)):
                            affinity_score[i] = sym_edges_with_node_confidence[remaining_nodes[indices[i]], remaining_nodes[indices]].sum()

                        # === get initial pose
                        checked_joint_types = set()
                        for i in np.argsort(affinity_score)[::-1]:
                            if person[int(node_info[remaining_nodes[indices[i]], 3]), -1] == -1:
                                person[int(node_info[remaining_nodes[indices[i]], 3]), :-1] = node_info[remaining_nodes[indices[i]]]
                                person[int(node_info[remaining_nodes[indices[i]], 3]), -1] = remaining_nodes[indices[i]]
                            checked_joint_types.add(int(node_info[remaining_nodes[indices[i]], 3]))
                            if len(checked_joint_types) == 17:
                                break
                        selected_indices = [int(person[j, -1]) for j in range(17) if int(person[j, -1]) >= 0]

                        to_update = True
                        counter = 0
                        # === iteratively update each node based on the affinity score to the selected set of nodes
                        #     stop if there is no more change, or there are more than 10 times update
                        while to_update:
                            if counter > 10:
                                break

                            to_update = False
                            affinity_score = np.zeros([len(indices)])
                            for i in range(len(indices)):
                                joint_type = int(node_info[remaining_nodes[indices[i]], 3])
                                selected_indices_of_other_joint_types = [selected_indices[k] for k in range(len(selected_indices)) if int(node_info[selected_indices[k], 3]) != joint_type]
                                affinity_score[i] = sym_edges_with_node_confidence[remaining_nodes[indices[i]], selected_indices_of_other_joint_types].sum()

                            checked_joint_types = set()
                            for i in np.argsort(affinity_score)[::-1]:
                                if int(node_info[remaining_nodes[indices[i]], 3]) in checked_joint_types:
                                    continue
                                if person[int(node_info[remaining_nodes[indices[i]], 3]), -1] != remaining_nodes[indices[i]]:
                                    person[int(node_info[remaining_nodes[indices[i]], 3]), :-1] = node_info[remaining_nodes[indices[i]]]
                                    person[int(node_info[remaining_nodes[indices[i]], 3]), -1] = remaining_nodes[indices[i]]
                                    to_update = True
                                checked_joint_types.add(int(node_info[remaining_nodes[indices[i]], 3]))
                                if len(checked_joint_types) == 17:
                                    break

                            selected_indices = [int(person[j, -1]) for j in range(17) if int(person[j, -1]) >= 0]
                            counter += 1

                        selected_score_confidence = person[person[:, 2] > -1, 2].mean()
                        selected_score_intra = sym_edges[selected_indices][:, selected_indices].sum()
                        selected_cnt_intra = len(selected_indices) ** 2

                        selected_avg_score_intra = selected_score_intra / selected_cnt_intra if selected_cnt_intra > 0 else 0.0

                    is_new = True
                    for p in pools:
                        if set(selected_indices).issubset(p):
                            is_new = False
                            break
                    if is_new:
                        pools.add(tuple(sorted(selected_indices)))
                        persons.append(person)
                        scores.append(selected_score_confidence + selected_avg_score_intra)
                        masks |= set(selected_indices)

        # === continue with remaining nodes
        remaining_nodes = sorted(list(set(remaining_nodes) - masks))
        remaining_nodes = np.array(remaining_nodes, dtype=np.int32)
        iterations += 1

    return persons, scores


# ==================================================
#  Below are codes for generating graph data
# ==================================================

def graph_generation(gt_list, det_list, gt_person_to_indices):
    """
    :params gt_list: list of GTNode
    :params det_list: list of DETNode
    :params gt_person_to_indices: list (person) of list (joint indices)

    :return nodes: list of GraphNode
    :return edges: matrix of size NxN, with values in [0, 1]
    :return edge_mask: matrix of size NxN, with values in [0, 1]
    """

    jointType_to_indices = [[] for _ in range(17)]
    for i, det_i in enumerate(det_list):
        jointType_to_indices[det_i.joint_type].append(i)

    nodes = []
    edges = np.zeros([len(det_list), len(det_list)])
    edge_mask = np.zeros([len(det_list), len(det_list)])

    # === labels:
    #     0: dets assigned to different gt persons (no matter whether it's top detection)
    #     1: dets assigned to same gt person (top detection for each joint)
    for i, det_i in enumerate(det_list):
        if det_i.gt_id >= 0:
            # === the first matching is considered inliner
            if i == gt_list[det_i.gt_id].det_id[0]:
                is_inlier = True
                edges[i, i] = 1  # indicating inliner at diagonals
            else:
                is_inlier = False
            for j in range(i + 1, len(det_list)):
                det_j = det_list[j]
                if det_j.gt_id >= 0:
                    if gt_list[det_i.gt_id].person_id != gt_list[det_j.gt_id].person_id:
                        edge_mask[i, j] = 1
                        edge_mask[j, i] = 1
                        edges[i, j] = 0
                        edges[j, i] = 0
                    else:
                        # === both i and j are inliers
                        if i == gt_list[det_i.gt_id].det_id[0] and j == gt_list[det_j.gt_id].det_id[0]:
                            edge_mask[i, j] = 1
                            edge_mask[j, i] = 1
                            edges[i, j] = 1
                            edges[j, i] = 1
        else:
            is_inlier = False

        graph_node = GraphNode(det_i.x, det_i.y, det_i.nx, det_i.ny, det_i.confidence, det_i.joint_type, det_i.feature, is_inlier)
        nodes.append(graph_node)

    # === rule 4
    for person_indices in gt_person_to_indices:
        indices_1 = set()
        indices_2 = set()
        for index in person_indices:
            indices_1 |= set(gt_list[index].det_id)
            indices_2 |= set(jointType_to_indices[gt_list[index].joint_type]) - indices_1
        dim_x, dim_y = np.meshgrid(list(indices_1), list(indices_2), indexing="ij")
        if len(indices_1) > 0 and len(indices_2) > 0:
            edges[dim_x, dim_y] = 0
            edges[dim_y, dim_x] = 0

            for index in indices_1:
                edge_mask[index, list(indices_2)] = np.maximum(edge_mask[index, list(indices_2)], det_list[index].gt_id_score)
                edge_mask[list(indices_2), index] = np.maximum(edge_mask[list(indices_2), index], det_list[index].gt_id_score)

    return nodes, edges, edge_mask


def gt_det_association(gt_list, det_list):
    """
    pair detections with ground truth nodes

    :params gt_list: list of GTNode
    :params det_list: list of DETNode
    """

    if len(gt_list) == 0 or len(det_list) == 0:
        return

    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    thresh = 0.5

    pairs = []

    for gt_id, gt in enumerate(gt_list):
        for det_id, det in enumerate(det_list):
            if det.joint_type == gt.joint_type:
                sqr_dist = (gt.x - det.x) ** 2 + (gt.y - det.y) ** 2
                e = np.exp(- sqr_dist / vars[det.joint_type] / (gt.area + np.spacing(1)) / 2)
                pairs.append([gt_id, det_id, sqr_dist, e])

    pairs = np.array(pairs)
    if len(pairs) == 0:
        return

    for idx in np.argsort(pairs[:, -1] * -1):
        gt_id, det_id, sd, e = pairs[idx]
        gt_id = int(gt_id)
        det_id = int(det_id)
        if e < thresh:
            break
        if det_list[det_id].gt_id >= 0:
            continue

        gt = gt_list[gt_id]
        det = det_list[det_id]

        gt.det_id.append(det_id)
        det.gt_id = gt_id
        det.gt_id_score = (e - thresh) / (1 - thresh)
