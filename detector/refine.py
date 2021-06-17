import os
import sys
import cv2
import json
import tqdm
import logging
from collections import defaultdict

import numpy as np
import torch

from task import pose as task
from data.coco import COCODataset
from utils.misc import get_transform, kpt_affine, resize


logger = logging.getLogger()
flipRef = [i - 1 for i in [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]]


def refine(det, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    """
    if len(tag.shape) == 3:
        tag = tag[:, :, :, None]

    tags = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] == 1:
            y, x = keypoints[i][:2].astype(np.int32)
            x = np.clip(x, 0, tag.shape[1] - 1)
            y = np.clip(y, 0, tag.shape[2] - 1)
            tags.append(tag[i, x, y])

    prev_tag = np.mean(tags, axis=0)
    ans = []

    for i in range(keypoints.shape[0]):
        tmp = det[i, :, :]
        tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
        tmp2 = tmp - np.round(tt)

        x, y = np.unravel_index(np.argmax(tmp2), tmp.shape)
        xx = x
        yy = y
        val = tmp[x, y]
        x += 0.5
        y += 0.5

        if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
            y += 0.25
        else:
            y -= 0.25

        if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
            x += 0.25
        else:
            x -= 0.25

        x, y = np.array([y, x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(17):
            if ans[i, 2] > 0 and keypoints[i, 2] <= 0:
                keypoints[i, :2] = ans[i, :2]
                keypoints[i, 2] = 1

    return keypoints


def multiperson(img, func, persons):
    """
    1. Resize the image to different scales and pass each scale through the network
    2. Merge the outputs across scales
    3. Find the missing joints of the people with a second pass of the heatmaps
    """

    scales = [2, 1., 0.5]

    height, width = img.shape[0:2]
    center = (width / 2, height / 2)
    dets, tags = None, []
    for idx, i in enumerate(scales):
        scale = max(height, width) / 200
        inp_res = int((i * 512 + 63) // 64 * 64)
        res = (inp_res, inp_res)

        mat_ = get_transform(center, scale, res)[:2]
        inp = cv2.warpAffine(img, mat_, res) / 255

        def array2dict(tmp):
            return {
                'det': tmp[0][:, :, :17],
                'tag': tmp[0][:, -1, 17:34]
            }

        tmp1 = array2dict(func([inp]))
        tmp2 = array2dict(func([inp[:, ::-1]]))

        tmp = {}
        for ii in tmp1:
            tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]), axis=0)

        # === tag: [2, 4, 17, res, res]
        # === det: [2, 17, res, res]

        det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][flipRef]
        if det.max() > 10:
            continue
        if dets is None:
            dets = det
            mat_fw = mat_
            mat_bw = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
        else:
            # === dets of different scales are resized to the largest res
            dets = dets + resize(det, dets.shape[1:3])

        if abs(i - 1) < 0.5:
            res = dets.shape[1:3]
            # === only keep tags for scale in (0.5, 1.5)
            tags += [resize(tmp['tag'][0], res), resize(tmp['tag'][1, :, :, ::-1][flipRef], res)]

    assert dets is not None
    assert len(tags) != 0

    # === tags: [17, res, res, 2]
    # === dets: [17, res, res]
    tags = np.concatenate([i[:, :, :, None] for i in tags], axis=3)
    dets = dets / len(scales) / 2

    dets = np.minimum(dets, 1)

    refined_persons = []
    for i in range(persons.shape[0]):
        tmp_person = persons[i].copy()
        if np.sum(tmp_person[:, 2] > 0) == 17:
            refined_persons.append(tmp_person)
            continue
        tmp_person[:, :2] = kpt_affine(tmp_person[:, :2], mat_fw) / 4.0
        refined_person = refine(dets, tags, tmp_person)
        refined_person[:, :2] = kpt_affine(refined_person[:, :2] * 4, mat_bw)
        refined_persons.append(refined_person)

    refined_persons = np.stack(refined_persons, axis=0)

    return refined_persons


def genDtByPred(pred, image_id=0):
    """
    Generate the json-style data for the output
    """
    ans = []
    for i in pred:
        val = pred[i] if type(pred) == dict else i
        if val[:, 2].max() > 0:
            tmp = {'image_id': int(image_id), "category_id": 1, "keypoints": [], "score": float(val[:, 2].mean())}
            p = val[val[:, 2] > 0][:, :2].mean(axis=0)
            for j in val:
                if j[2] > 0.:
                    tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
                else:
                    tmp["keypoints"] += [float(p[0]), float(p[1]), 1]
            ans.append(tmp)
    return ans


def complete_pose(input_json_file, output_json_file, mode):

    def init():
        config = task.__config__

        func = task.make_network(config)

        resume_file = os.path.join("exp", "checkpoint.pth.tar")
        if os.path.isfile(resume_file):
            logger.info("=> loading checkpoint {}".format(resume_file))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found: {}".format(resume_file))
            exit(0)

        return func, config

    func, config = init()

    def runner(imgs):
        return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)))['preds']

    def do(img, kps, scores, image_id):
        persons = np.array(kps)
        assert persons.shape[1] == 51, persons.shape
        persons = persons.reshape([-1, 17, 3])
        refined_persons = multiperson(img, runner, persons)
        pred = genDtByPred(refined_persons, image_id=image_id)

        for i, score in zip(pred, scores):
            i['score'] = float(score)
        return pred

    dataset = COCODataset(mode)

    preds = []
    logger.info("Loading result file {}".format(input_json_file))
    with open(input_json_file, "r") as f:
        results = json.load(f)
    image_to_dets = defaultdict(list)
    for det in results:
        image_to_dets[det["image_id"]].append(det)

    for i in tqdm.tqdm(range(len(dataset)), total=len(dataset), ncols=100):
        data = dataset[i]
        img = data["img"]
        img_id = data["img_id"]

        if img_id not in image_to_dets:
            continue

        kps = [det["keypoints"] for det in image_to_dets[img_id]]
        scores = [det["score"] for det in image_to_dets[img_id]]

        pred = do(img, kps, scores, image_id=img_id)
        preds += pred

    logger.info("Save output to {}".format(output_json_file))
    with open(output_json_file, "w") as h:
        json.dump(preds, h, indent=4)


if __name__ == "__main__":
    complete_pose(
        input_json_file=sys.argv[1],
        output_json_file=sys.argv[2],
        mode="valid",
    )
