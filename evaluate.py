import sys
import numpy as np
import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == "__main__":
    res_json_ref = "valid_multi.json"
    res_json_complete = sys.argv[1]
    res_json_complete_final = sys.argv[2]

    cocoGt = COCO("data/coco/annotations/person_keypoints_val2017.json")
    gt_imgIds = cocoGt.getImgIds(catIds=cocoGt.getCatIds())

    with open(res_json_ref, "r") as f:
        dets_ref = json.load(f)

    with open(res_json_complete, "r") as f:
        dets_complete = json.load(f)

    dets_complete_no_duplicate = []
    image_ids_ref = set()
    image_ids_complete = set()

    # === remove duplicate
    print("{} dets before removing duplicate".format(len(dets_complete)))
    image_to_det = defaultdict(list)
    for det in dets_complete:
        assert det["image_id"] in gt_imgIds
        image_ids_complete.add(det["image_id"])
        kps = np.array(det["keypoints"]).reshape([17, 3])
        kps = kps[:, :2]
        exist = False
        old_det = None
        for other_det in image_to_det[det["image_id"]]:
            other_kps = np.array(other_det["keypoints"]).reshape([17, 3])
            other_kps = other_kps[:, :2]
            diff = np.sum(np.abs(kps - other_kps))
            if diff < 1e-6:
                exist = True
                old_det = other_det
                break
        if not exist:
            dets_complete_no_duplicate.append(det)
            image_to_det[det["image_id"]].append(det)
        else:
            if det["score"] > old_det["score"]:
                old_det["score"] = det["score"]
    print("{} dets after removing duplicate".format(len(dets_complete_no_duplicate)))

    # === add missing images from reference
    for det in dets_ref:
        if det["image_id"] not in gt_imgIds:
            continue
        image_ids_ref.add(det["image_id"])
        if det["image_id"] not in image_ids_complete:
            dets_complete_no_duplicate.append(det)
    print("{} dets after adding dets from reference".format(len(dets_complete_no_duplicate)))

    # === save final output
    with open(res_json_complete_final, "w") as f:
        json.dump(dets_complete_no_duplicate, f)

    # === evaluation
    cocoDt = cocoGt.loadRes(res_json_ref)
    cocoDt_2 = cocoGt.loadRes(res_json_complete_final)

    print("GT Images: {}".format(len(gt_imgIds)))
    print("DET Images (reference): {}".format(len(image_ids_ref)))
    print("DET Images (final): {}".format(len(image_ids_complete)))
    eval_img_ids = gt_imgIds
    eval_img_ids = sorted(list(eval_img_ids))

    assert len(eval_img_ids) > 0
    print("Evaluate on {} images".format(len(eval_img_ids)))

    print("=" * 50)
    print("REFERENCE:")
    cocoEval = COCOeval(cocoGt, cocoDt, "keypoints")
    cocoEval.params.imgIds = eval_img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("=" * 50)
    print("RESULT:")
    cocoEval = COCOeval(cocoGt, cocoDt_2, "keypoints")
    cocoEval.params.imgIds = eval_img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
