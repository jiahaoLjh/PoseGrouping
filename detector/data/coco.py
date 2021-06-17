import cv2
import logging

from pycocotools.coco import COCO


image_path = {
    "train": "../data/coco/images/train2017",
    "valid": "../data/coco/images/val2017",
    "test": "../data/coco/images/test2017",
}
anno_path = {
    "train": "../data/coco/annotations/person_keypoints_train2017.json",
    "valid": "../data/coco/annotations/person_keypoints_val2017.json",
    "test": "../data/coco/annotations/image_info_test-dev2017.json",
}


class COCODataset(object):

    def __init__(self, mode):
        self.logger = logging.getLogger(self.__class__.__name__)

        assert mode in ["train", "valid", "test"], mode
        self.mode = mode

        self.image_path = image_path[mode]
        self.anno_path = anno_path[mode]

        self.coco = COCO(self.anno_path)

        if mode == "test":
            self.imgIds = self.coco.getImgIds()
        else:
            self.catIds = self.coco.getCatIds()
            self.imgIds = self.coco.getImgIds(catIds=self.catIds)

        self.logger.info("{} images in {} dataset".format(self.__len__(), mode))

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        assert 0 <= idx < self.__len__(), "{}, {}".format(idx, self.__len__())

        image_id = self.imgIds[idx]

        img_info = self.coco.loadImgs(ids=image_id)[0]
        img_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]

        img_bgr = cv2.imread("{}/{}".format(self.image_path, img_name))
        h, w, _ = img_bgr.shape
        assert height == h, "height {} != {} for image {}".format(height, h, img_name)
        assert width == w, "width {} != {} for image {}".format(width, w, img_name)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.mode == "test":
            anns = [{"image_id": image_id}]
        else:
            annIds = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ids=annIds)

        return {
            "idx": idx,
            "img": img_rgb,
            "height": height,
            "width": width,
            "img_name": img_name,
            "img_id": image_id,
            "anns": anns,
        }
