import logging


class Config(object):

    def __init__(self):
        self.logger = logging.getLogger("Config")

        self.params = set()

        self.set("inference_split", "valid")

        self.set("exp_root", "exps")

        self.set("batch_size", 4)
        self.set("num_workers", 0)
        self.set("gpu", 0)

        self.set("anno_file_train", "data/coco/annotations/person_keypoints_train2017.json")
        self.set("anno_file_valid", "data/coco/annotations/person_keypoints_val2017.json")
        self.set("anno_file_test", "data/coco/annotations/image_info_test-dev2017.json")

        self.set("cache_dir", "data/coco/cache")

        self.set("max_n_det", 250)

        self.set("gnn_n_layers_geometry", 3)
        self.set("gnn_n_layers_visual", 1)

        self.set("learning_rate", 5e-4)
        self.set("num_epochs", 500)
        self.set("num_steps_per_epoch", 2000)

        self.set("training", True)
        self.set("validation", True)
        self.set("save_prediction", True)
        self.set("save_model", True)

    def set(self, k, v):
        self.logger.debug("Setting {}({}) to {}({})".format(k, type(k), v, type(v)))
        self.__setattr__(k, v)
        self.params.add(k)

    def get(self, k):
        if k in self.params and k in self.__dict__:
            return self.__getattribute__(k)
        else:
            return None

    def print_all_params(self):
        self.logger.debug("Printing configs:")
        for k in sorted(self.params):
            self.logger.debug("\t{}: {}".format(k, self.get(k)))


cfg = Config()
