import os
import tqdm
import h5py
import json
import argparse
import logging
import logging.config
import coloredlogs

import torch
import torch.nn as nn

from config import cfg
from utils import AverageMeter
from models.gnn import GNN
from datasets.coco import COCODataset


logger = logging.getLogger()
coloredlogs.install(level="DEBUG", logger=logger)


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="experiment to run inference on")
    parser.add_argument("--epoch", type=int, help="epoch to load")
    args, _ = parser.parse_known_args()
    return args


def load_model(model, exp_path, epoch):
    load_path = os.path.join(exp_path, "ckpt", "ckpt_{}.pth.tar".format(epoch))

    if not os.path.isfile(load_path):
        logger.info("Model does not exist: {}".format(load_path))
        return

    ckpt = torch.load(load_path)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    logger.info("Load model from {}".format(load_path))


def main():
    args = parse_command_line()

    # === load exp folder
    exp_tag = args.exp
    exp_path = os.path.join(cfg.get("exp_root"), exp_tag)
    assert os.path.isdir(exp_path), "Invalid experiment to load: {} (invalid path)".format(exp_path)
    cfg.set("exp_tag", exp_tag)
    cfg.set("exp_path", exp_path)

    cfg.print_all_params()

    # === gnn model
    gnn_model = GNN(34, 9 * 256, cfg)
    gnn_model = gnn_model.cuda(cfg.get("gpu"))

    load_model(gnn_model, exp_path, args.epoch)

    # === dataset & dataloader
    dataset = COCODataset(cfg, cfg.get("inference_split"))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.get("batch_size"), shuffle=False, num_workers=cfg.get("num_workers"), pin_memory=True, collate_fn=COCODataset.collate)

    # === inference
    logger.info("Inference of exp {} on gpu {}".format(exp_tag, cfg.get("gpu")))
    gnn_model.eval()

    acc_meter = AverageMeter()
    fp_meter = AverageMeter()
    fn_meter = AverageMeter()
    tot_meter = AverageMeter()

    tbar = tqdm.tqdm(total=len(dataloader), ncols=100)

    saved_predictions = []

    with torch.no_grad():
        for step, batch_data in enumerate(dataloader):

            batch_node_xy = batch_data["node_xy"]
            batch_node_visual = batch_data["node_visual"]
            batch_edge_initial = batch_data["edge_initial"]
            batch_edge_masks = batch_data["edge_masks"]
            batch_edge_labels = batch_data["edge_labels"]

            batch_node_info = batch_data["node_info"]
            batch_ids = batch_data["ids"]
            batch_jt_lists = batch_data["jt_lists"]

            batch_node_xy = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_node_xy]
            batch_node_visual = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_node_visual]
            batch_edge_initial = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_initial]
            batch_edge_masks = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_masks]
            batch_edge_labels = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_labels]

            pred_edges = gnn_model(batch_node_xy, batch_node_visual, batch_edge_initial, batch_jt_lists)

            acc, fp, fn, tot = 0, 0, 0, 0
            for b in range(len(batch_node_xy)):
                pred = pred_edges[-1][b].cpu().data.numpy()
                gt = batch_edge_labels[b].cpu().data.numpy()
                mask = batch_edge_masks[b].cpu().data.numpy()

                sample_acc, sample_fp, sample_fn, sample_tot = gnn_model.np_sample_accuracy(pred, gt, mask)
                acc += sample_acc
                fp += sample_fp
                fn += sample_fn
                tot += sample_tot

                saved_predictions.append((batch_ids[b], pred, batch_node_info[b], batch_jt_lists[b]))

            acc_meter.add(acc)
            fp_meter.add(fp)
            fn_meter.add(fn)
            tot_meter.add(tot)

            tbar.set_description("Valid Accuracy = {:.3f}. FP = {:.3f}. FN = {:.3f}".format(acc / tot if tot != 0 else 0.0, fp / tot if tot != 0 else 0.0, fn / tot if tot != 0 else 0.0))
            tbar.update(1)

    tbar.close()

    logger.info("Validation Accuracy = {:.4f}. FP = {:.4f}. FN = {:.4f}".format(
        acc_meter.sum() / tot_meter.sum() if tot_meter.sum() != 0 else 0.0,
        fp_meter.sum() / tot_meter.sum() if tot_meter.sum() != 0 else 0.0,
        fn_meter.sum() / tot_meter.sum() if tot_meter.sum() != 0 else 0.0,
    ))

    # === save graph output
    prediction_path = os.path.join(exp_path, "prediction_inference.h5")
    f = h5py.File(prediction_path, "w")
    for idx, pred_edge, _, _ in saved_predictions:
        f.create_dataset("{}/edge_prediction".format(idx), data=pred_edge)
    f.close()
    logger.info("Save predictions at {}".format(prediction_path))

    # === clustering
    logger.info("Clustering...")
    results_json = dataset.cluster(saved_predictions)

    logger.info("Save predictions to json file")
    with open(os.path.join(exp_path, "{}.json".format(cfg.get("inference_split"))), "w") as f:
        json.dump(results_json, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
