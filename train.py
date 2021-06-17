import os
import time
import tqdm
import h5py
import datetime
import logging
import logging.config
import coloredlogs

import torch
import torch.nn as nn

from config import cfg
from utils import mkdir, AverageMeter
from models.gnn import GNN
from datasets.coco import COCODataset


logger = logging.getLogger()
coloredlogs.install(level="DEBUG", logger=logger)


def save_model(model, optimizer, exp_path, epoch, step):
    save_path = os.path.join(exp_path, "ckpt", "ckpt_{}.pth.tar".format(epoch))

    torch.save({
        "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }, save_path)

    logger.info("Save model to {}".format(save_path))


def main():
    # === create exp folder
    exp_tag = datetime.datetime.now().strftime("%m%d_%H%M%S")
    exp_path = os.path.join(cfg.get("exp_root"), exp_tag)
    cfg.set("exp_tag", exp_tag)
    cfg.set("exp_path", exp_path)

    mkdir(exp_path)
    mkdir(os.path.join(exp_path, "ckpt"))

    # === logging to file
    file_handler = logging.FileHandler(os.path.join(exp_path, "log.txt"))
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    cfg.print_all_params()

    # === gnn model & optimizer
    gnn_model = GNN(34, 9 * 256, cfg)
    gnn_model = gnn_model.cuda(cfg.get("gpu"))

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=cfg.get("learning_rate"))

    # === dataset & dataloader
    if cfg.get("training"):
        dataset_train = COCODataset(cfg, "train")
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=cfg.get("batch_size"),
            shuffle=True,
            num_workers=cfg.get("num_workers"),
            pin_memory=True,
            collate_fn=COCODataset.collate)
    if cfg.get("validation"):
        dataset_valid = COCODataset(cfg, "valid")
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=cfg.get("batch_size"), shuffle=False, num_workers=cfg.get("num_workers"), pin_memory=True, collate_fn=COCODataset.collate)

    # === train
    tot_step = 0
    best_val = 0.0
    start_time = time.time()
    for epoch in range(cfg.get("num_epochs") if cfg.get("training") else 1):
        logger.info("Epoch {} of exp {} on gpu {}".format(epoch + 1, exp_tag, cfg.get("gpu")))

        epoch_time = time.time()

        if cfg.get("training"):
            gnn_model.train()

            dataloader = dataloader_train

            tot_step_per_epoch = len(dataloader) if cfg.get("num_steps_per_epoch") is None else cfg.get("num_steps_per_epoch")
            tbar = tqdm.tqdm(total=tot_step_per_epoch, ncols=100)

            for step, batch_data in enumerate(dataloader):
                if step >= tot_step_per_epoch:
                    break

                batch_node_xy = batch_data["node_xy"]
                batch_node_visual = batch_data["node_visual"]
                batch_edge_initial = batch_data["edge_initial"]
                batch_edge_masks = batch_data["edge_masks"]
                batch_edge_labels = batch_data["edge_labels"]

                batch_jt_lists = batch_data["jt_lists"]

                batch_node_xy = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_node_xy]
                batch_node_visual = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_node_visual]
                batch_edge_initial = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_initial]
                batch_edge_masks = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_masks]
                batch_edge_labels = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_labels]

                pred_edges = gnn_model(batch_node_xy, batch_node_visual, batch_edge_initial, batch_jt_lists)

                loss_edge = torch.tensor(0.0).cuda(cfg.get("gpu"))
                for pred in pred_edges:
                    loss = gnn_model.get_loss_edge(pred, batch_edge_labels, batch_edge_masks)
                    loss_edge += loss
                if len(pred_edges) > 0:
                    loss_edge /= len(pred_edges)
                tot_loss = loss_edge

                if tot_loss != 0.0:
                    optimizer.zero_grad()
                    tot_loss.backward()
                    optimizer.step()
                tot_step += 1

                tbar.set_description("Train Loss = {:.6f}".format(tot_loss.item()))
                tbar.update(1)

            tbar.close()

        if cfg.get("validation"):
            gnn_model.eval()

            dataloader = dataloader_valid

            loss_meter = AverageMeter()
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

                    batch_ids = batch_data["ids"]
                    batch_jt_lists = batch_data["jt_lists"]

                    batch_node_xy = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_node_xy]
                    batch_node_visual = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_node_visual]
                    batch_edge_initial = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_initial]
                    batch_edge_masks = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_masks]
                    batch_edge_labels = [torch.from_numpy(v).float().cuda(cfg.get("gpu")) for v in batch_edge_labels]

                    pred_edges = gnn_model(batch_node_xy, batch_node_visual, batch_edge_initial, batch_jt_lists)

                    loss_edge = torch.tensor(0.0).cuda(cfg.get("gpu"))
                    for pred in pred_edges:
                        loss = gnn_model.get_loss_edge(pred, batch_edge_labels, batch_edge_masks)
                        loss_edge += loss
                    if len(pred_edges) > 0:
                        loss_edge /= len(pred_edges)
                    tot_loss = loss_edge
                    loss_meter.add(tot_loss.item())

                    # === accuracy
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

                        saved_predictions.append((batch_ids[b], pred))

                    acc_meter.add(acc)
                    fp_meter.add(fp)
                    fn_meter.add(fn)
                    tot_meter.add(tot)

                    tbar.set_description("Valid Loss = {:.6f}. Accuracy = {:.4f}".format(tot_loss.item(), acc / tot if tot != 0 else 0.0))
                    tbar.update(1)

            tbar.close()

            if acc_meter.sum() / tot_meter.sum() > best_val:
                best_val = acc_meter.sum() / tot_meter.sum()

                if cfg.get("save_prediction"):
                    prediction_path = os.path.join(exp_path, "prediction.h5")
                    f = h5py.File(prediction_path, "w")
                    for idx, pred_edge in saved_predictions:
                        f.create_dataset("{}/edge_prediction".format(idx), data=pred_edge)
                    f.close()
                    logger.info("Save predictions at epoch {} with accuracy {:.4f}".format(epoch + 1, best_val))
                if cfg.get("save_model"):
                    save_model(gnn_model, optimizer, exp_path, epoch, tot_step)

            logger.info("Validation Loss = {:.6f}. Accuracy = {:.4f}".format(
                loss_meter.avg(),
                acc_meter.sum() / tot_meter.sum() if tot_meter.sum() != 0 else 0.0,
            ))

        new_time = time.time()
        logger.info("Epoch {} time {:.3f}s. Total time {:.3f}s".format(epoch + 1, new_time - epoch_time, new_time - start_time))
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
