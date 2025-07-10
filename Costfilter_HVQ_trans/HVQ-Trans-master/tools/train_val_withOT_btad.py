import argparse
import logging
import os
import pprint
import shutil
import time
import pandas as pd

import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper1 import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    load_state_visa,
    load_state_visa_0,
    save_checkpoint,
    set_random_seed,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound, visualize_single
# from models.HVQ_TR_switch import HVQ_TR_switch
from models.HVQ_TR_switch_OT import HVQ_TR_switch_OT
import os
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms")


parser = argparse.ArgumentParser(description="UniAD Framework")




parser.add_argument("--config", default="./config_btad_withOT.yaml")  




parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument('--train_only_four_decoder',default=False,type=bool)

def main():
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # config.port = config.get("port", None)
    # rank, world_size = setup_distributed(port=config.port)

    rank = 0
    world_size = 1
    
    # writer = SummaryWriter(log_dir='/home/ZZ/anomaly/GLAD-main/model/anomalydino/logs_test_train_2dunet_dino_triloss_cost_anomalydino')
    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    print('config.save_path', config.save_path)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = HVQ_TR_switch_OT(channel=272, embed_dim=256)
    # C
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # model.cuda()
    # local_rank = int(os.environ["LOCAL_RANK"])
    # model = DDP(
    #     model,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=True,
    # )
    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    # parameters = [
    #     {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    # ]
    parameters = [
        {'params': filter(lambda p: p.requires_grad, model.parameters())}
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    # for name,p in model.named_parameters():
    #     if id(p) in map(id, optimizer.param_groups[0]['params']):
    #         print(name)

    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    resume_model = config.saver.get("resume_model", None)
    print('resume_model', resume_model)

    print('HVQ_TR_switch_OT')

    if resume_model:
        best_metric, last_epoch = load_state_visa_0(resume_model, model, optimizer=optimizer)

    train_loader, val_loader = build_dataloader(config.dataset, distributed=False)

    if args.evaluate:
        loaded_state_dict = torch.load(resume_model)["state_dict"]



        # 定义保存输出的文件
        output_file = "/home/ZZ/anomaly/HVQ-Trans-master/experiments/MVTec-AD/checkpoints/HVQ_TR_switch_BTAD_from0_withOT/model_comparison.txt"

        # 将输出重定向到文件
        with open(output_file, "w") as f:
            sys.stdout = f  # 重定向标准输出

            # 打印模型结构
            print("Model structure:")
            for name, param in model.named_parameters():
                print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")

            # 打印参数文件中的键
            if loaded_state_dict:
                print("\nModel parameters in file:")
                for key in loaded_state_dict.keys():
                    print(f"Param name in file: {key}")

            # 比较模型结构和参数文件
            if loaded_state_dict:
                print("\nComparison of model layers and parameter file keys:")
                model_keys = set([name for name, _ in model.named_parameters()])
                param_file_keys = set(loaded_state_dict.keys())

                missing_in_model = param_file_keys - model_keys
                missing_in_file = model_keys - param_file_keys
                matching_keys = model_keys & param_file_keys

                print(f"Matching keys ({len(matching_keys)}): {sorted(list(matching_keys))}")
                print(f"Keys missing in model ({len(missing_in_model)}): {sorted(list(missing_in_model))}")
                print(f"Keys missing in parameter file ({len(missing_in_file)}): {sorted(list(missing_in_file))}")

            # 恢复标准输出
            sys.stdout = sys.__stdout__

        print(f"Output saved to {output_file}")




        model.load_state_dict(loaded_state_dict)
        validate(val_loader, model, device)
        return

    criterion = build_criterion(config.criterion)
    a_ = 0
    for epoch in range(last_epoch, config.trainer.max_epoch):
    
        
        # train_loader.sampler.set_epoch(epoch)
        # val_loader.sampler.set_epoch(epoch)
        last_iter = epoch * len(train_loader)
        a_ = train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
            device,
            a_
        )
        lr_scheduler.step(epoch)

        if epoch > 20 and (epoch+2) % config.trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, model, device)
            # only ret_metrics on rank0 is not empty
            if rank == 0:
                ret_key_metric = ret_metrics[key_metric]
                print('Epoch :',epoch + 1,'Best Metric:',best_metric,'Current Metric:',ret_key_metric)
                is_best = ret_key_metric >= best_metric
                best_metric = max(ret_key_metric, best_metric)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": config.net,
                        "state_dict": model.state_dict(),
                        "best_metric": best_metric,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    config,
                )


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
    device,
    a_
    ):
    model.training = True
    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()

    # world_size = dist.get_world_size()
    # rank = dist.get_rank()

    world_size = 1
    rank = 0


    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
    # for i, input in enumerate(tqdm(train_loader)):
        a_ += 1
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        outputs = model(input, device)

        loss = outputs['loss']

        # reduced_loss = loss.clone()
        # dist.all_reduce(reduced_loss)
        # reduced_loss = reduced_loss / world_size
        losses.update(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()
    return a_

def validate(val_loader, model, device):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    model.training = False

    rank = 0
    world_size = 1

    # rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    # dist.barrier()

    with torch.no_grad():
        # for i, input in enumerate(val_loader):
        for i, input in enumerate(tqdm(val_loader, desc="Validation Progress")):
            # forward
            outputs = model(input, device)
            dump(config.evaluator.eval_dir, outputs)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 100 == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    # gather final results
    # dist.barrier()
    # total_num = torch.Tensor([losses.count]).cuda()
    # loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    # dist.all_reduce(total_num, async_op=True)
    # dist.all_reduce(loss_sum, async_op=True)
    # final_loss = loss_sum.item() / total_num.item()


    total_num = losses.count
    loss_sum = losses.avg * losses.count
    final_loss = loss_sum / total_num

    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num))
        fileinfos, preds, masks, pred_imgs = merge_together(config.evaluator.eval_dir)
        shutil.rmtree(config.evaluator.eval_dir)
        # evaluate, log & vis
        ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)

        if args.evaluate and config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                pred_imgs,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
        if args.evaluate and config.evaluator.get("vis_single", None):
            visualize_single(
                fileinfos,
                preds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
            )
    model.train()
    return ret_metrics


if __name__ == "__main__":
    rank = 0
    world_size = 1

    main()
