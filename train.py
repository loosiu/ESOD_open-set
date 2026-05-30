# ############ EDL ############
import argparse
import logging
import math
import os
import random
import time
import warnings
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, norm_imgs
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, target2mask, check_mask
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution, plot_cluster
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel, time_synchronized
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)

@torch.no_grad()
def edl_stats_one_batch(model, dataloader, device):
    model.eval()

    batch = next(iter(dataloader))
    imgs = batch[0]

    imgs = imgs.to(device, non_blocking=True).float()
    imgs = norm_imgs(imgs, model)

    (pred_det, pred_seg), pred_masks = model(imgs, hm_only=True)

    if pred_masks is None or len(pred_masks) == 0:
        return {"ok": False, "reason": "pred_masks is None/empty"}

    # === Dual + Gating: pred_masks = [heat(1ch), edl(2ch), gate-logit(1ch)] ===
    if (len(pred_masks) == 3 and
        pred_masks[0].shape[1] == 1 and
        pred_masks[1].shape[1] == 2 and
        pred_masks[2].shape[1] == 1):
        heat = pred_masks[0].detach()
        edl  = pred_masks[1].detach()
        gate_logit = pred_masks[2].detach()
        prob = heat[:, 0].sigmoid()
        evidence = F.softplus(edl)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1)
        v_raw = 2.0 / S                           # raw vacuity
        gate = gate_logit[:, 0].sigmoid()         # learnable spatial gate ∈(0,1)

        pf = prob.flatten().float() - prob.mean()
        vf = v_raw.flatten().float() - v_raw.mean()
        corr = (pf * vf).mean() / (pf.std(unbiased=False) * vf.std(unbiased=False) + 1e-12)

        return {
            "ok": True,
            "mode": "dual_gate",
            "heatmapC": "1+2+1",
            "p_mean":     float(prob.mean()),
            "p_max":      float(prob.max()),
            "v_raw_mean": float(v_raw.mean()),
            "v_raw_max":  float(v_raw.max()),
            "gate_mean":  float(gate.mean()),
            "gate_min":   float(gate.min()),
            "gate_max":   float(gate.max()),
            "corr":       float(corr.item()),
        }

    # === Dual + max: pred_masks = [heat(1ch), edl(2ch)] ===
    if (len(pred_masks) == 2 and
        pred_masks[0].shape[1] == 1 and
        pred_masks[1].shape[1] == 2):
        heat = pred_masks[0].detach()
        edl  = pred_masks[1].detach()
        prob = heat[:, 0].sigmoid()
        evidence = F.softplus(edl)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1)
        v = 2.0 / S

        pf = prob.flatten().float() - prob.mean()
        vf = v.flatten().float() - v.mean()
        corr = (pf * vf).mean() / (pf.std(unbiased=False) * vf.std(unbiased=False) + 1e-12)

        return {
            "ok": True,
            "mode": "dual",
            "heatmapC": "1+2",
            "p_mean": float(prob.mean()),
            "p_max":  float(prob.max()),
            "v_mean": float(v.mean()),
            "v_max":  float(v.max()),
            "corr":   float(corr.item()),
        }

    # === EDL only case (legacy): pred_masks[0] is 2ch ===
    hm = pred_masks[0].detach()
    C = hm.shape[1]
    if C != 2:
        return {"ok": False, "heatmapC": int(C),
                "reason": f"unexpected pred_masks structure (len={len(pred_masks)}, C={C})"}

    evidence = F.softplus(hm)
    alpha = evidence + 1.0
    S = alpha.sum(dim=1)
    p = alpha[:, 1] / S
    v = 2.0 / S

    pf = p.flatten().float() - p.mean()
    vf = v.flatten().float() - v.mean()
    corr = (pf * vf).mean() / (pf.std(unbiased=False) * vf.std(unbiased=False) + 1e-12)

    return {
        "ok": True,
        "mode": "edl_only",
        "heatmapC": 2,
        "p_mean": float(p.mean()),
        "p_max":  float(p.max()),
        "v_mean": float(v.mean()),
        "v_max":  float(v.max()),
        "corr":   float(corr.item()),
    }
    
def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    half_precision = not opt.disable_half

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    if rank in [-1, 0] and not opt.resume:
        os.mkdir(save_dir / 'scripts')
        for file in ['train.py', 'test.py', 'utils/loss.py', 'utils/general.py', 'utils/datasets.py',
                     'models/yolo.py', 'models/common.py', opt.cfg]:
            shutil.copyfile(file, save_dir / 'scripts' / os.path.basename(file))

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    is_coco = opt.data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume and not opt.freeze else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude, cfg_path=opt.cfg)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    unfreeze = []
    if opt.freeze:
        raise NotImplementedError('Navigate to here to manually set the layers to freeze and uncomment this error.')
        hyp['weight_decay'] = 0.
        # freeze = ['model.{}.'.format(ii) for ii in range(8)]
        # print('freeze', freeze)
        unfreeze = ['model.5', 'model.6'] # DWConv and Segmenter
        print('unfreeze', unfreeze)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze) or (len(unfreeze) and all(x not in k for x in unfreeze)):
            # print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter) and (v.bias.requires_grad or True):
            pg2.append(v.bias)  # biases
        if (isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.LayerNorm)) and (v.weight.requires_grad or True):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter) and (v.weight.requires_grad or True):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        # optimizer = optim.AdamW(pg0, lr=hyp['lr0'] * 0.02)  # adjust beta1 to momentum
        optimizer = optim.AdamW(pg0, lr=hyp['lr0'] * 0.1, betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # ============================
    # SecondLook debug wrapper (학습에는 영향 X)
    # - autoshape()가 SecondLook 래퍼로 바뀌어있다는 가정
    # - EMA 모델로 래핑해서 train/eval 토글 영향 최소화
    # ============================
    sl_debug = None
    if rank in [-1, 0]:
        try:
            if ema is not None:
                sl_debug = ema.ema.autoshape()
            else:
                sl_debug = model.autoshape()
        except Exception as e:
            print("[SecondLook] debug wrapper build failed:", e)
            sl_debug = None
    debug_interval = 200  # 더 자주 보고 싶으면 20~50으로 낮춰
    # ============================

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:  # TODO: freeze
        # Optimizer
        if opt.resume and ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            if rank in [-1, 0]:
                print('loaded optimizer, best_fitness = %.2f' % best_fitness)

        # EMA
        if opt.resume and ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
            if rank in [-1, 0]:
                print('loaded ema, updates = %d' % ckpt['updates'])

        # Results
        if opt.resume and ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        if opt.resume:
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs > start_epoch > 0:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs - start_epoch))
            # epochs += start_epoch  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 64)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    # OWOD ignore mode: cls>=100 (sentinel 999) 은 max-label assertion 에서 제외
    _all_cls = np.concatenate(dataset.labels, 0)[:, 0]
    import os as _os_mlc
    if _os_mlc.environ.get('ESOD_OWOD_MODE', '').strip().lower() == 'ignore':
        _all_cls = _all_cls[_all_cls < 100]
    mlc = _all_cls.max() if len(_all_cls) else 0  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0., prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            if plots:
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        find_unused_parameters = opt.hm_only or any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    find_unused_parameters=find_unused_parameters)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = round(hyp['warmup_epochs'] * nb) if hyp['warmup_epochs'] > 0 else -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        use_gt = epoch < epochs * 0.6
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(7, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 11) %
                    ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'pixl', 'area', 'dist', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        warmup_flag = False
        for i, (imgs, targets, masks, m_weights, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            warmup_flag = ni <= nw
            imgs = imgs.to(device, non_blocking=True).float()
            imgs = norm_imgs(imgs, model)
            masks = masks.to(device, non_blocking=True).float()
            m_weights = m_weights.to(device, non_blocking=True).float()

            # OWOD ignore mode: hold-out class GT region 의 m_weights 를 0 으로 set
            #   → segmenter loss (BCE 또는 EDL Dirichlet MSE) 에서 자동 제외 (weight=0 ⇒ contribution 0)
            #   ESOD_OWOD_MODE=ignore 일 때만 활성. (3-way ambiguous zone 없이 단순 hold-out ignore 만)
            import os as _os_owod
            if _os_owod.environ.get('ESOD_OWOD_MODE', '').strip().lower() == 'ignore':
                _ho_targets = targets[targets[:, 1] >= 100]   # ignore sentinel (999)
                if len(_ho_targets) > 0:
                    _Hm, _Wm = m_weights.shape[-2:]
                    for _t in _ho_targets:
                        _bi = int(_t[0].item())
                        _cx, _cy, _w, _h = _t[2:6].tolist()
                        _x1 = max(0, int((_cx - _w / 2) * _Wm))
                        _x2 = min(_Wm, int((_cx + _w / 2) * _Wm) + 1)
                        _y1 = max(0, int((_cy - _h / 2) * _Hm))
                        _y2 = min(_Hm, int((_cy + _h / 2) * _Hm) + 1)
                        if _x2 > _x1 and _y2 > _y1:
                            m_weights[_bi, 0, _y1:_y2, _x1:_x2] = 0.0

            # ============================
            # SecondLook debug call during training (학습 중 적용 여부 바로 확인)
            # - debug_interval 마다 1장만 EMA 모델로 추론을 호출
            # - 이때 AutoShapeSecondLookMargin 내부 print("[SecondLook] total crops: ...")가 찍혀야 정상
            # ============================
            if (sl_debug is not None) and (rank in [-1, 0]) and (i % debug_interval == 0):
                with torch.no_grad():
                    im = imgs[0].detach().float().cpu()  # (C,H,W)
                    # 값 범위가 0~1이 아닐 수 있어서 안전 변환
                    im_min = float(im.min().item())
                    im_max = float(im.max().item())
                    if im_max <= 1.5 and im_min >= -0.5:
                        im = (im * 255.0).clamp(0, 255)
                    else:
                        # min-max normalize to 0~255
                        im = (im - im.min()) / (im.max() - im.min() + 1e-6)
                        im = (im * 255.0).clamp(0, 255)
                    im = im.byte().permute(1, 2, 0).numpy()  # HWC uint8

                    # EMA autoshape 호출(추론). 여기서 SecondLook 로그가 떠야 함
                    # _ = sl_debug(im, size=imgsz)
            # ============================

            # Warmup
            if warmup_flag:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                raise NotImplementedError('Interplotation will destroy the heatmap labels')
                sz = random.randrange(imgsz * 0.75, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    mask_stride = imgs.size(2) // masks.size(2)
                    assert mask_stride == 8
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    masks = F.interpolate(masks, size=[_s // mask_stride for _s in ns], mode='nearest')

            # Forward
            with amp.autocast(enabled=cuda and half_precision):
                targets = targets.to(device)
                pred = model((imgs, [masks]) if (use_gt and warmup_flag) else imgs, hm_only=opt.hm_only)  # forward
                ComputeLoss._edl_epoch = epoch  # EDL KL annealing용 epoch 전달
                loss, loss_items = compute_loss(pred, targets, imgsz=imgs.shape, masks=masks, m_weights=m_weights)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 9) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    if tb_writer and ni == 0 and not opt.sync_bn:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')  # suppress jit trace warning
                            tb_writer.add_graph(torch.jit.trace(de_parallel(model), imgs, strict=False), [])  # graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({'Mosaics': [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            try:
                edl_stats = edl_stats_one_batch(ema.ema if ema is not None else model, testloader, device)
                if edl_stats.get("ok", False):
                    if 'gate_mean' in edl_stats:
                        # Dual + Learnable Spatial Gating mode
                        print(f"[EDL][epoch {epoch}] mode={edl_stats['mode']} C={edl_stats['heatmapC']} "
                              f"p_mean={edl_stats['p_mean']:.4f} p_max={edl_stats['p_max']:.4f} "
                              f"v_raw_mean={edl_stats['v_raw_mean']:.4f} v_raw_max={edl_stats['v_raw_max']:.4f} "
                              f"gate_mean={edl_stats['gate_mean']:.4f} gate_min={edl_stats['gate_min']:.4f} gate_max={edl_stats['gate_max']:.4f} "
                              f"corr={edl_stats['corr']:.4f}")
                    else:
                        print(f"[EDL][epoch {epoch}] mode={edl_stats.get('mode','?')} C={edl_stats['heatmapC']} "
                              f"p_mean={edl_stats['p_mean']:.4f} p_max={edl_stats['p_max']:.4f} "
                              f"v_mean={edl_stats['v_mean']:.4f} v_max={edl_stats['v_max']:.4f} "
                              f"corr={edl_stats['corr']:.4f}")
                else:
                    print(f"[EDL][epoch {epoch}] NOT_OK "
                        f"heatmapC={edl_stats.get('heatmapC', None)} reason={edl_stats.get('reason', '')}")
            except Exception as e:
                print(f"[EDL][epoch {epoch}] stats failed: {e}")
        
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict,
                                                 use_gt=use_gt and warmup_flag,
                                                 conf_thres=0.1 if warmup_flag else 0.001,
                                                 batch_size=batch_size * 2 if '_tr' not in opt.cfg else 1,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 save_json=is_coco and final_epoch,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 half_precision=half_precision,
                                                 hm_metric=opt.hm_metric,
                                                 is_coco=is_coco)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 12 % results + '\n')  # append metrics, val_loss

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'train/pixl_loss', 'train/area_loss', 'train/dist_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'metrics/bestpr', 'metrics/occupy',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'val/pixl_loss', 'val/area_loss', 'val/dist_loss',  # train loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results[:4]).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})

        if not opt.evolve:
            if is_coco:  # COCO dataset
                for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                    results, _, _ = test.test(opt.data,
                                              batch_size=batch_size * 2 if '_tr' not in opt.cfg else 1,
                                              imgsz=imgsz_test,
                                              conf_thres=0.001,
                                              iou_thres=0.7,
                                              model=attempt_load(m, device).half(),
                                              single_cls=opt.single_cls,
                                              dataloader=testloader,
                                              save_dir=save_dir,
                                              save_json=True,
                                              plots=False,
                                              half_precision=half_precision,
                                              is_coco=is_coco)

            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
            if wandb_logger.wandb:  # Log the stripped model
                wandb_logger.wandb.log_artifact(str(best if best.exists() else last), type='model',
                                                name='run_' + wandb_logger.wandb_run.id + '_model',
                                                aliases=['latest', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--addition', action='store_true', help='additional training')
    parser.add_argument('--freeze', action='store_true', help='freeze shadow backbone for fine-tuning')
    parser.add_argument('--hm-only', action='store_true', help='training on heatmap prediction only')
    parser.add_argument('--hm-metric', action='store_true', help='use heatmap-related evaluation metrics')
    parser.add_argument('--disable-half', action='store_true', help='disable FP16 half-precision training')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0] and False:
        check_git_status()
        check_requirements(exclude=('pycocotools', 'thop'))

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        addition = opt.addition
        if addition:
            epochs, hyp = opt.epochs, opt.hyp
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        if addition:
            opt.epochs, opt.hyp = opt.epochs * 0 + epochs, hyp  # TODO: epochs
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0] and False:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard

        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        meta = {'lr0': (1, 1e-5, 1e-1),
                'lrf': (1, 0.01, 1.0),
                'momentum': (0.3, 0.6, 0.98),
                'weight_decay': (1, 0.0, 0.001),
                'warmup_epochs': (1, 0.0, 5.0),
                'warmup_momentum': (1, 0.0, 0.95),
                'warmup_bias_lr': (1, 0.0, 0.2),
                'box': (1, 0.02, 0.2),
                'cls': (1, 0.2, 4.0),
                'cls_pw': (1, 0.5, 2.0),
                'obj': (1, 0.2, 4.0),
                'obj_pw': (1, 0.5, 2.0),
                'iou_t': (0, 0.1, 0.7),
                'anchor_t': (1, 2.0, 8.0),
                'anchors': (2, 2.0, 10.0),
                'fl_gamma': (0, 0.0, 2.0),
                'hsv_h': (1, 0.0, 0.1),
                'hsv_s': (1, 0.0, 0.9),
                'hsv_v': (1, 0.0, 0.9),
                'degrees': (1, 0.0, 45.0),
                'translate': (1, 0.0, 0.9),
                'scale': (1, 0.0, 0.9),
                'shear': (1, 0.0, 10.0),
                'perspective': (0, 0.0, 0.001),
                'flipud': (1, 0.0, 1.0),
                'fliplr': (0, 0.0, 1.0),
                'mosaic': (1, 0.0, 1.0),
                'mixup': (1, 0.0, 1.0)}

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)

        for _ in range(300):
            if Path('evolve.txt').exists():
                parent = 'single'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min()
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                mp, s = 0.8, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])

            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])
                hyp[k] = min(hyp[k], v[2])
                hyp[k] = round(hyp[k], 5)

            results = train(hyp.copy(), opt, device)

            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')

### graph ###
# import argparse
# import logging
# import math
# import os
# import random
# import time
# import warnings
# from copy import deepcopy
# from pathlib import Path
# from threading import Thread

# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# import torch.utils.data
# import yaml
# from torch.cuda import amp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import shutil

# import test  # import test.py to get mAP after each epoch
# from models.experimental import attempt_load
# from models.yolo import Model
# from utils.autoanchor import check_anchors
# from utils.datasets import create_dataloader, norm_imgs
# from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
#     fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
#     check_requirements, print_mutation, set_logging, one_cycle, colorstr, target2mask, check_mask
# from utils.google_utils import attempt_download
# from utils.loss import ComputeLoss
# from utils.plots import plot_images, plot_labels, plot_results, plot_evolution, plot_cluster
# from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel, time_synchronized
# from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

# logger = logging.getLogger(__name__)

# @torch.no_grad()
# def edl_stats_one_batch(model, dataloader, device):
#     model.eval()

#     batch = next(iter(dataloader))
#     imgs = batch[0]  # dataloader가 뭘 더 주든 imgs는 0번

#     imgs = imgs.to(device, non_blocking=True).float()
#     imgs = norm_imgs(imgs, model)

#     # hm_only=True로 Segmenter 출력만 뽑기
#     (pred_det, pred_seg), pred_masks = model(imgs, hm_only=True)

#     if pred_masks is None or len(pred_masks) == 0:
#         return {"ok": False, "reason": "pred_masks is None/empty"}

#     hm = pred_masks[0].detach()  # [B,C,H,W]
#     C = hm.shape[1]
#     if C != 2:
#         return {"ok": False, "heatmapC": int(C), "reason": "heatmapC != 2"}

#     evidence = F.softplus(hm)
#     alpha = evidence + 1.0
#     S = alpha.sum(dim=1)        # [B,H,W]
#     p = alpha[:, 1] / S
#     v = 2.0 / S

#     pf = p.flatten().float()
#     vf = v.flatten().float()
#     pf = pf - pf.mean()
#     vf = vf - vf.mean()
#     corr = (pf * vf).mean() / (pf.std(unbiased=False) * vf.std(unbiased=False) + 1e-12)
#     corr = float(corr.item())

#     return {
#         "ok": True,
#         "heatmapC": 2,
#         "p_mean": float(p.mean()),
#         "p_max": float(p.max()),
#         "v_mean": float(v.mean()),
#         "v_max": float(v.max()),
#         "corr": float(corr),
#     }
    
# def train(hyp, opt, device, tb_writer=None):
#     logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
#     save_dir, epochs, batch_size, total_batch_size, weights, rank = \
#         Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
#     half_precision = not opt.disable_half

#     # Directories
#     wdir = save_dir / 'weights'
#     wdir.mkdir(parents=True, exist_ok=True)  # make dir
#     last = wdir / 'last.pt'
#     best = wdir / 'best.pt'
#     results_file = save_dir / 'results.txt'

#     # Save run settings
#     with open(save_dir / 'hyp.yaml', 'w') as f:
#         yaml.safe_dump(hyp, f, sort_keys=False)
#     with open(save_dir / 'opt.yaml', 'w') as f:
#         yaml.safe_dump(vars(opt), f, sort_keys=False)
#     if rank in [-1, 0] and not opt.resume:
#         os.mkdir(save_dir / 'scripts')
#         for file in ['train.py', 'test.py', 'utils/loss.py', 'utils/general.py', 'utils/datasets.py',
#                      'models/yolo.py', 'models/common.py', opt.cfg]:
#             shutil.copyfile(file, save_dir / 'scripts' / os.path.basename(file))

#     # Configure
#     plots = not opt.evolve  # create plots
#     cuda = device.type != 'cpu'
#     init_seeds(2 + rank)
#     with open(opt.data) as f:
#         data_dict = yaml.safe_load(f)  # data dict

#     # Logging- Doing this before checking the dataset. Might update data_dict
#     loggers = {'wandb': None}  # loggers dict
#     if rank in [-1, 0]:
#         opt.hyp = hyp  # add hyperparameters
#         run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
#         wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
#         loggers['wandb'] = wandb_logger.wandb
#         data_dict = wandb_logger.data_dict
#         if wandb_logger.wandb:
#             weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

#     nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
#     names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
#     assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
#     is_coco = opt.data.endswith('coco.yaml') and nc == 80  # COCO dataset

#     # Model
#     pretrained = weights.endswith('.pt')
#     if pretrained:
#         with torch_distributed_zero_first(rank):
#             weights = attempt_download(weights)  # download if not found locally
#         ckpt = torch.load(weights, map_location=device)  # load checkpoint
#         model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
#         exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume and not opt.freeze else []  # exclude keys
#         state_dict = ckpt['model'].float().state_dict()  # to FP32
#         state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude, cfg_path=opt.cfg)  # intersect
#         model.load_state_dict(state_dict, strict=False)  # load
#         logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
#     else:
#         model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
#     with torch_distributed_zero_first(rank):
#         check_dataset(data_dict)  # check
#     train_path = data_dict['train']
#     test_path = data_dict['val']

#     # Freeze
#     freeze = []  # parameter names to freeze (full or partial)
#     unfreeze = []
#     if opt.freeze:
#         raise NotImplementedError('Navigate to here to manually set the layers to freeze and uncomment this error.')
#         hyp['weight_decay'] = 0.
#         # freeze = ['model.{}.'.format(ii) for ii in range(8)]
#         # print('freeze', freeze)
#         unfreeze = ['model.5', 'model.6'] # DWConv and Segmenter
#         print('unfreeze', unfreeze)
#     for k, v in model.named_parameters():
#         v.requires_grad = True  # train all layers
#         if any(x in k for x in freeze) or (len(unfreeze) and all(x not in k for x in unfreeze)):
#             # print('freezing %s' % k)
#             v.requires_grad = False

#     # Optimizer
#     nbs = 64  # nominal batch size
#     accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
#     hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
#     logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

#     pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
#     for k, v in model.named_modules():
#         if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter) and (v.bias.requires_grad or True):
#             pg2.append(v.bias)  # biases
#         if (isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.LayerNorm)) and (v.weight.requires_grad or True):
#             pg0.append(v.weight)  # no decay
#         elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter) and (v.weight.requires_grad or True):
#             pg1.append(v.weight)  # apply decay

#     if opt.adam:
#         # optimizer = optim.AdamW(pg0, lr=hyp['lr0'] * 0.02)  # adjust beta1 to momentum
#         optimizer = optim.AdamW(pg0, lr=hyp['lr0'] * 0.1, betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
#     else:
#         optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

#     optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
#     optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
#     logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
#     del pg0, pg1, pg2

#     # Scheduler https://arxiv.org/pdf/1812.01187.pdf
#     # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
#     if opt.linear_lr:
#         lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
#     else:
#         lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
#     # plot_lr_scheduler(optimizer, scheduler, epochs)

#     # EMA
#     ema = ModelEMA(model) if rank in [-1, 0] else None

#     # ============================
#     # SecondLook debug wrapper (학습에는 영향 X)
#     # - autoshape()가 SecondLook 래퍼로 바뀌어있다는 가정
#     # - EMA 모델로 래핑해서 train/eval 토글 영향 최소화
#     # ============================
#     sl_debug = None
#     if rank in [-1, 0]:
#         try:
#             if ema is not None:
#                 sl_debug = ema.ema.autoshape()
#             else:
#                 sl_debug = model.autoshape()
#         except Exception as e:
#             print("[SecondLook] debug wrapper build failed:", e)
#             sl_debug = None
#     debug_interval = 200  # 더 자주 보고 싶으면 20~50으로 낮춰
#     # ============================

#     # Resume
#     start_epoch, best_fitness = 0, 0.0
#     if pretrained:  # TODO: freeze
#         # Optimizer
#         if opt.resume and ckpt['optimizer'] is not None:
#             optimizer.load_state_dict(ckpt['optimizer'])
#             best_fitness = ckpt['best_fitness']
#             if rank in [-1, 0]:
#                 print('loaded optimizer, best_fitness = %.2f' % best_fitness)

#         # EMA
#         if opt.resume and ema and ckpt.get('ema'):
#             ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
#             ema.updates = ckpt['updates']
#             if rank in [-1, 0]:
#                 print('loaded ema, updates = %d' % ckpt['updates'])

#         # Results
#         if opt.resume and ckpt.get('training_results') is not None:
#             results_file.write_text(ckpt['training_results'])  # write results.txt

#         # Epochs
#         if opt.resume:
#             start_epoch = ckpt['epoch'] + 1
#             assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
#         if epochs > start_epoch > 0:
#             logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
#                         (weights, ckpt['epoch'], epochs - start_epoch))
#             # epochs += start_epoch  # finetune additional epochs

#         del ckpt, state_dict

#     # Image sizes
#     gs = max(int(model.stride.max()), 64)  # grid size (max stride)
#     nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
#     imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

#     # DP mode
#     if cuda and rank == -1 and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model)

#     # SyncBatchNorm
#     if opt.sync_bn and cuda and rank != -1:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
#         logger.info('Using SyncBatchNorm()')

#     # Trainloader
#     dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
#                                             hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
#                                             world_size=opt.world_size, workers=opt.workers,
#                                             image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
#     mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
#     nb = len(dataloader)  # number of batches
#     assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

#     # Process 0
#     if rank in [-1, 0]:
#         testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,  # testloader
#                                        hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
#                                        world_size=opt.world_size, workers=opt.workers,
#                                        pad=0., prefix=colorstr('val: '))[0]

#         if not opt.resume:
#             labels = np.concatenate(dataset.labels, 0)
#             c = torch.tensor(labels[:, 0])  # classes
#             if plots:
#                 if tb_writer:
#                     tb_writer.add_histogram('classes', c, 0)

#             # Anchors
#             if not opt.noautoanchor:
#                 check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
#             model.half().float()  # pre-reduce anchor precision

#     # DDP mode
#     if cuda and rank != -1:
#         find_unused_parameters = opt.hm_only or any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
#         model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
#                     find_unused_parameters=find_unused_parameters)

#     # Model parameters
#     hyp['box'] *= 3. / nl  # scale to layers
#     hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
#     hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
#     hyp['label_smoothing'] = opt.label_smoothing
#     model.nc = nc  # attach number of classes to model
#     model.hyp = hyp  # attach hyperparameters to model
#     model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
#     model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
#     model.names = names

#     # ===== Enable Hybrid Sparse Class Graph on cls logits (no patch gate) =====
#     if getattr(opt, 'use_class_graph', False):
#         try:
#             base_model = de_parallel(model)
#             base_model.enable_class_graph(
#                 class_names=names,
#                 device=device,
#                 clip_model_name=getattr(opt, 'graph_clip_model', 'ViT-B-32'),
#                 clip_pretrained=getattr(opt, 'graph_clip_pretrained', ''),
#                 alpha=getattr(opt, 'graph_alpha', 0.3),
#                 top_r=getattr(opt, 'graph_top_r', 2),
#                 ema_beta=getattr(opt, 'graph_ema_beta', 0.9),
#                 lambda_start=getattr(opt, 'graph_lambda_start', 0.4),
#                 lambda_end=getattr(opt, 'graph_lambda_end', 0.1),
#                 warmup_epochs=getattr(opt, 'graph_warmup_epochs', 5),
#             )
#             if ema is not None:
#                 de_parallel(ema.ema).enable_class_graph(
#                     class_names=names,
#                     device=device,
#                     clip_model_name=getattr(opt, 'graph_clip_model', 'ViT-B-32'),
#                 clip_pretrained=getattr(opt, 'graph_clip_pretrained', ''),
#                     alpha=getattr(opt, 'graph_alpha', 0.3),
#                     top_r=getattr(opt, 'graph_top_r', 2),
#                     ema_beta=getattr(opt, 'graph_ema_beta', 0.9),
#                     lambda_start=getattr(opt, 'graph_lambda_start', 0.4),
#                     lambda_end=getattr(opt, 'graph_lambda_end', 0.1),
#                     warmup_epochs=getattr(opt, 'graph_warmup_epochs', 5),
#                 )
#             print(f"[ClassGraph] enabled: clip={getattr(opt,'graph_clip_model','ViT-B-32')} alpha={getattr(opt,'graph_alpha',0.3)} top_r={getattr(opt,'graph_top_r',2)}")
#         except Exception as e:
#             print('[ClassGraph] enable failed:', e)

#     # Start training
#     t0 = time.time()
#     nw = round(hyp['warmup_epochs'] * nb) if hyp['warmup_epochs'] > 0 else -1
#     maps = np.zeros(nc)  # mAP per class
#     results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
#     scheduler.last_epoch = start_epoch - 1  # do not move
#     scaler = amp.GradScaler(enabled=cuda)
#     compute_loss = ComputeLoss(model)  # init loss class
#     logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
#                 f'Using {dataloader.num_workers} dataloader workers\n'
#                 f'Logging results to {save_dir}\n'
#                 f'Starting training for {epochs} epochs...')
#     for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
#         use_gt = epoch < epochs * 0.6
#         model.train()

#         # Update image weights (optional)
#         if opt.image_weights:
#             # Generate indices
#             if rank in [-1, 0]:
#                 cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
#                 iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
#                 dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
#             # Broadcast if DDP
#             if rank != -1:
#                 indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
#                 dist.broadcast(indices, 0)
#                 if rank != 0:
#                     dataset.indices = indices.cpu().numpy()

#         mloss = torch.zeros(7, device=device)  # mean losses
#         if rank != -1:
#             dataloader.sampler.set_epoch(epoch)
#         pbar = enumerate(dataloader)
#         logger.info(('\n' + '%10s' * 11) %
#                     ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'pixl', 'area', 'dist', 'total', 'labels', 'img_size'))
#         if rank in [-1, 0]:
#             pbar = tqdm(pbar, total=nb)  # progress bar
#         warmup_flag = False
#         for i, (imgs, targets, masks, m_weights, paths, _) in pbar:  # batch -------------------------------------------------------------
#             ni = i + nb * epoch  # number integrated batches (since train start)
#             warmup_flag = ni <= nw
#             imgs = imgs.to(device, non_blocking=True).float()
#             imgs = norm_imgs(imgs, model)
#             masks = masks.to(device, non_blocking=True).float()
#             m_weights = m_weights.to(device, non_blocking=True).float()

#             # ============================
#             # SecondLook debug call during training (학습 중 적용 여부 바로 확인)
#             # - debug_interval 마다 1장만 EMA 모델로 추론을 호출
#             # - 이때 AutoShapeSecondLookMargin 내부 print("[SecondLook] total crops: ...")가 찍혀야 정상
#             # ============================
#             if (sl_debug is not None) and (rank in [-1, 0]) and (i % debug_interval == 0):
#                 with torch.no_grad():
#                     im = imgs[0].detach().float().cpu()  # (C,H,W)
#                     # 값 범위가 0~1이 아닐 수 있어서 안전 변환
#                     im_min = float(im.min().item())
#                     im_max = float(im.max().item())
#                     if im_max <= 1.5 and im_min >= -0.5:
#                         im = (im * 255.0).clamp(0, 255)
#                     else:
#                         # min-max normalize to 0~255
#                         im = (im - im.min()) / (im.max() - im.min() + 1e-6)
#                         im = (im * 255.0).clamp(0, 255)
#                     im = im.byte().permute(1, 2, 0).numpy()  # HWC uint8

#                     # EMA autoshape 호출(추론). 여기서 SecondLook 로그가 떠야 함
#                     # _ = sl_debug(im, size=imgsz)
#             # ============================

#             # Warmup
#             if warmup_flag:
#                 xi = [0, nw]  # x interp
#                 accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
#                 for j, x in enumerate(optimizer.param_groups):
#                     x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
#                     if 'momentum' in x:
#                         x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

#             # Multi-scale
#             if opt.multi_scale:
#                 raise NotImplementedError('Interplotation will destroy the heatmap labels')
#                 sz = random.randrange(imgsz * 0.75, imgsz * 1.5 + gs) // gs * gs  # size
#                 sf = sz / max(imgs.shape[2:])  # scale factor
#                 if sf != 1:
#                     ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
#                     mask_stride = imgs.size(2) // masks.size(2)
#                     assert mask_stride == 8
#                     imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
#                     masks = F.interpolate(masks, size=[_s // mask_stride for _s in ns], mode='nearest')

#             # Forward
#             with amp.autocast(enabled=cuda and half_precision):
#                 targets = targets.to(device)
#                 pred = model((imgs, [masks]) if (use_gt and warmup_flag) else imgs, hm_only=opt.hm_only)  # forward
#                 loss, loss_items = compute_loss(pred, targets, imgsz=imgs.shape, masks=masks, m_weights=m_weights)  # loss scaled by batch_size
#                 if rank != -1:
#                     loss *= opt.world_size  # gradient averaged between devices in DDP mode
#                 if opt.quad:
#                     loss *= 4.

#             # Backward
#             scaler.scale(loss).backward()

#             # Optimize
#             if ni % accumulate == 0:
#                 scaler.step(optimizer)  # optimizer.step
#                 scaler.update()
#                 optimizer.zero_grad()
#                 if ema:
#                     ema.update(model)

#             # Print
#             if rank in [-1, 0]:
#                 mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
#                 mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
#                 s = ('%10s' * 2 + '%10.4g' * 9) % (
#                     f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
#                 pbar.set_description(s)

#                 # Plot
#                 if plots and ni < 3:
#                     f = save_dir / f'train_batch{ni}.jpg'  # filename
#                     Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
#                     if tb_writer and ni == 0 and not opt.sync_bn:
#                         with warnings.catch_warnings():
#                             warnings.simplefilter('ignore')  # suppress jit trace warning
#                             tb_writer.add_graph(torch.jit.trace(de_parallel(model), imgs, strict=False), [])  # graph
#                 elif plots and ni == 10 and wandb_logger.wandb:
#                     wandb_logger.log({'Mosaics': [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
#                                                   save_dir.glob('train*.jpg') if x.exists()]})

#             # end batch ------------------------------------------------------------------------------------------------
#         # end epoch ----------------------------------------------------------------------------------------------------

#         # Scheduler
#         lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
#         scheduler.step()

#         # DDP process 0 or single-GPU
#         if rank in [-1, 0]:
#             try:
#                 edl_stats = edl_stats_one_batch(ema.ema if ema is not None else model, testloader, device)
#                 if edl_stats.get("ok", False):
#                     print(f"[EDL][epoch {epoch}] C={edl_stats['heatmapC']} "
#                         f"p_mean={edl_stats['p_mean']:.4f} p_max={edl_stats['p_max']:.4f} "
#                         f"v_mean={edl_stats['v_mean']:.4f} v_max={edl_stats['v_max']:.4f} "
#                         f"corr={edl_stats['corr']:.4f}")
#                 else:
#                     # EDL이 안 걸리면 이유까지 출력
#                     print(f"[EDL][epoch {epoch}] NOT_OK "
#                         f"heatmapC={edl_stats.get('heatmapC', None)} reason={edl_stats.get('reason', '')}")
#             except Exception as e:
#                 print(f"[EDL][epoch {epoch}] stats failed: {e}")
        
#         if rank in [-1, 0]:
#             # mAP
#             ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
#             final_epoch = epoch + 1 == epochs
#             if not opt.notest or final_epoch:  # Calculate mAP
#                 wandb_logger.current_epoch = epoch + 1
#                 if getattr(opt, 'use_class_graph', False):
#                     results, maps, times, conf_mat = test.test(data_dict,
#                                              return_confusion=True,
#                                              use_gt=use_gt and warmup_flag,
#                                              conf_thres=0.1 if warmup_flag else 0.001,
#                                              batch_size=batch_size * 2 if '_tr' not in opt.cfg else 1,
#                                              imgsz=imgsz_test,
#                                              model=ema.ema,
#                                              single_cls=opt.single_cls,
#                                              dataloader=testloader,
#                                              save_dir=save_dir,
#                                              save_json=is_coco and final_epoch,
#                                              verbose=nc < 50 and final_epoch,
#                                              plots=plots and final_epoch,
#                                              wandb_logger=wandb_logger,
#                                              compute_loss=compute_loss,
#                                              half_precision=half_precision,
#                                              hm_metric=opt.hm_metric,
#                                              is_coco=is_coco)
#                     try:
#                         de_parallel(model).update_class_graph(conf_mat, epoch, epochs)
#                         if ema is not None:
#                             de_parallel(ema.ema).update_class_graph(conf_mat, epoch, epochs)
#                     except Exception as e:
#                         print('[ClassGraph] update failed:', e)
#                 else:
#                     results, maps, times = test.test(data_dict,
#                                          use_gt=use_gt and warmup_flag,
#                                          conf_thres=0.1 if warmup_flag else 0.001,
#                                          batch_size=batch_size * 2 if '_tr' not in opt.cfg else 1,
#                                          imgsz=imgsz_test,
#                                          model=ema.ema,
#                                          single_cls=opt.single_cls,
#                                          dataloader=testloader,
#                                          save_dir=save_dir,
#                                          save_json=is_coco and final_epoch,
#                                          verbose=nc < 50 and final_epoch,
#                                          plots=plots and final_epoch,
#                                          wandb_logger=wandb_logger,
#                                          compute_loss=compute_loss,
#                                          half_precision=half_precision,
#                                          hm_metric=opt.hm_metric,
#                                          is_coco=is_coco)

#             # Write
#             with open(results_file, 'a') as f:
#                 f.write(s + '%10.4g' * 12 % results + '\n')  # append metrics, val_loss

#             # Log
#             tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
#                     'train/pixl_loss', 'train/area_loss', 'train/dist_loss',  # train loss
#                     'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
#                     'metrics/bestpr', 'metrics/occupy',
#                     'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
#                     'val/pixl_loss', 'val/area_loss', 'val/dist_loss',  # train loss
#                     'x/lr0', 'x/lr1', 'x/lr2']  # params
#             for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
#                 if tb_writer:
#                     tb_writer.add_scalar(tag, x, epoch)  # tensorboard
#                 if wandb_logger.wandb:
#                     wandb_logger.log({tag: x})  # W&B

#             # Update best mAP
#             fi = fitness(np.array(results[:4]).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
#             if fi > best_fitness:
#                 best_fitness = fi
#             wandb_logger.end_epoch(best_result=best_fitness == fi)

#             # Save model
#             if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
#                 ckpt = {'epoch': epoch,
#                         'best_fitness': best_fitness,
#                         'training_results': results_file.read_text(),
#                         'model': deepcopy(de_parallel(model)).half(),
#                         'ema': deepcopy(ema.ema).half(),
#                         'updates': ema.updates,
#                         'optimizer': optimizer.state_dict(),
#                         'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

#                 # Save last, best and delete
#                 torch.save(ckpt, last)
#                 if best_fitness == fi:
#                     torch.save(ckpt, best)
#                 if wandb_logger.wandb:
#                     if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
#                         wandb_logger.log_model(
#                             last.parent, opt, epoch, fi, best_model=best_fitness == fi)
#                 del ckpt

#         # end epoch ----------------------------------------------------------------------------------------------------
#     # end training
#     if rank in [-1, 0]:
#         logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
#         if plots:
#             plot_results(save_dir=save_dir)  # save as results.png
#             if wandb_logger.wandb:
#                 files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
#                 wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
#                                               if (save_dir / f).exists()]})

#         if not opt.evolve:
#             if is_coco:  # COCO dataset
#                 for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
#                     results, _, _ = test.test(opt.data,
#                                               batch_size=batch_size * 2 if '_tr' not in opt.cfg else 1,
#                                               imgsz=imgsz_test,
#                                               conf_thres=0.001,
#                                               iou_thres=0.7,
#                                               model=attempt_load(m, device).half(),
#                                               single_cls=opt.single_cls,
#                                               dataloader=testloader,
#                                               save_dir=save_dir,
#                                               save_json=True,
#                                               plots=False,
#                                               half_precision=half_precision,
#                                               is_coco=is_coco)

#             # Strip optimizers
#             for f in last, best:
#                 if f.exists():
#                     strip_optimizer(f)  # strip optimizers
#             if wandb_logger.wandb:  # Log the stripped model
#                 wandb_logger.wandb.log_artifact(str(best if best.exists() else last), type='model',
#                                                 name='run_' + wandb_logger.wandb_run.id + '_model',
#                                                 aliases=['latest', 'best', 'stripped'])
#         wandb_logger.finish_run()
#     else:
#         dist.destroy_process_group()
#     torch.cuda.empty_cache()
#     return results


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
#     parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
#     parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
#     parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
#     parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
#     parser.add_argument('--rect', action='store_true', help='rectangular training')
#     parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
#     parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
#     parser.add_argument('--notest', action='store_true', help='only test final epoch')
#     parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
#     parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
#     parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
#     parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
#     parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
#     parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
#     parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
#     parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
#     parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
#     parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
#     parser.add_argument('--project', default='runs/train', help='save to project/name')
#     parser.add_argument('--entity', default=None, help='W&B entity')
#     parser.add_argument('--name', default='exp', help='save to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--quad', action='store_true', help='quad dataloader')
#     parser.add_argument('--linear-lr', action='store_true', help='linear LR')
#     parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
#     parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
#     parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
#     parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
#     parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
#     parser.add_argument('--addition', action='store_true', help='additional training')
#     parser.add_argument('--freeze', action='store_true', help='freeze shadow backbone for fine-tuning')
#     parser.add_argument('--hm-only', action='store_true', help='training on heatmap prediction only')
#     parser.add_argument('--hm-metric', action='store_true', help='use heatmap-related evaluation metrics')
#     parser.add_argument('--disable-half', action='store_true', help='disable FP16 half-precision training')
#     # ===== Hybrid Sparse Class Graph (CLIP semantic + EMA confusion) =====
#     parser.add_argument('--use-class-graph', action='store_true', help='enable hybrid sparse class graph refinement on cls logits')
#     parser.add_argument('--graph-clip-model', type=str, default='ViT-B-32', help='CLIP text encoder model name (open_clip preferred), e.g., ViT-B-32')
#     parser.add_argument('--graph-clip-pretrained', type=str, default='', help='local CLIP checkpoint file or directory (e.g., /data/siwoo/clip_ckpt/clip_vitb32_local or .../open_clip_model.safetensors)')
#     parser.add_argument('--graph-alpha', type=float, default=0.3, help='refinement strength alpha')
#     parser.add_argument('--graph-top-r', type=int, default=2, help='top-r neighbors per class')
#     parser.add_argument('--graph-ema-beta', type=float, default=0.9, help='EMA beta for confusion matrix')
#     parser.add_argument('--graph-lambda-start', type=float, default=0.4, help='lambda start (semantic weight)')
#     parser.add_argument('--graph-lambda-end', type=float, default=0.1, help='lambda end (semantic weight)')
#     parser.add_argument('--graph-warmup-epochs', type=int, default=5, help='epochs before enabling EMA confusion updates')
#     opt = parser.parse_args()

#     # Set DDP variables
#     opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
#     opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
#     set_logging(opt.global_rank)
#     if opt.global_rank in [-1, 0] and False:
#         check_git_status()
#         check_requirements(exclude=('pycocotools', 'thop'))

#     # Resume
#     wandb_run = check_wandb_resume(opt)
#     if opt.resume and not wandb_run:  # resume an interrupted run
#         ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
#         assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
#         apriori = opt.global_rank, opt.local_rank
#         addition = opt.addition
#         if addition:
#             epochs, hyp = opt.epochs, opt.hyp
#         with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
#             opt = argparse.Namespace(**yaml.safe_load(f))  # replace
#         opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
#             '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
#         if addition:
#             opt.epochs, opt.hyp = opt.epochs * 0 + epochs, hyp  # TODO: epochs
#         logger.info('Resuming training from %s' % ckpt)
#     else:
#         opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
#         assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
#         opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
#         opt.name = 'evolve' if opt.evolve else opt.name
#     opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))

#     # DDP mode
#     opt.total_batch_size = opt.batch_size
#     device = select_device(opt.device, batch_size=opt.batch_size)
#     if opt.local_rank != -1:
#         assert torch.cuda.device_count() > opt.local_rank
#         torch.cuda.set_device(opt.local_rank)
#         device = torch.device('cuda', opt.local_rank)
#         dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
#         assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
#         assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
#         opt.batch_size = opt.total_batch_size // opt.world_size

#     # Hyperparameters
#     with open(opt.hyp) as f:
#         hyp = yaml.safe_load(f)  # load hyps

#     # Train
#     logger.info(opt)
#     if not opt.evolve:
#         tb_writer = None  # init loggers
#         if opt.global_rank in [-1, 0] and False:
#             prefix = colorstr('tensorboard: ')
#             logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
#             tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard

#         train(hyp, opt, device, tb_writer)

#     # Evolve hyperparameters (optional)
#     else:
#         meta = {'lr0': (1, 1e-5, 1e-1),
#                 'lrf': (1, 0.01, 1.0),
#                 'momentum': (0.3, 0.6, 0.98),
#                 'weight_decay': (1, 0.0, 0.001),
#                 'warmup_epochs': (1, 0.0, 5.0),
#                 'warmup_momentum': (1, 0.0, 0.95),
#                 'warmup_bias_lr': (1, 0.0, 0.2),
#                 'box': (1, 0.02, 0.2),
#                 'cls': (1, 0.2, 4.0),
#                 'cls_pw': (1, 0.5, 2.0),
#                 'obj': (1, 0.2, 4.0),
#                 'obj_pw': (1, 0.5, 2.0),
#                 'iou_t': (0, 0.1, 0.7),
#                 'anchor_t': (1, 2.0, 8.0),
#                 'anchors': (2, 2.0, 10.0),
#                 'fl_gamma': (0, 0.0, 2.0),
#                 'hsv_h': (1, 0.0, 0.1),
#                 'hsv_s': (1, 0.0, 0.9),
#                 'hsv_v': (1, 0.0, 0.9),
#                 'degrees': (1, 0.0, 45.0),
#                 'translate': (1, 0.0, 0.9),
#                 'scale': (1, 0.0, 0.9),
#                 'shear': (1, 0.0, 10.0),
#                 'perspective': (0, 0.0, 0.001),
#                 'flipud': (1, 0.0, 1.0),
#                 'fliplr': (0, 0.0, 1.0),
#                 'mosaic': (1, 0.0, 1.0),
#                 'mixup': (1, 0.0, 1.0)}

#         assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
#         opt.notest, opt.nosave = True, True
#         yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'
#         if opt.bucket:
#             os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)

#         for _ in range(300):
#             if Path('evolve.txt').exists():
#                 parent = 'single'
#                 x = np.loadtxt('evolve.txt', ndmin=2)
#                 n = min(5, len(x))
#                 x = x[np.argsort(-fitness(x))][:n]
#                 w = fitness(x) - fitness(x).min()
#                 if parent == 'single' or len(x) == 1:
#                     x = x[random.choices(range(n), weights=w)[0]]
#                 elif parent == 'weighted':
#                     x = (x * w.reshape(n, 1)).sum(0) / w.sum()

#                 mp, s = 0.8, 0.2
#                 npr = np.random
#                 npr.seed(int(time.time()))
#                 g = np.array([x[0] for x in meta.values()])
#                 ng = len(meta)
#                 v = np.ones(ng)
#                 while all(v == 1):
#                     v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
#                 for i, k in enumerate(hyp.keys()):
#                     hyp[k] = float(x[i + 7] * v[i])

#             for k, v in meta.items():
#                 hyp[k] = max(hyp[k], v[1])
#                 hyp[k] = min(hyp[k], v[2])
#                 hyp[k] = round(hyp[k], 5)

#             results = train(hyp.copy(), opt, device)

#             print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

#         plot_evolution(yaml_file)
#         print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
#               f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')