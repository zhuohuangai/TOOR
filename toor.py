#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from easydl import *
import argparse, math, time, json, os

from lib import wrn_d, transform
from config import config
import numpy as np
import random
# from build_dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="PI", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL]")
parser.add_argument("--em", default=0.2, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=500, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="crossset", type=str, help="dataset name : [crossset, svhn, cifar10, cifar100, mnist, imagenet32, tin]")
parser.add_argument("--n_labels", "-n", default=1000, type=int, help="the number of labeled data")
parser.add_argument("--n_unlabels", "-u", default=100000, type=int, help="the number of unlabeled data")
parser.add_argument('--n_valid', default=5000, type=int)
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument('--temper', default=0.8, type=float, help='temperature scaling')
parser.add_argument("--ratio", "-rat", default=0.5, type=float, help="ratio for class mismatch")
parser.add_argument("--th", "-th", default=0.6, type=float, help="threshold")
parser.add_argument("--wth", "-wth", default=0.5, type=float, help="threshold")
parser.add_argument("--gpus", default=1, type=int, help="number of GPUs") # using 1 GPUs.
parser.add_argument("--num-classes", "-ncls", default=6, type=int, help="number of classes.")
parser.add_argument("--seed", "-s", default=0, type=int, help="train seed")
args = parser.parse_args()

# calculate softmax score
def cal_smscore(outputs, temper):
    # Using temperature scaling
    nnOutputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = nnOutputs - torch.max(nnOutputs, 1, keepdim=True).values
    nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs), 1, keepdim=True)
    smscore = torch.max(nnOutputs, 1).values
    smscore -= smscore.min(0).values
    smscore = smscore / (smscore.max(0).values - smscore.min(0).values)

    return smscore

# calculate prediction margin
def cal_pred_weight(outputs):
    n = len(outputs)
    nnOutputs = outputs - torch.max(outputs, 1, keepdim=True).values
    nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs), 1, keepdim=True)
    first_score = torch.max(nnOutputs, 1, keepdim=True).values
    tmp = torch.zeros_like(nnOutputs)
    tmp[torch.arange(n), torch.argmax(nnOutputs, 1)] = nnOutputs[torch.arange(n), torch.argmax(nnOutputs, 1)]
    outputs_ = nnOutputs - tmp
    second_score = torch.max(outputs_, 1, keepdim=True).values
    pred_margin = torch.sub(first_score, second_score)
    pred_margin -= pred_margin.min(0).values
    pred_margin = pred_margin / (pred_margin.max(0).values - pred_margin.min(0).values)
    return pred_margin

def weight_norm(weight):
    mean = torch.mean(weight)
    weight_norm = weight / mean
    return weight_norm

def combine_weight(w_c, w_d):
    var_c = w_c.var()
    var_d = w_d.var()
    weight = var_d / (var_d + var_c) * w_d + var_c / (var_d + var_c) * w_c
    return weight

if args.gpus < 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    device = "cpu"
else:
    gpu_ids = select_GPUs(args.gpus)
    device = gpu_ids[0]

condition = {}
exp_name = ""

def TempScale(p, t):
    return p / t

class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

_NUM_WORKERS = 2

print("dataset : {}".format(args.dataset))
condition["dataset"] = args.dataset
exp_name += str(args.dataset) + "_"

dataset_cfg = config[args.dataset]
transform_fn = transform.transform(*dataset_cfg["transform"]) # transform function (flip, crop, noise)
args.num_classes = dataset_cfg['num_classes']

l_train_dataset = dataset_cfg["dataset"](args.root, "l_train")
u_train_dataset = dataset_cfg["dataset"](args.root, "u_train")
val_dataset = dataset_cfg["dataset"](args.root, "val")
test_dataset = dataset_cfg["dataset"](args.root, "test")

# val_cifar100_dataset = dataset_cfg["dataset"](args.root, "val_cifar100")
# test_cifar100_dataset = dataset_cfg["dataset"](args.root, "test_cifar100")
# val_tinyimagenet_dataset = dataset_cfg["dataset"](args.root, "val_tinyimagenet")
# test_tinyimagenet_dataset = dataset_cfg["dataset"](args.root, "test_tinyimagenet")

print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
condition["number_of_data"] = {
    "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
    "validation":len(val_dataset), "test":len(test_dataset)
}

shared_cfg = config["shared"]

if args.alg != "supervised":
    # batch size = 0.5 x batch size
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2),
        num_workers=_NUM_WORKERS, # pin_memory=True,
    )
else:
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"], drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]),
        num_workers=_NUM_WORKERS, # pin_memory=True,
    )
print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

u_loader = DataLoader(
    u_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2),
    num_workers=_NUM_WORKERS, # pin_memory=True,
)

val_loader = DataLoader(val_dataset, 128, num_workers=_NUM_WORKERS, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 128, num_workers=_NUM_WORKERS, shuffle=False, drop_last=False)

print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

if args.em > 0:
    print("entropy minimization : {}".format(args.em))
    exp_name += "em_"
condition["entropy_maximization"] = args.em

model = wrn_d.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
model = nn.DataParallel(model, device_ids=gpu_ids, output_device=device).train(True)
optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])

trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
print("trainable parameters : {}".format(trainable_paramters))

if args.alg == "VAT": # virtual adversarial training
    from lib.algs.vat import VAT
    ssl_obj = VAT(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
elif args.alg == "PL": # pseudo label
    from lib.algs.pseudo_label import PL
    ssl_obj = PL(alg_cfg["threashold"])
elif args.alg == "MT": # mean teacher
    from lib.algs.mean_teacher import MT
    t_model = wrn_d.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    t_model = nn.DataParallel(t_model, device_ids=gpu_ids, output_device=device).train(True)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = MT(t_model, alg_cfg["ema_factor"])
elif args.alg == "PI": # PI Model
    from lib.algs.pimodel import PiModel
    ssl_obj = PiModel()
elif args.alg == "MM": # MixMatch
    from lib.algs.mixmatch import MixMatch
    ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
elif args.alg == "supervised":
    pass
else:
    raise ValueError("{} is unknown algorithm".format(args.alg))

fname = 'toor_usesmscore_' + args.alg + '_' + args.dataset + '_' + str(args.th) + '_' + str(args.wth) + '_' + str(args.ratio) + '_' + str(args.seed) + '.txt'

print()
iteration = 0
maximum_val_acc = 0
ema_pred = torch.zeros(len(u_train_dataset), dataset_cfg["num_classes"]).to(device)
weight_all = torch.zeros(len(u_train_dataset), ).to(device)
sm_score = torch.zeros(shared_cfg["batch_size"]//2, ).to(device)
domain_weight = torch.zeros(shared_cfg["batch_size"], ).to(device)
class_weight = torch.zeros(shared_cfg["batch_size"], ).to(device)
weight = torch.zeros(shared_cfg["batch_size"], ).to(device)
iters_per_epoch = len(u_train_dataset) // (shared_cfg["batch_size"]//2)
pred_id_mask = torch.zeros_like(sm_score)
pred_ood_mask = torch.zeros_like(sm_score)
utr_mask = torch.zeros_like(domain_weight)

epoch = 0
s = time.time()
for l_data, u_data in zip(l_loader, u_loader):
    iteration += 1
    if (iteration-1) % iters_per_epoch == 0:
        epoch = epoch + 1
    l_input, target, _ = l_data
    l_input, target = l_input.to(device).float(), target.to(device).long()

    if args.alg != "supervised": # for ssl algorithm
        u_input, dummy_target, u_idx = u_data
        u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()
        id_mask = dummy_target < args.num_classes
        ood_mask = dummy_target > args.num_classes - 1

        inputs = torch.cat([l_input, u_input], 0)

        dis, outputs = model(inputs)

        with torch.no_grad():
            ema_pred[u_idx] = outputs[len(target):].data.clone()*0.5 + ema_pred[u_idx]*0.5
            ema_iter_pred = ema_pred[u_idx] / torch.clamp_min(torch.sum(ema_pred[u_idx], dim=1, keepdim=True), min=1.0)
            sm_score = ema_iter_pred.max(1).values
            pred_id_mask = sm_score > args.th
            pred_ood_mask = sm_score <= args.th

            dis = dis - dis.min(0).values
            domain_weight = weight_norm(dis).squeeze()

            class_weight = weight_norm(sm_score).squeeze()

            weight_all[u_idx] = combine_weight(class_weight[-len(target):], domain_weight[-len(target):])
            weight = weight_all[u_idx]
            weight[pred_id_mask] = 1
            utr_mask = weight < args.wth
            weight[utr_mask] = 0

        adv_coef = math.exp(-5 * (1 - min((iteration - 50000)/500000, 1))**2)

        tmp1 = weight * (nn.BCELoss(reduction='none')(dis[-len(target):], torch.zeros_like(dis[-len(target):]))).mean(1)
        tmp2 = nn.BCELoss()(dis[:len(target)], torch.ones_like(dis[:len(target)]))
        adv_loss = torch.mean(tmp1) + tmp2
        adv_loss = adv_loss * adv_coef

        # ramp up exp(-5(1 - t)^2)
        coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration/shared_cfg["warmup"], 1))**2)
        ssl_loss = ssl_obj(inputs[len(target):], outputs[len(target):].detach(), model, pred_id_mask.float()) * coef

    else:
        _, outputs = model(l_input)
        coef = 0
        ssl_loss = torch.zeros(1).to(device)
        adv_loss = torch.zeros(1).to(device)


    # supervised loss
    cls_loss = F.cross_entropy(outputs[:len(target)], target, reduction="none", ignore_index=-1).mean()

    loss = cls_loss + ssl_loss + adv_loss # + cls_loss_2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.alg == "MT" or args.alg == "ICT":
        # parameter update with exponential moving average
        ssl_obj.moving_average(model.parameters())
    # display
    if iteration == 1 or (iteration % 100) == 0:
        wasted_time = time.time() - s
        rest = (shared_cfg["iteration"] - iteration)/100 * wasted_time / 60
        print("it[{}/{}] clsloss{:.2e} SSLloss{:.2e} coef{:.2f} advloss{:.2e} advco{:.2f} tim{:.1f} it/sec rst{:.1f}min lr{:.2e} idnum{} oodnum{}".format(
            iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, adv_loss.item(), adv_coef, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"], pred_id_mask.float().sum(), pred_ood_mask.float().sum() ),
            "\r", end="")
        s = time.time()

    # validation
    if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"]:
        with torch.no_grad():
            model.eval()
            print()
            print("### validation ###")
            sum_acc = 0.
            s = time.time()
            tot = 0
            for j, data in enumerate(val_loader):
                input, target, _ = data
                input, target = input.to(device).float(), target.to(device).long()

                _, output = model(input)

                pred_label = output.max(1)[1]
                # args.th = pred_val.mean().item()
                sum_acc += (pred_label == target).float().sum()
                tot += pred_label.size(0)
                if ((j+1) % 10) == 0:
                    d_p_s = 10/(time.time()-s)
                    print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                        j+1, len(val_loader), d_p_s, (len(val_loader) - j-1)/d_p_s
                    ), "\r", end="")
                    s = time.time()
            acc = sum_acc / tot
            with open(os.path.join('parasens_exp', fname), 'a') as f:
                f.write(str(acc.item()) + ' ' + str(pred_ood_mask.int().sum().item()) + '\n')
            print()
            print("varidation accuracy : {}".format(acc))
            # test
            if maximum_val_acc < acc:
                print("### test ###")
                maximum_val_acc = acc
                sum_acc = 0.
                s = time.time()
                tot = 0
                for j, data in enumerate(test_loader):
                    input, target, _ = data
                    input, target = input.to(device).float(), target.to(device).long()
                    _, output = model(input)
                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                    tot += pred_label.size(0)
                    if ((j+1) % 10) == 0:
                        d_p_s = 100/(time.time()-s)
                        print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                            j+1, len(test_loader), d_p_s, (len(test_loader) - j-1)/d_p_s
                        ), "\r", end="")
                        s = time.time()
                print()
                test_acc = sum_acc / tot
                print("test accuracy : {}".format(test_acc))
                torch.save(model.state_dict(), os.path.join(args.output, "best_model.pth"))
        model.train()
        s = time.time()
    # lr decay
    if iteration == shared_cfg["lr_decay_iter"]:
        optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]

print("test acc : {}".format(test_acc))
condition["test_acc"] = test_acc.item()

exp_name += str(int(time.time())) # unique ID
if not os.path.exists(args.output):
    os.mkdir(args.output)
with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
    json.dump(condition, f)