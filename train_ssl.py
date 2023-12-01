#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydl import *
import argparse, math, time, json, os

from lib import wrn_d, transform
from config import config
from build_dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="PI", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL]")
parser.add_argument("--em", default=0.2, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=500, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="svhn", type=str, help="dataset name : [svhn, cifar10, mnist]")
parser.add_argument("--n_labels", "-n", default=2400, type=int, help="the number of labeled data")
parser.add_argument("--n_unlabels", "-u", default=20000, type=int, help="the number of unlabeled data")
parser.add_argument('--n_valid', default=5000, type=int)
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument('--temper', default=0.8, type=float, help='temperature scaling')
parser.add_argument("--ratio", "-rat", default=0.0, type=float, help="ratio for class mismatch")
parser.add_argument("--threshold", "-th", default=1-1e-3, type=float, help="threshold")
parser.add_argument("--gpus", default=1, type=int, help="number of GPUs") # using 2 GPUs.
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

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

if args.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    device = "cpu"
else:
    gpu_ids = select_GPUs(args.gpus)
    device = gpu_ids[0]

condition = {}
exp_name = ""

print("dataset : {}".format(args.dataset))
condition["dataset"] = args.dataset
exp_name += str(args.dataset) + "_"

dataset_cfg = config[args.dataset]
transform_fn = transform.transform(*dataset_cfg["transform"]) # transform function (flip, crop, noise)


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

shared_cfg = config["shared"]
print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

data_loaders = get_dataloaders(dataset=args.dataset, n_labels=args.n_labels, n_unlabels=args.n_unlabels, n_valid=args.n_valid,
                                l_batch_size=shared_cfg["batch_size"]//2, ul_batch_size=shared_cfg["batch_size"]//2,
                                test_batch_size=128, iterations=shared_cfg["iteration"], ratio=args.ratio)
l_loader = data_loaders['labeled']
u_loader = data_loaders['unlabeled']
test_loader = data_loaders['test']
val_loader = data_loaders['valid']

print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

model = wrn_d.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
model = nn.DataParallel(model, device_ids=gpu_ids, output_device=device).train(True)
optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])

if args.alg == "VAT": # virtual adversarial training
    from lib.algs.vat import VAT
    ssl_obj = VAT(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
elif args.alg == "PL": # pseudo label
    from lib.algs.pseudo_label import PL
    ssl_obj = PL(alg_cfg["threashold"])
elif args.alg == "MT": # mean teacher
    from lib.algs.mean_teacher import MT
    t_model = wrn_d.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = MT(t_model, alg_cfg["ema_factor"])
elif args.alg == "PI": # PI Model
    from lib.algs.pimodel import PiModel
    ssl_obj = PiModel()
elif args.alg == "supervised":
    pass
else:
    raise ValueError("{} is unknown algorithm".format(args.alg))


print()
iteration = 0
maximum_val_acc = 0
ens_ood = torch.zeros((args.n_unlabels,), requires_grad=False)
weight_all = torch.zeros((args.n_unlabels,), requires_grad=False).cuda()
iterations_per_epoch = (args.n_unlabels) // (shared_cfg["batch_size"]//2)
s = time.time()
for l_data, u_data in zip(l_loader, u_loader):
    iteration += 1
    l_input, target, _ = l_data
    l_input, target = l_input.to(device).float(), target.to(device).long()

    if args.alg != "supervised": # for ssl algorithm
        u_input, u_target, u_idx = u_data
        u_input, u_target = u_input.to(device).float(), u_target.to(device).long()

        target = torch.cat([target, u_target], 0)
        id_data_mask = u_target < 6
        ood_data_mask = u_target >= 6
        target[-len(u_target):] = -1
        unlabeled_mask = (target == -1)
        labeled_mask = (target != -1)

        inputs = torch.cat([l_input, u_input], 0)
        outputs, outputs_d = model(inputs)
        smscore = cal_smscore(outputs.detach(), args.temper).cpu()
        
        if iteration < iterations_per_epoch:
            ens_ood[u_idx] = smscore[unlabeled_mask]
        else:
            ens_ood[u_idx] = 0.95 * ens_ood[u_idx] + 0.05 * smscore[unlabeled_mask]

        ood_mask = (ens_ood[u_idx] < args.threshold).to(device)
        id_mask = (ens_ood[u_idx] >= args.threshold).to(device)
        labeled_mask[-len(u_target):] = id_mask

        outputs_d = outputs_d - outputs_d.min(0).values
        domain_weight = outputs_d / (outputs_d.max(0).values - outputs_d.min(0).values)
        domain_weight = weight_norm(domain_weight).squeeze()

        class_weight = cal_pred_weight(outputs)
        class_weight = weight_norm(class_weight).squeeze()

        weight = combine_weight(class_weight[-len(u_target):], domain_weight[-len(u_target):])
        weight_all[u_idx] = weight

        if iteration < 50000:
            adv_coef = 0
            adv_loss = torch.zeros(1).to(device)
        else:
            adv_coef = math.exp(-5 * (1 - min((iteration - 50000)/500000, 1))**2)
            weight = weight_all[u_idx]
            utr_mask = weight < 0.95
            weight[utr_mask] = 0

            tmp = weight * nn.BCELoss(reduction='none')(outputs_d[-len(u_target):][ood_mask], torch.zeros_like(outputs_d[-len(u_target):][ood_mask]))
            adv_loss = torch.mean(tmp)
            adv_loss += nn.BCELoss()(outputs_d[id_mask], torch.ones_like(outputs_d[id_mask])).data[0]
            adv_loss = adv_loss * adv_coef

        # ramp up exp(-5(1 - t)^2)
        coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration/shared_cfg["warmup"], 1))**2)
        ssl_loss = ssl_obj(inputs, outputs.detach(), model, labeled_mask.float()) * coef

    else:
        output, _ = model(l_input)
        coef = 0
        ssl_loss = torch.zeros(1).to(device)

    # supervised loss
    cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
    loss = cls_loss + ssl_loss + adv_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.alg == "MT":
        # parameter update with exponential moving average
        ssl_obj.moving_average(model.parameters())
    # display
    if iteration == 1 or (iteration % 100) == 0:
        wasted_time = time.time() - s
        rest = (shared_cfg["iteration"] - iteration)/100 * wasted_time / 60
        print("it[{}/{}] clsloss{:.2e} SSLloss{:.2e} coef{:.2f} advloss{:.2e} advco{:.2f} tim{:.1f} it/sec rst{:.1f}min lr{:.2e} idnum{} oodnum{}".format(
            iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, adv_loss.item(), adv_coef, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"], id_mask.float().sum(), ood_mask.float().sum() ),
            "\r", end="")
        s = time.time()

    del inputs, l_input, u_input, id_data_mask, ood_data_mask, labeled_mask, unlabeled_mask, smscore, ood_mask, id_mask, outputs_d, domain_weight, class_weight, weight
    torch.cuda.empty_cache()

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
                output, _ = model(input)
                pred_label = output.max(1)[1]
                sum_acc += (pred_label == target).float().sum()
                tot += pred_label.size(0)
                if ((j+1) % 10) == 0:
                    d_p_s = 10/(time.time()-s)
                    print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                        j+1, len(val_loader), d_p_s, (len(val_loader) - j-1)/d_p_s
                    ), "\r", end="")
                    s = time.time()
            acc = sum_acc / tot
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
                    output, _  = model(input)
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
                # torch.save(model.state_dict(), os.path.join(args.output, "model.pth"))
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
