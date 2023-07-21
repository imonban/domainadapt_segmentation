import sys
import json
from helpers import *
from data_factories.kits_factory import kit_factory
from monai.data import DataLoader
from models.model_factory import model_factory
import pickle as pkl
from monai.losses import DiceCELoss
import torch
from monai.data import DataLoader
import os
import pdb
from batch_iterators.train_iterators import *
from torch.utils.data import WeightedRandomSampler
from torch import _dynamo
import random
import shutil
import os

torch._dynamo.config.suppress_errors = True
torch.multiprocessing.set_sharing_strategy("file_system")

def get_config():
    conf_path = sys.argv[1]
    with open(conf_path, "r") as f:
        conf = json.load(f)
    return conf


def load_data(data_path):
    # all code will assume this has (train,test,val)
    with open(data_path, "rb") as f:
        all_data = pkl.load(f)
    return all_data


def makeWeightedsampler(ds):
    classes = [0, 1]
    phase_list = [e["phase"] for e in ds]
    cls_counts = [0, 0]
    cls_counts[0] = len(phase_list) - sum(phase_list)
    cls_counts[1] = sum(phase_list)
    cls_weights = [1 / e for e in cls_counts]
    sample_weight = list()
    for e in phase_list:
        sample_weight.append(cls_weights[e])
    sample_weight = torch.tensor(sample_weight)
    return WeightedRandomSampler(sample_weight, len(sample_weight))


if __name__ == "__main__":
    conf = get_config()
    dset = kit_factory("cached")
    train, val, test = load_data(conf["data_path"])
    # use short circuitting to check if dev is  a field
    if "dev" in conf.keys() and conf["dev"] == True:
        print(
            "we are outputting to devset we are therefore using a smaller train sample for dev"
        )
        train = random.sample(train, 50)
        val = random.sample(val, 30)

    print(f"Len train is {len(train)}")
    print(f"Len val is {len(val)}")
    print(f"Len test is {len(test)}")
    lr = conf["learn_rate"]
    momentum = conf["momentum"]
    vox_dim = conf["vox_dim"]
    pix_dim = conf["pix_dim"]
    train_transform, val_transform = gen_transforms(conf)
    # apperently i can somehow use some caching
    batch_size = conf["batch_size"]
    cache_dir = conf["cache_dir"]
    train_ds = dset(
        train,
        transform=train_transform,
        cache_dir=os.path.join(os.environ["HOME"], cache_dir),
    )
    num_workers = conf["num_workers"]
    if conf["mode"] == "debias" or conf["mode"] == "mixed":
        sampler = makeWeightedsampler(train)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ramonPad(),
        sampler=sampler,
    )
    val_ds = dset(
        val,
        transform=val_transform,
        cache_dir=os.path.join(os.environ["HOME"], cache_dir),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ramonPad(),
    )
    val_train_ds = dset(
        train,
        transform=val_transform,
        cache_dir=os.path.join(os.environ["HOME"], cache_dir),
    )
    val_train_loader = DataLoader(
        val_train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ramonPad(),
    )
    loaders = (train_loader, val_loader, val_train_loader)
    DEVICE = torch.device(conf["device"])
    model = model_factory(config=conf)
    if "pretrained" in conf.keys():
        ck = torch.load(conf["pretrained"])
        model.load_state_dict(ck["state_dict"])
        print("state dict loaded")
    model = model.to(torch.float32).to(DEVICE)
    model = torch.compile(model, fullgraph=False, dynamic=True)

    # TODO: make the dice metric and loss function modifiable
    loss_function = DiceCELoss(
        include_background=True, reduction="mean", to_onehot_y=True, softmax=True
    )
    if conf["model_name"] == "deepsuper" and conf["mode"] != "debias":
        # maybe also start with 0.01  for this  one
        # originally the best was 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=conf["epochs"]
        )
        train_deepsup_batch(
            model,
            loaders,
            optimizer,
            lr_scheduler,
            loss_function,
            device=DEVICE,
            config=conf,
        )
    if conf["model_name"] == "twoBranch":
        # get the params for each of the 3 poritons  of the model
        unet_enc_params = list()
        unet_dec_params = list()
        bottle_discrim_params = list()
        mask_discrim_params = list()
        param_names = list()
        optis = dict()
        schs = dict()
        optis["all_model"] = torch.optim.Adagrad(model.parameters(), lr)
        schs["all_model"] = torch.optim.lr_scheduler.PolynomialLR(optis["all_model"])
        """
        for n,e in model.named_parameters():
            if 'lin_layer' in n : 
                bottle_discrim_params.append(e)
            else:
                if 'mask_discrim' in n: 
                    mask_discrim_params.append(e)
                else: 
                    if 'input' in n  or 'downsamples' in n or 'bottleneck' in n:
                        unet_enc_params.append(e) 
                    if 'upsamples' in n or 'output' in n:
                        unet_dec_params.append(e) 
        optis=dict() 
        optis['']
        optis['unet_enc'] = torch.optim.Adagrad(unet_enc_params,lr)
        optis['unet_dec'] = torch.optim.Adagrad(unet_dec_params,lr)
        optis['bottle_discrim']=  torch.optim.Adagrad(bottle_discrim_params,lr)
        optis['mask_discrim']= torch.optim.Adagrad(mask_discrim_params,lr)
        schs = dict() # schedulers 
        schs['unet_enc']= torch.optim.lr_scheduler.PolynomialLR(optis['unet_enc']) 
        schs['unet_dec']= torch.optim.lr_scheduler.PolynomialLR(optis['unet_dec']) 
        schs['bottle_discrim'] = torch.optim.lr_scheduler.StepLR(optis['bottle_discrim'],step_size=100)
        schs['mask_discrim']= torch.optim.lr_scheduler.StepLR(optis['mask_discrim'],step_size=100)
        """
        train_two_branch(
            model,
            loaders,
            optis,
            schs,
            loss_function,
            device=DEVICE,
            config=conf,
        )

    if conf["model_name"] == "debias_unet" and conf["mode"] == "debias":
        optimizer = torch.optim.SGD(model.parameters(), lr)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=conf["epochs"]
        )
        train_debias_batch(
            model,
            loaders,
            optimizer,
            lr_scheduler,
            loss_function,
            device=DEVICE,
            config=conf,
        )
    if conf["model_name"] == "debias_dinsdale" and conf["mode"] == "debias":
        encoder, decoder, discrim = list(), list(), list()
        for n, e in model.named_parameters():
            if (
                n.startswith("input")
                or n.startswith("downsamples")
                or n.startswith("bottleneck")
            ):
                encoder.append(e)
            if n.startswith("upsamples") or n.startswith("output"):
                decoder.append(e)
            if n.startswith("domain"):
                discrim.append(e)
        encoder_opti = torch.optim.SGD(encoder, lr)
        decoder_opti = torch.optim.SGD(decoder, lr)
        discrim_opti = torch.optim.SGD(discrim, lr)
        optimizer = (encoder_opti, decoder_opti, discrim_opti)
        lr_scheduler = [
            torch.optim.lr_scheduler.PolynomialLR(
                encoder_opti, total_iters=conf["epochs"]
            ),
            torch.optim.lr_scheduler.PolynomialLR(
                discrim_opti, total_iters=conf["epochs"]
            ),
            torch.optim.lr_scheduler.PolynomialLR(
                decoder_opti, total_iters=conf["epochs"]
            ),
        ]
        train_debias_dinsdale(
            model,
            loaders,
            optimizer,
            lr_scheduler,
            loss_function,
            device=DEVICE,
            config=conf,
        )

    if (
        conf["model_name"] == "unet"
        and conf["mode"] != "debias"
        or conf["model_name"] == "kits23unet"
    ):
        # lr should be 0.01 for these experiments
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=conf["epochs"]
        )
        train_batch(
            model,
            loaders,
            optimizer,
            lr_scheduler,
            loss_function,
            device=DEVICE,
            config=conf,
        )
