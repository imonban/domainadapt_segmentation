import torch
import helper_utils.configs as help_configs 
import helper_utils.utils   as help_utils 
from models.model_factory import model_factory
from monai.data import DataLoader
from data_factories.kits_factory import kit_factory
import  helper_utils.transforms  as help_transforms 
import pickle  as pkl 
## unknown imports 
import torch
import pickle as pkl
from data_factories.kits_factory import kit_factory
from models.model_factory import model_factory
from monai.inferers import sliding_window_inference
from monai.data import (
    decollate_batch,
)  # this is needed wherever i run the iterator
from tqdm import tqdm
from monai.transforms import (
    Compose,
    AsDiscrete,
    AsDiscreted,
    Activationsd
)
from monai.transforms import Invertd, SaveImaged, RemoveSmallObjectsd
from hashlib import sha224

def subject_formater(metadict,self):
    pid = metadict['filename_or_obj']
    out_form=sha224(pid.encode('utf-8')).hexdigest()
    return {'subject':f"{out_form}","idx":"0"}
def make_post_transforms(test_conf,test_transforms):
    out_dir = test_conf["output_dir"]
    bin_preds = True #TODO: is it woth having continious outputs 
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred",argmax=True),
            RemoveSmallObjectsd(keys="pred", min_size=500),
            SaveImaged(
                keys="pred",
                meta_keys="label",
                output_dir=out_dir,
                output_postfix="seg",
                resample=False,
                data_root_dir="",
                output_name_formatter=subject_formater
            ),
        ]
    )
    return post_transforms

def main():
    config = help_configs.get_test_params() 
    weight_path = config['model_weight'] 
    output_dir = config['output_dir']
    device= config['device']
    train_conf, weights = help_utils.load_weights(weight_path=weight_path)
    model= model_factory(config=train_conf) 
    model.load_state_dict(weights)
    with open(train_conf['data_path'],'rb' ) as f : 
        test = pkl.load(f) 
        test = test[-1] # TODO: DON'T KEEP THIS FOREVER 
    dset = kit_factory('basic') # dset that is not cached 
    test_t = help_transforms.gen_test_transforms(confi=train_conf)
    test_ds = dset(test,transform=test_t)
    test_loader = DataLoader(test_ds,
    batch_size = 1,
    shuffle=False,
    num_workers = 8,
    collate_fn = help_transforms.ramonPad())
    model = model.to(device=device)
    model.eval() 
    post_transform = make_post_transforms(config,test_transforms=test_t)
    roi_size = (96,96,32)# train_conf['spacing_vox_dim']
    sw_batch_size=  1 
    with torch.no_grad(): 
        for test_data in tqdm(test_loader,total=len(test_loader)):
            test_inputs = test_data['image'].to(device)
            roi_size = roi_size 
            test_data["pred"] = sliding_window_inference(
                test_inputs,
                roi_size,
                sw_batch_size,
                model,
            )
            other  = [post_transform(i) for i in decollate_batch(test_data)][0]
            breakpoint()
if __name__ =='__main__': 
    main() 