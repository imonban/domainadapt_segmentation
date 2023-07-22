import argparse
import json  
import pdb 
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        data_dict =json.load(values)
        arg_list = list()
        for e in data_dict: 
            arg_list.append(e)
            arg_list.append(data_dict[e])
        parser.parse_args(arg_list)

def build_args():
    """Parses args. Must include all hyperparameters you want to tune."""
    parser = argparse.ArgumentParser(
        description="Confguration for my deep learning model training for segmentation"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to pickle file containing tuple of form (train_set,val_set,test_set) see Readme for more)",
    )  # TODO: uPDATE README TO EXPLAIN CONFI OF PICKLE FILE
    parser.add_argument(
        '--config_path',
        required=True,
        type=open,
        action=LoadFromFile,
        help = 'Path'
    )
    parser.add_argument(
        "--vox_dim",
        required=list,
        type=int,
        help="Number of voxels in a patch used during training (dim,dim,32)",
    )
    parser.add_argument(
        "--pix_dim",
        required=list,
        type=float,
        help="Pixel Dimension again  ( dimxdimxdim",
    )
    parser.add_argument(
        "--learn_rate",
        required=True,
        type=float,
        help="Initial Learning rate of our model ",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=["unet"],
        help="Name of model to be used ",
    )
    parser.add_argument("--epochs", required=True, type=int, help="")
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--spacing_vox_dim", type=list, required=True)
    parser.add_argument("--spacing_pix_dim", type=list, required=True)
    parser.add_argument(
        "--spacing_img_interp", type=str, required=True, choices=["bilinear"]
    )
    parser.add_argument(
        "--spacing_lbl_interp", type=str, required=True, choices=["nearest"]
    )
    parser.add_argument(
        "--scale_intensity_vmin", type=float, required=True, default=-79
    )
    parser.add_argument(
        "--scale_intensity_vmax", type=float, required=True, default=-304
    )
    parser.add_argument("--scale_intensity_bmin", type=float, required=True, default=0)
    parser.add_argument("--scale_intensity_bmax", type=float, required=True, default=1)
    parser.add_argument(
        "--scale_intensity_clip", type=bool, required=True, default=True
    )
    parser.add_argument(
        "--orientation_axcode", type=str, required=True, default="RAS", choices=["RAS"],help='This is the orientation of the MRI/CT. Careful when selecting'
    ) 
    add_rand_crop_params(parser)
    add_rand_flip_params(parser)
    add_rand_affine_params(parser)
    add_rand_gauss_params(parser)
    add_rand_shift_params(parser)

    return parser

def get_params(): 
    parser  = build_args()
    args = parser.parse_args()
def add_rand_crop_params(parser): 
    parser.add_argument(
        "--rand_crop_label_num_samples",
        required=True,
        type=float,
        help="Each Image is cropped into patches. How many random patches should we get for each image. Note batch will be NumberImages*NumberSamples"
    )
    parser.add_argument(
        "--rand_crop_label_positive_samples",
        required=True,
        type=float, 
    )
    parser.add_argument(
        "--rand_crop_label_allow_smaller",
        required=True,
        type=bool, 
    )
def add_rand_shift_params(parser):
    parser.add_argument(
        "--rand_shift_intensity_offset",
        required=True,
        type=float, 
    )
    parser.add_argument(
        "--rand_shift_intensity_prob",
        required=True,
        type=float, 
    )
def add_rand_gauss_params(parser): 
    parser.add_argument(
        '--rand_gauss_sigma',
        required = True,
        type = list,
    )
def add_rand_flip_params(parser:argparse.ArgumentParser): 
    parser.add_argument( 
        '--rand_flip_prob',
        required=True, 
        type=bool 
    )
def add_rand_affine_params(parser:argparse.ArgumentParser): 
    parser.add_argument(
        '--rand_affine_prob',
        required= True , 
        type = float
    )
    parser.add_argument(
        '--rand_affine_rotation_range',
        required=True ,
        type = list 
    )
    parser.add_argument(
        '--rand_affine_scale_range',
        required=True , 
        type = list 
    )


    
if __name__=='__main__': 
    args = build_args()
    args.parse_args()