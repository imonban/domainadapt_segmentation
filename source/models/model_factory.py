from monai.networks.nets.unet import UNet


def model_factory(config):
    model_name = config["model"]
    num_seg_labels = config["num_seg_labels"]
    if model_name == "unet":
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_seg_labels,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            act="LEAKYRELU",
        )
        return net
