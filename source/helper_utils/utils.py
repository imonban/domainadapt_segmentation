

def figure_version(path : str,load_past=False):
    #  when saving model  checkpoints and logs. Need to make sure i don't overwrite previous experiemtns
    avail = glob(f"{path}/run_*")
    if len(avail) == 0:
        ver = "run_0"
    else:
        avail = sorted(avail, key=lambda x: int(x.split("_")[-1]))
        oldest = int(avail[-1].split("_")[-1])
        if load_past: 
            ver = f"run_{oldest}"
        else: 
            ver = f"run_{oldest+1}"
    return os.path.join(path,ver)

def show_large_slice(input_dict): 
    #TODO: MAKE IT SO I CAN USE THIS IN TENSORBOARD LOGGING 
    lbl = input_dict['label']
    l_idx = (lbl[0]!=0).sum(dim=0).sum(dim=0).argmax()
    lbl_max = lbl[0,:,:,l_idx] 
    img = input_dict['image'][0,:,:,l_idx]
    plt.imshow(img,cmap='gray')
    plt.imshow(lbl_max,alpha=0.5)

def dice_score(truth,pred):
    #TODO USE THIS IN TEST PHASE OF FINAL MODEL 
    seg = pred.flatten() 
    gt = truth.flatten() 
    return  np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))