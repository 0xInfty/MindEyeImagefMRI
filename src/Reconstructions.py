# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

windows = True # Set to false if running in Ubuntu

import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, VersatileDiffusionPriorNetwork

import pyvdirs.dirs as dirs

if utils.is_interactive():
    %load_ext autoreload
    %autoreload 2

seed=42
utils.seed_everything(seed=seed)

# %% [markdown]
# # Configurations

# %%
# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # Example use
    jupyter_args = f"--data_path={os.path.join(dirs.DATA_HOME, 'MindEye')} \
                    --subj=1 \
                    --model_name=prior_257_final_subj01_bimixco_softclip_byol \
                    --vd_cache_dir={os.path.join(dirs.MODELS_HOME,'versatile_diffusion_cache')}"
    
    jupyter_args = jupyter_args.split()
    print(jupyter_args)

# %%
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of trained model",
)
parser.add_argument(
    "--autoencoder_name", type=str, default="None",
    help="name of trained autoencoder model",
)
parser.add_argument(
    "--data_path", type=str, default="/fMRI-reconstruction-NSD/src/datasets",
    help="Path to where NSD data is stored (see README)",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,5,7],
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
    help="How much img2img (1=no img2img; 0=outputting the low-level image itself)",
)
parser.add_argument(
    "--recons_per_sample", type=int, default=1,
    help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='/fMRI-reconstruction-NSD/train_logs',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()
    
if args.autoencoder_name=="None":
    args.autoencoder_name = None

# %%
if args.subj == 1:
    num_voxels = 15724
elif args.subj == 2:
    num_voxels = 14278
elif args.subj == 3:
    num_voxels = 15226
elif args.subj == 4:
    num_voxels = 13153
elif args.subj == 5:
    num_voxels = 13039
elif args.subj == 6:
    num_voxels = 17907
elif args.subj == 7:
    num_voxels = 12682
elif args.subj == 8:
    num_voxels = 14386
print("subj", args.subj, "num_voxels", num_voxels)

# %% [markdown]
# ## Dataset

# %%
if windows: val_url = (f"file:{args.data_path}/test_subj0{args.subj}_"+"{0..1}.tar").replace(os.path.sep, '/')
else: val_url = os.path.join(args.data_path, f"test_subj0{args.subj}_"+"{0..1}.tar")
meta_url = os.path.join( args.data_path, f"metadata_subj0{args.subj}.json" )

num_train = 8559 + 300
num_val = 982
batch_size = val_batch_size = 1
voxels_key = 'nsdgeneral.npy' # 1d inputs

val_data = wds.WebDataset(val_url, resampled=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(val_batch_size, partial=False)

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# Check that your data loader is working
for val_i, (voxel, img_input, coco) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape)
    print("img_input.shape",img_input.shape)
    break

# %% [markdown]
# ## Load autoencoder

# %%
from models import Voxel2StableDiffusionModel

outdir = f"{args.data_path}/{args.autoencoder_name}"
ckpt_path = os.path.join(outdir, f'epoch120.pth')

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels)

    voxel2sd.load_state_dict(state_dict,strict=False)
    voxel2sd.eval()
    voxel2sd.to(device)
    print("Loaded low-level model!")
else:
    print("No valid path for low-level model specified; not using img2img!") 
    img2img_strength = 1

# %% [markdown]
# # Load VD pipe

# %%
print('Creating versatile diffusion reconstruction pipeline...')
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
try:
    # vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device).to(torch.float16)
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
        "fMRI-reconstruction-NSD/train_logs/prior_257_final_subj01_bimixco_softclip_byol",
        cache_dir=args.vd_cache_dir).to(device).to(torch.float16)
except:
    print("Downloading Versatile Diffusion to", args.vd_cache_dir)
    vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            cache_dir = args.vd_cache_dir).to(device).to(torch.float16)
vd_pipe.image_unet.eval()
vd_pipe.vae.eval()
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

# vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir, subfolder="scheduler")

num_inference_steps = 20

# Set weighting of Dual-Guidance 
text_image_ratio = .0 # .5 means equally weight text and image, 0 means use only image
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = text_image_ratio
        for i, type in enumerate(("text", "image")):
            if type == "text":
                module.condition_lengths[i] = 77
                module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
            else:
                module.condition_lengths[i] = 257
                module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

unet = vd_pipe.image_unet
vae = vd_pipe.vae
noise_scheduler = vd_pipe.scheduler

# %% [markdown]
# ## Load Versatile Diffusion model

# %%
img_variations = False

out_dim = 257 * 768
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
voxel2clip.requires_grad_(False)
voxel2clip.eval()

out_dim = 768
depth = 6
dim_head = 64
heads = 12 # heads * dim_head = 12 * 64 = 768
timesteps = 100 #100

prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        learned_query_mode="pos_emb"
    )

diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
)

outdir = os.path.join(args.data_path, args.model_name)
ckpt_path = os.path.join(outdir, "last.pth")

checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']
print("EPOCH: ",checkpoint['epoch'])
diffusion_prior.load_state_dict(state_dict,strict=False)
diffusion_prior.eval().to(device)
diffusion_priors = [diffusion_prior]
pass

# %% [markdown]
# # Reconstruct one-at-a-time

# %%
# Setting...
# plotting = True
# ind_include = [0]
# saving = False

# %%
vd_pipe = vd_pipe.to(torch.float32)
unet = unet.to(torch.float32)
vae = vae.to(torch.float32)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

retrieve = False
plotting = False
saving = True
verbose = False
imsize = 512

if img_variations:
    guidance_scale = 7.5
else:
    guidance_scale = 3.5
    
ind_include = list(range(num_val))
for i in [93, 107]: ind_include.remove(i)
all_brain_recons = None
    
only_lowlevel = False
if img2img_strength == 1:
    img2img = False
elif img2img_strength == 0:
    img2img = True
    only_lowlevel = True
else:
    img2img = True
    
for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=len(ind_include))):
    if val_i<np.min(ind_include):
        continue
    voxel = torch.mean(voxel,axis=1).to(device)
    # voxel = voxel[:,0].to(device)
    
    with torch.no_grad():
        if img2img:
            ae_preds = voxel2sd(voxel.float())
            blurry_recons = vd_pipe.vae.decode(ae_preds.to(device).half()/0.18215).sample / 2 + 0.5

            if val_i==0:
                plt.imshow(utils.torch_to_Image(blurry_recons))
                plt.show()
        else:
            blurry_recons = None

        if only_lowlevel:
            brain_recons = blurry_recons
        else:
            grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                voxel2clip_cls = None, #diffusion_prior_cls.voxel2clip,
                diffusion_priors = diffusion_priors,
                text_token = None,
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_sample = args.recons_per_sample,
                guidance_scale = guidance_scale,
                img2img_strength = img2img_strength, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps_prior = 100,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                img_variations = img_variations,
                verbose = verbose,
            )

            if plotting:
                plt.show()
                # grid.savefig(f'evals/{model_name}_{val_i}.png')

            brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

        if all_brain_recons is None:
            all_brain_recons = brain_recons
            all_images = img
        else:
            all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
            all_images = torch.vstack((all_images,img))

    if val_i>=np.max(ind_include):
        break

all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if saving:
    torch.save(all_images, os.path.join(dirs.RESULTS_HOME, 'all_images.pt'))
    torch.save(all_brain_recons, 
               os.path.join(dirs.RESULTS_HOME, 
                            f'{args.model_name}_recons_img2img{img2img_strength}_{args.recons_per_sample}samples.pt'))

if not utils.is_interactive():
    sys.exit(0)


