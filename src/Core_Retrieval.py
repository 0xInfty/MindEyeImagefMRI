# %% [markdown]
# # Core Retrieval
# Try to isolate the CLIP image encoder and the brain encoder to run retrieval
# 
# Calculate similarity score with all elements in the test set for subject 1
# - Find the 10 images that best match a brain signal
# - Find the 10 brain signals that best match an image

# %% [markdown]
# ## What I understood from the code

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# - I'll restrict myself to subject 1
#     - Scotti et al only trained on 1, 2, 5, and 7 (the ones with data across all sessions)
#     - They used the 1k shared images as test set
#     - 2770 test samples were averaged over the 3 repetitions, resulting on 982 test samples
#     - They refer to the test set as val, apparently. Did they use it for validation during training?
# - I need to work with embeddings shaped $257\times 768$
#     - Picked by Scotti et al to match the size at the last hidden layer of CLIP ViT/L-14
#     - They should look like pictures. I'd love to plot them.
# - I assume `Clipper` is the OpenAI CLIP encoder: 
#     - For pre-trained models, it loads OpenAI's `"ViT-L/14"`
# - I assume `OpenClipper` is the OpenCLIP encoder
#     - It says `"THIS IS NOT WORKING CURRENTLY!"` on line 156, so maybe it's not fully functional x'D

# %% [markdown]
# ## Global definitions

# %% [markdown]
# ### Imports

# %%
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse

import pyvdirs.dirs as dirs
import pyvtools.image as vim

import utils
from models import Clipper, BrainNetwork
import my_utils as mutils

%load_ext autoreload
%autoreload 2

# %% [markdown]
# ### Parameters

# %%
# Individual in the 1 to 8 range
subj = 1 

# Voxels per individual, sorted from Nr 1 to Nr 8
voxels_per_individual = [15724, 14278, 15226, 13153, 13039, 17907, 12682, 14386]
voxels_key = 'nsdgeneral.npy' # 1d inputs

# Path to where NSD data is stored
# data_path = "/fsx/proj-medarc/fmri/natural-scenes-dataset"
data_path = dirs.DATA_HOME
# Code is supposed to download dataset, if it's not found in there

# Path to metadata for all subjects
metadir = data_path

# Path to where the models are stored
# outdir = f'../train_logs/{autoencoder_name}'
outdir = dirs.MODELS_HOME

# Name of 257x768 model, used for everything except LAION-5B retrieval
model_name = "prior_257_final_subj01_bimixco_softclip_byol"

# Name of 1x768 model, used for LAION-5B retrieval
model_name2 = "prior_1x768_final_subj01_bimixco_softclip_byol"

# Name of the low-level pipeline's autoencoder
autoencoder_name = None

# Distribution factor between high and low-level pipelines
# 0 outputs the low-level image, 1 outputs the high-level image instead
img2img_strength = 1 # Not using img2img with the low-level pipeline on top of the high-level pipeline

# Network embedding parameters
out_dim = 257 * 768

# Retrieval test parameters
batch_size = 300 # same as used in mind_reader
test_batch_size = 300 # number of samples randomly picked (including the correct one) to assess accuracy
test_loops = 30 # number of times to go through the entire test set

# How many recons to output, to then automatically pick the best one (MindEye uses 16)
recons_per_sample = 1

# %% [markdown]
# ### Definitions

# %%
num_voxels = voxels_per_individual[subj-1]
print("Subj", subj, "=> Num_voxels", num_voxels)

# %%
# test_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
test_url = f"{data_path}/test_subj0{subj}_" + "{0..1}.tar"
# meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
meta_url = f"{data_path}/webdataset_avg_split_metadata_subj0{subj}.json"

# %%
metadata = json.load(open(meta_url))
num_train = metadata['totals']['train']
num_val = metadata['totals']['val']
num_test = metadata['totals']['test']

# %% [markdown]
# What is `num_train`? There are supposedly 24980 training samplesm not 8859. Mmm... It's close to 24980/3, but it's not exactly the same.

# %% [markdown]
# ### General configuration

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# %%
seed = 42 # Random seed picked in the original code
utils.seed_everything(seed=seed)

# %% [markdown]
# ### Test dataset

# %%
test_data = wds.WebDataset(test_url, resampled=True)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(test_batch_size, partial=False)\
    .with_epoch(test_loops)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=None, shuffle=False)

# Check that your data loader is working
for idx, (voxel, img_input, coco) in enumerate(test_dataloader):
    print("IDx", idx)
    print("Voxel shape", voxel.shape)
    print("Input image shape", img_input.shape)
    break
del idx, voxel, img_input, coco

# %% [markdown]
# ### Common neural networks

# %% [markdown]
# #### CLIP image encoder

# %%
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)

# %% [markdown]
# #### Brain encoder

# %%
voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim, use_projector=True, device=device)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
voxel2clip.requires_grad_(False)
voxel2clip.eval()

# %%
ckpt_path = os.path.join(outdir, model_name, f'voxel2clip.pth')
# ckpt_path = os.path.join(outdir, model_name, f'last.pth')

checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model']
# state_dict = checkpoint['model_state_dict']
print("Checkpoint's number of epochs: ", checkpoint['epoch'])

# %%
p = next(voxel2clip.named_parameters())
p

# %%
voxel2clip.load_state_dict(state_dict, strict=False)
voxel2clip.eval().to(device)
del state_dict, checkpoint

# %%
p = next(voxel2clip.named_parameters())
p

# %% [markdown]
# ## Test set retrieval

# %%
percents_correct_forwards, percents_correct_backwards = [], []

with torch.no_grad():
    k = 0
    for idx, (voxel, img, coco) in enumerate(tqdm(test_dataloader, total=test_loops)):

        if idx<3 and k<2:
            print("Voxel's shape", voxel.shape)
            print("Image's shape", img.shape)
            vim.plot_images(*img[:3].cpu().detach().numpy().transpose(0,2,3,1))
            mutils.plot_brain_signals(*voxel.cpu().detach().numpy()[:3])

        voxel = torch.mean(voxel, axis=1).to(device) # average across repetitions
        # voxel = voxel[:,np.random.randint(3)].to(device) # random one of the single-trial samples

        emb_img = clip_extractor.embed_image(img.to(device)).float() # CLIP-Image
        
        _, emb_brain = voxel2clip(voxel.float()) # CLIP-Brain
        
        if idx<3 and k<2:
            print("Averaged voxel's shape", voxel.shape)
            print("Image embedding's shape", emb_img.shape)
            print("Brain embedding's shape", emb_brain.shape)
            vim.plot_images(*emb_img[:3].cpu().detach().numpy(), labels=["","Image embeddings",""])
            vim.plot_images(*emb_brain[:3].cpu().detach().numpy(), labels=["","Brain embeddings",""])
        
        # Flatten if necessary
        emb_img = emb_img.reshape(len(emb_img),-1)
        emb_brain = emb_brain.reshape(len(emb_brain),-1)
        
        # L2 normalization
        emb_img = nn.functional.normalize(emb_img, dim=-1)
        emb_brain = nn.functional.normalize(emb_brain, dim=-1)
        
        labels = torch.arange(len(emb_img)).to(device)
        similarity = utils.batchwise_cosine_similarity(emb_brain, emb_img)  # Brain, CLIP

        # similarity_backwards = utils.batchwise_cosine_similarity(emb_img, emb_brain)  # clip, brain
        # similarity_forwards = utils.batchwise_cosine_similarity(emb_brain, emb_img)  # brain, clip
        # print(np.any((similarity_backwards.cpu().detach().numpy().T != similarity_forwards.cpu().detach().numpy()).flatten()))

        if idx<3 and k<2:
            print("Similarity matrix's shape", similarity.shape)
            vim.plot_image(similarity.cpu().detach().numpy(), 
                            dpi=300, interpolation="none", colormap="magma")
        
        top1_forwards = utils.topk(similarity, labels, k=1).item()
        top1_backwards = utils.topk(torch.transpose(similarity, 0, 1), labels, k=1).item()
        print("Top-1 Accuracy per Brain Signal", top1_forwards*100, r"%")
        print("Top-1 Accuracy per Image", top1_backwards*100, r"%")
        
        percents_correct_forwards.append(top1_forwards)
        percents_correct_backwards.append(top1_backwards)
            
        k += 1
        # break

# %%
percent_correct_fwd = np.mean(percents_correct_forwards)
fwd_sd = np.std(percents_correct_forwards) / np.sqrt(len(percents_correct_forwards))
fwd_ci = stats.norm.interval(0.95, loc=percent_correct_fwd, scale=fwd_sd)

percent_correct_bwd = np.mean(percents_correct_backwards)
bwd_sd = np.std(percents_correct_backwards) / np.sqrt(len(percents_correct_backwards))
bwd_ci = stats.norm.interval(0.95, loc=percent_correct_bwd, scale=bwd_sd)

print(f"fwd percent_correct: {percent_correct_fwd:.4f} 95% CI: [{fwd_ci[0]:.4f},{fwd_ci[1]:.4f}]")
print(f"bwd percent_correct: {percent_correct_bwd:.4f} 95% CI: [{bwd_ci[0]:.4f},{bwd_ci[1]:.4f}]")


