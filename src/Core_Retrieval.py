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

import pyvtools.dirs as dirs
import pyvtools.image as vim

import utils
from models import Clipper, BrainNetwork

%load_ext autoreload
%autoreload 2

# %% [markdown]
# ### Functions

# %%
def plot_brain_signal(brain_signal, title=None, dark=True, ax=None):

    # Expect brain signal with shape (n_reps, n_voxels)

    try:
        n_reps, n_voxels = brain_signal.shape
    except:
        n_reps = 1
        n_voxels = len(brain_signal)
        brain_signal = np.expand_dims(brain_signal, axis=0)

    if ax is None: 
        fig, ax = plt.subplots(gridspec_kw=dict(left=0, right=1, top=1, bottom=0))
    else: fig = ax.get_figure()

    for i in range(n_reps):
        ax.plot(brain_signal[i], linewidth=1, alpha=0.5)
    ax.axhline(0, linewidth=1, color="black")

    if title is not None: 
        if dark: ax.set_title(title, fontsize="small", color="w")
        else: ax.set_title(title, fontsize="small")

    if dark:
        ax.set_xlabel("Voxel", color="w"); ax.set_ylabel("Amplitude", color="w")
        fig.patch.set_facecolor('k')
        [t.set_color('w') for t in ax.xaxis.get_ticklabels()]
        [t.set_color('w') for t in ax.yaxis.get_ticklabels()]
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
    else:
        ax.set_xlabel("Voxel"); ax.set_ylabel("Amplitude")

def plot_brain_signals(*brain_signals, labels:str|list[str]=None, title:str=None,
                        dark=True, **kwargs):
    """Plots several brain signal samples

    Parameters
    ----------
    brain_signals : list or tuple of np.ndarray
        Either unidimensional arrays of shape (n_voxels,) or 
        bidimensional arrays of shape (n_repetitions, n_voxels).
    labels : list of str, optional
        Image titles. Defaults present no title.
    dark : bool, optional
        Whether to use a black figure background or a white one. 
        Default is True, to produce a black background.
    dpi : int, optional
        Dots per inch. Default is 200.
    **kwargs : dict, optional
        Accepts Matplotlib's `imshow` kwargs.
    """

    if labels is None: labels = [None]*len(brain_signals)

    if title is not None: top = 1.01
    else: top = 1

    fig, axes = plt.subplots(ncols=len(brain_signals), squeeze=False, 
                             gridspec_kw=dict(left=0, right=1, top=top, bottom=0),
                             figsize=(2*len(brain_signals), 2))
    
    for k, brain_signal in enumerate(brain_signals):
        plot_brain_signal(brain_signal, labels[k], ax=axes[0][k], dark=dark, **kwargs)

    if title is not None:
        if dark: plt.suptitle(title, fontsize="medium", color="white", y=0.98)
        else: plt.suptitle(title, fontsize="medium", y=0.98)
    
    for i in range(len(brain_signals)):
        if i > 0: axes[0][i].set_ylabel(None)
    
    return

def plot_brain_signals_grid(*brain_signals_grid:list[np.ndarray]|np.ndarray, 
                            columns_labels:list[str]=None, 
                            rows_labels:list[str]=None, 
                            rows_info:dict[str, list|np.ndarray]=None, 
                            dark=True):
    """Plots a grid of images

    Parameters
    ----------
    brain_signals : list or tuple of np.ndarray
        Either unidimensional arrays of shape (n_voxels,) or 
        bidimensional arrays of shape (n_repetitions, n_voxels).
    columns_labels : list of str, optional
        Column titles. Defaults present no title.
    rows_labels : list of str, optional
        Rows titles. Defaults present no title.
    rows_info : dict of lists or np.ndarrays
        Additional row information. Dictionary keys could be metric labels 
        such as "MSE" or "SSIM" in case the value iterables contain the 
        metric associated to each row.
    dark : bool, optional
        Whether to use a black figure background or a white one. 
        Default is True, to produce a black background.
    dpi : int, optional
        Dots per inch. Default is 200.
    **kwargs : dict, optional
        Accepts Matplotlib's `imshow` kwargs.
    """

    if not isinstance(brain_signals_grid)==np.ndarray:
        brain_signals_grid = np.array(brain_signals_grid)
    assert brain_signals_grid.dim >= 2, "Brain signals must be on a 2D grid"
    
    n_columns = len(brain_signals_grid)
    n_rows = len(brain_signals_grid[0])
    mid_column = int(np.floor(n_columns/2))

    if columns_labels is None: 
        columns_labels = [None]*len(n_columns)

    if rows_labels is not None:
        labels = [[lab+lab_2 for lab_2 in rows_labels] for lab in columns_labels]
    else:
        labels = [[lab]+[None]*(n_rows-1) for lab in columns_labels]
    
    sec_labels = []
    if rows_info!={}:
        for i in range(n_rows):
            sec_labels.append([f"{k} {values[i]}" for k, values in rows_info.items()])
        if len(rows_info)>1:
            sec_labels = [" : "+", ".join(ls) for ls in sec_labels]
    if len(sec_labels)==0:
        sec_labels = [""]*n_rows

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2*n_columns, 2*n_rows), 
                             squeeze=False)
    for i in range(n_rows):
        for k, brain_signals in enumerate(brain_signals_grid):
            if k==mid_column: label = labels[k][i]+sec_labels[i]
            else: label = labels[k][i]
            plot_brain_signal(brain_signals[i], label, ax=axes[i][k], dark=dark)
    
    return

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

# %%
print(num_train*3)

# %%
num_test

# %% [markdown]
# ### General configuration

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# %%
seed = 42 # Random seed picked in the original code
utils.seed_everything(seed=seed)

# %% [markdown]
# ### Validation dataset

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

# %%
# Check that your data loader is working
for idx, (voxel, img_input, coco) in tqdm(enumerate(test_dataloader)):
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
ckpt_path = os.path.join(outdir, model_name, f'last.pth')

checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']
print("Checkpoint's number of epochs: ", checkpoint['epoch'])

# %%
voxel2clip.load_state_dict(state_dict, strict=False)
voxel2clip.eval().to(device)
del state_dict, checkpoint

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
            plot_brain_signals(*voxel.cpu().detach().numpy()[:3])

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
bwd_ci = stats.norm.interval(0.95, loc=perdcent_correct_bwd, scale=bwd_sd)

print(f"fwd percent_correct: {percent_correct_fwd:.4f} 95% CI: [{fwd_ci[0]:.4f},{fwd_ci[1]:.4f}]")
print(f"bwd percent_correct: {percent_correct_bwd:.4f} 95% CI: [{bwd_ci[0]:.4f},{bwd_ci[1]:.4f}]")


