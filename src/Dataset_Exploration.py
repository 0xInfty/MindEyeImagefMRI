# %% [markdown]
# # Dataset Exploration
# Having isolatad the CLIP image encoder and the brain encoder...
# 
# 1. Calculate image and brain embeddings for the dataset
# 1. Run PCA on the embeddings

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
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets, decomposition
import torch
import torch.nn as nn
from tqdm import tqdm
import webdataset as wds

import pyvtools.dirs as dirs
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
train_url = f"{data_path}/train_subj0{subj}_" + "{0..17}.tar"
val_url = f"{data_path}/val_subj0{subj}_" + "{0..0}.tar"
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
voxel2clip.load_state_dict(state_dict, strict=False)
voxel2clip.eval().to(device)
del state_dict, checkpoint

# %%
p = next(voxel2clip.named_parameters())
p

# %% [markdown]
# ## Test dataset

# %% [markdown]
# #### One by one

# %%
test_data = wds.WebDataset(test_url, shardshuffle=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

# Check that your data loader is working
for idx, (voxel, img_input, coco) in enumerate(test_dataloader):
    print("IDx", idx)
    print("Voxel shape", voxel.shape)
    print("Input image shape", img_input.shape)
    print("Coco IDX shape", coco.shape)
    break
# del idx, voxel, img_input, coco

# %%
vim.plot_image(img_input[0].cpu().detach().numpy().transpose(1,2,0))

# %% [markdown]
# #### Three at a time

# %%
test_data = wds.WebDataset(test_url, shardshuffle=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=3, shuffle=False)

# Check that your data loader is working
for idx, (voxel, img_input, coco) in enumerate(test_dataloader):
    print("IDx", idx)
    print("Voxel shape", voxel.shape)
    print("Input image shape", img_input.shape)
    print("Coco IDX shape", coco.shape)
    break
# del idx, voxel, img_input, coco

# %%
vim.plot_images(*img_input.cpu().detach().numpy().transpose(0,2,3,1))

# %% [markdown]
# ### Batches

# %%
test_data = wds.WebDataset(test_url, shardshuffle=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(test_batch_size, partial=True)
    # .with_epoch(test_loops)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=None, shuffle=False)

# Check that your data loader is working
for idx, (voxel, img_input, coco) in enumerate(test_dataloader):
    print("IDx", idx)
    print("Voxel shape", voxel.shape)
    print("Image shape", img_input.shape)
    print("Coco IDX shape", coco.shape)
del idx, voxel, img_input, coco

# %% [markdown]
# ## Test Embeddings

# %%
img_embeddings, brain_embeddings = [], []

with torch.no_grad():
    k = 0
    for idx, (voxel, img, coco) in enumerate(tqdm(test_dataloader)):

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
        
        # # L2 normalization
        # emb_img = nn.functional.normalize(emb_img, dim=-1)
        # emb_brain = nn.functional.normalize(emb_brain, dim=-1)
        
        img_embeddings += list(emb_img.cpu().detach().numpy())
        brain_embeddings += list(emb_brain.cpu().detach().numpy())
            
        k += 1
        # break

img_embeddings = np.array(img_embeddings)
brain_embeddings = np.array(brain_embeddings)
print("All image embedding's shape", img_embeddings.shape)
print("All brain embedding's shape", brain_embeddings.shape)

# %% [markdown]
# ## Test Principal Components Analysis (PCA)

# %% [markdown]
# ### Run without normalization

# %%
img_pca = decomposition.PCA(n_components=2)
img_pca.fit(img_embeddings)

img_pca_data = img_pca.transform(img_embeddings)
print("Image data PCA shape", img_pca_data.shape)

# %%
brain_pca = decomposition.PCA(n_components=2)
brain_pca.fit(brain_embeddings)

brain_pca_data = brain_pca.transform(brain_embeddings)
print("Brain data PCA shape", brain_pca_data.shape)

# %%
plt.scatter(*img_pca_data.T, alpha=.3)
plt.scatter(*brain_pca_data.T, alpha=.3)

# %% [markdown]
# ### Normalize after running

# %%
img_pca_data_norm = img_pca_data.T / np.mean(np.linalg.norm(img_pca_data, axis=1))
brain_pca_data_norm = brain_pca_data.T / np.mean(np.linalg.norm(brain_pca_data, axis=1))

# %%
plt.scatter(*img_pca_data_norm, alpha=.3)
plt.scatter(*brain_pca_data_norm, alpha=.3)

# %% [markdown]
# ### Normalize before running

# %%
img_embeddings_norm = img_embeddings / np.mean(np.linalg.norm(img_embeddings, axis=1))
brain_embeddings_norm = brain_embeddings / np.mean(np.linalg.norm(brain_embeddings, axis=1))

# %%
img_pca = decomposition.PCA(n_components=2)
img_pca.fit(img_embeddings_norm)

img_pca_data = img_pca.transform(img_embeddings_norm)
print("Image data PCA shape", img_pca_data.shape)

# %%
brain_pca = decomposition.PCA(n_components=2)
brain_pca.fit(brain_embeddings_norm)

brain_pca_data = brain_pca.transform(brain_embeddings_norm)
print("Brain data PCA shape", brain_pca_data.shape)

# %%
plt.scatter(*img_pca_data.T, alpha=.3)
plt.scatter(*brain_pca_data.T, alpha=.3)

# %%
out_dim * num_train * 32

# %% [markdown]
# ## Training dataset

# %%
train_data = wds.WebDataset(train_url, shardshuffle=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(test_batch_size, partial=True)
    # .with_epoch(test_loops)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=None, shuffle=False)

# Check that your data loader is working
for idx, (voxel, img_input, coco) in enumerate(train_dataloader):
    print("IDx", idx)
    print("Voxel shape", voxel.shape)
    print("Image shape", img_input.shape)
    print("Coco IDX shape", coco.shape)
    break
del idx, voxel, img_input, coco

# %% [markdown]
# ## Embeddings

# %%
img_embeddings, brain_embeddings = [], []

with torch.no_grad():
    k = 0
    for idx, (voxel, img, coco) in enumerate(tqdm(train_dataloader)):

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
        
        # # L2 normalization
        # emb_img = nn.functional.normalize(emb_img, dim=-1)
        # emb_brain = nn.functional.normalize(emb_brain, dim=-1)
        
        img_embeddings += list(emb_img.cpu().detach().numpy())
        brain_embeddings += list(emb_brain.cpu().detach().numpy())
            
        k += 1
        # break

img_embeddings = np.array(img_embeddings)
brain_embeddings = np.array(brain_embeddings)
print("All image embedding's shape", img_embeddings.shape)
print("All brain embedding's shape", brain_embeddings.shape)

# %% [markdown]
# ## Principal Components Analysis (PCA)

# %% [markdown]
# ### Run without normalization

# %%
img_pca = decomposition.PCA(n_components=2)
img_pca.fit(img_embeddings)

img_pca_data = img_pca.transform(img_embeddings)
print("Image data PCA shape", img_pca_data.shape)

# %%
brain_pca = decomposition.PCA(n_components=2)
brain_pca.fit(brain_embeddings)

brain_pca_data = brain_pca.transform(brain_embeddings)
print("Brain data PCA shape", brain_pca_data.shape)

# %%
plt.scatter(*img_pca_data.T, alpha=.05)
plt.scatter(*brain_pca_data.T, alpha=.05)

# %% [markdown]
# ### Normalize after running

# %%
img_pca_data_norm = img_pca_data.T / np.mean(np.linalg.norm(img_pca_data, axis=1))
brain_pca_data_norm = brain_pca_data.T / np.mean(np.linalg.norm(brain_pca_data, axis=1))

# %%
plt.scatter(*img_pca_data_norm, alpha=.05)
plt.scatter(*brain_pca_data_norm, alpha=.05)

# %% [markdown]
# ### Normalize before running

# %%
img_embeddings_norm = img_embeddings / np.mean(np.linalg.norm(img_embeddings, axis=1))
brain_embeddings_norm = brain_embeddings / np.mean(np.linalg.norm(brain_embeddings, axis=1))

# %%
img_pca = decomposition.PCA(n_components=2)
img_pca.fit(img_embeddings_norm)

img_pca_data = img_pca.transform(img_embeddings_norm)
print("Image data PCA shape", img_pca_data.shape)

# %%
brain_pca = decomposition.PCA(n_components=2)
brain_pca.fit(brain_embeddings_norm)

brain_pca_data = brain_pca.transform(brain_embeddings_norm)
print("Brain data PCA shape", brain_pca_data.shape)

# %%
plt.scatter(*img_pca_data.T, alpha=.05)
plt.scatter(*brain_pca_data.T, alpha=.05)


