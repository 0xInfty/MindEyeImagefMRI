import numpy as np
import matplotlib.pyplot as plt

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