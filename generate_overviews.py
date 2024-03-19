import os
import numpy as np
from datetime import date
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import seaborn as sns
sns.set_theme(style="whitegrid")

from mouse_dataset import mouse_dataset

from utils.utils import plot_item_to, get_file_list, make_filepath


def save_overview(items=[{}], cols=5, rows=8, save_to='overview/labeled', fname=None, title=None, add_meta=False, add_source_file=False, sort_by=None, keys=['Sxx']):
    if sort_by is not None:
        items = sorted(items, key=lambda d: d[sort_by])
    
    axs_p_fig = cols * rows
    nfigs = int(np.ceil(len(items)/axs_p_fig))
    figs = [plt.figure(figsize=(20, 25), layout='constrained') for _ in range(nfigs)]
    
    if fname is None:
        fname = os.path.basename(items[0]['csv_file']).split('.')[0]
    
    if title is None:
        title = items[0]['csv_file'].replace('\\', '/')

    subfigs = [F.subfigures(rows, cols).ravel() for F in figs]
    
    for n, item in enumerate(items):
        nf = int(n / axs_p_fig)
        # subfig = figs[nf].add_subfigure(rows, cols, (n % axs_p_fig) + 1)
        subfig = subfigs[nf][n % axs_p_fig]
        # ax = figs[nf].add_subplot(rows, cols, (n % axs_p_fig) + 1)
        for j, key in enumerate(keys):
            ax = subfig.add_subplot(len(keys), 1, j+1)
            plot_item_to(ax, item, add_meta=add_meta, add_source_file=add_source_file, print_title=(j==0), key=key)
            ax.title.set_size(14)
            ax.title.set_weight('bold')
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
            ax.tick_params('both', labelsize=6)
    
    for nf in range(len(figs)):
        figs[nf].suptitle( title + f'   -   {nf+1:02}/{len(figs):02}   -   created {date.today()}', fontsize=18, fontweight='bold')
        for sfig in subfigs[nf]:
            sfig.set_edgecolor('#D4D3DB')
            sfig.set_frameon(True)
            sfig.set_linewidth(0.1)

    if save_to is not None:
        for nf in range(len(figs)):
            _fname = fname + f'-{nf+1:02}'
            path = make_filepath(os.path.join(save_to, _fname), '.png')
            figs[nf].savefig(path, dpi=300)
            plt.close(figs[nf])
    
    return figs


def save_dataset_overview(D: mouse_dataset, cols=5, rows=8, save_to='overview/labeled', title=None):
    items = [D.getsorted(i) for i in range(len(D))]
    
    fname = os.path.basename(D.csv_files[0]).split('.')[0]
    
    if title is None:
        title = D.csv_files[0].replace('\\', '/')
    
    save_overview(items, cols=cols, rows=rows, save_to=save_to, fname=fname, title=title)





if __name__ == '__main__':
    # folder = 'data/labeled/'
    # save_to = 'overview/labeled'
    folder = 'data/automatic_detection_manual_classification/'
    save_to = 'overview/automatic_detection_manual_classification_new'
    
    
    wavs = get_file_list(folder, '.WAV')
    csvs = get_file_list(folder, '.csv')
    

    datasets = []
    for W, C, in zip(wavs, csvs):
        datasets.append(mouse_dataset([W], [C], pad_start_ms=10, pad_end_ms=10, skip_pickle=True))
    
    print('Generating overviews...')
    for D in tqdm(datasets):
        # print(f'[{n+1:02}/{len(datasets):02}] in progress')
        # save_dataset_overview(D, save_to=save_to)
        items = [I for I in D]
        fname = os.path.basename(D.csv_files[0]).split('.')[0]
        title = D.csv_files[0].replace('\\', '/')
        save_overview(items, cols=6, rows=4, save_to=save_to, fname=fname, title=title, sort_by='real_category', keys=['Sxx', 'Sxx_clean', 'Sxx_small'])