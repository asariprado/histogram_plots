#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import sys
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sb


def plot_hist(cfit):
        
    """Plots histograms for a pandas DataFrame using pyplot."""

    df = pd.read_table(cfit, header=None) 
    df.index = ['Fitness Metric', 'Mass Frac Y', 'Mass Frac I1', 'Mass Frac I2', 'Mass Frac O']
    
    axs = plt.figure(figsize=(6*2.5,5*2)).subplots(2, 3, sharey=True)
    axs = trim_axs(axs,len(df.index))
        
    for i in range(len(df.index)):
        axs.flat[i].hist(df.iloc[i],edgecolor='white', linewidth=1)
        axs.flat[i].set_title(df.index[i],weight='bold', size=14)
        axs[i].tick_params(axis='x', labelrotation=30)
        plt.xticks(ha='right')
        axs.flat[i].ticklabel_format(scilimits=(-3,0))
    
    plt.show()
        

def trim_axs(axs, N):
        """
        Reduce *axs* to *N* Axes. All further Axes are removed from the figure. 
        (removes an axis so that way there wont be a plot at that entry..blank space)
        """
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()

        return axs[:N]


def pairwisescat(cfit):
    
    """Plots pairwise histograms and scatterplots using seaborn."""
    
    df = pd.read_table(cfit, header=None)
    df.index = ['Fitness Metric', 'Mass Frac Y', 'Mass Frac I1', 'Mass Frac I2', 'Mass Frac O']
    df = df.T
    sb.set_style("darkgrid")

    pairplots = sb.PairGrid(df)
#     plt.gcf().subplots_adjust(wspace=0, hspace=0)
#     pairplots.fig.set_size_inches(8,8)
    pairplots.map_diag(sb.histplot)
    pairplots.map_offdiag(sb.scatterplot)
    pairplots.add_legend()
    plt.show()



if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Incorrect number of arguments. Please provide a .cfit file name and plot type (pair or none).")
        sys.exit(1)

    filename = sys.argv[1]
    plot_type = sys.argv[2]
    
    if plot_type == 'pair':
        plot_hist(filename)
        pairwisescat(filename)
    else:
        # else if not 'pair', just plot the histograms
        plot_hist(filename)
        

