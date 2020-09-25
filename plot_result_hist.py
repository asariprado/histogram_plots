#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import sys
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sb


#Add pyplot histogram function

def pairwisescat(cfit):
    """Plots pairwise histograms and scatterplots using seaborn."""
    
    df = pd.read_table(cfit, header=None)
    df.index = ['Fitness Metric', 'Mass Frac Y', 'Mass Frac I1', 'Mass Frac I2', 'Mass Frac O']
    df = df.T
    sb.set_style("darkgrid")

    pairplots = sb.PairGrid(df)#,  hue='Extinction (A_v)',palette='husl')#, hue='Extinction (A_v)', height = 3, aspect = 3, layout_pad=2)
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    pairplots.map_diag(sb.histplot)
    pairplots.map_offdiag(sb.scatterplot)
    pairplots.add_legend()
    plt.show()

#     plt.tight_layout()

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Incorrect number of arguments. Please provide a .cfit file name and plot type (pair or none).")
        sys.exit(1)

    filename = sys.argv[1]
    plot_type = sys.argv[2]
    
    if plot_type == 'pair':
        pairwisescat(filename)
    else:
        pass
        #func(filename)
