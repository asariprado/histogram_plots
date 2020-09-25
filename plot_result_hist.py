#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sb




def pairwisescat(cfit):
    
    df = pd.read_table(cfit, header=None)
    df.index = ['Extinction (A_v)', 'Mass Frac Y', 'Mass Frac I1', 'Mass Frac I2', 'Mass Frac O']
    df = df.T
    sb.set_style("darkgrid")

    pairplots = sb.PairGrid(df,  hue='Extinction (A_v)',palette='husl')#, hue='Extinction (A_v)', height = 3, aspect = 3, layout_pad=2)
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    pairplots.map_diag(sb.histplot)
    pairplots.map_offdiag(sb.scatterplot)
    pairplots.add_legend()
    plt.show()

#     plt.tight_layout()
pairwisescat('tau_2gyr_newdfk_sn_020_ps_000_r000_model_fits.cfit')