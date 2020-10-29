import sys
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy import stats
import time
import os.path 

def corrfunc(x, y, **kws):
    """Plots value of Pearson coeff on plot."""
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(r"$\rho$ = {:.2f}".format(r),xy=(.5, .9), xycoords=ax.transAxes)

def set_lim(x, y, **kws):
    """Sets x and y limits based on per value."""
    per = kws['per']
    min_per = 50 - per/2
    max_per = per/2 + 50
    xper = np.nanpercentile(x,[min_per,max_per])
    yper = np.nanpercentile(y,[min_per,max_per])
    ax = plt.gca()
    ax.set_xlim(xper)
    ax.set_ylim(yper)

def show_stats(x, **kws):
    """Plots basis statistics on plot"""
    mean = np.mean(x)
    median = np.median(x)
    std = np.std(x,ddof=1)
    ax = plt.gca()
    ax.annotate("Mean: {:.2f}\nMedian: {:.2f}\n$\sigma$: {:.3e}".format(mean,median,std), xy=(.6,.3),xycoords=ax.transAxes, fontsize=9)


    
    
def mass_fraction_conversion(cfitfile):
    """ Returns conversion coefficient to multiply b_i (cfit values) by to obtain mass fraction // stellar mass formed per population
    """
    #  extracting the cfit file name prefix 
    file_prefix = cfitfile.split('_model_fits.cfit')[0]

    # assigning the prefix to new files to import the files corresponding to our cfit file
    gal_norm_file = file_prefix+'_off_axis_normal_fact.dat'
    model_norm_file = file_prefix+'_model_normal_fact.dat'
    masstable_file = file_prefix+'_masstable.tab'


    #   reading in the .off_axis_norm_factors.dat & model_norm_factors.dat files 
    #   with the same prefix name as the cfit file
    G_i = np.loadtxt(gal_norm_file)
    model_normalizations = np.loadtxt(model_norm_file)
    stellar_mass = np.loadtxt(masstable_file)
    
    
    # Only select the normalization factors corresponding to solar metallicities ~ 0.2 
    solar_met = 0.02

    N_i = []
    for col in model_normalizations:
        if col[0] == solar_met:
            N_i.append(col[2])
    
    stellar_mass_formed = []
    total_stellar_mass = []
    for col in stellar_mass:
        if col[0] == solar_met:
            stellar_mass_formed.append(col[2]) # == a_i
            total_stellar_mass.append(col[2] + col[3])
    
    stellar_mass_formed_fraction = np.array(stellar_mass_formed) # / np.array(total_stellar_mass
    
    # Stellar Mass Formed for each population:
    # a_i = b_i*G/N_i , where b_i are the cfit values

    normalization_factors = G_i/N_i
    
    return (normalization_factors, stellar_mass_formed_fraction)
    
    
    
    
    
def seaborn_pairwise(data, columns=None):
    
    """Plots pairwise scatterplot and histograms using seaborn"""

# #     Import data as a dataframe, name the columns
    b_i = pandas.DataFrame(data=data, columns=columns)

    normalization_factors, stellar_mass_formed_fraction = mass_fraction_conversion(cfitfile)[0], mass_fraction_conversion(cfitfile)[1]
    
    # a_i = b_i*G/N_i
    a_i = b_i*normalization_factors

#   # df as: mass fraction = a_i / sum(a_i)
    df = a_i / stellar_mass_formed_fraction
#     df = a_i / np.sum(a_i)

#     df = pandas.DataFrame(data=data, columns=columns)

    
    #Find best params
    #First column (fitness metric) is minimal
    best_x = df[df.iloc[:,0] == df.iloc[:,0].min()].iloc[0,:]
    
    #Creates PairGrid object for flexibility in controlling what gets plotted 
    g = sns.PairGrid(df, diag_sharey=False)
    #Plots pairwise scatter plots on the lower triangle of plot
    g.map_lower(sns.scatterplot)
    #Plots kernel density estimation of pairwise scatter plots on upper triangle on plot
    g.map_upper(sns.kdeplot)#, bw_method=0.3)
    #Calculate the Pearson correlation in each pairwise scatterplot and add the value in text to plots
    #in the upper triangle
    g.map_upper(corrfunc)
    #Plot a smoothed histogram on the diagonal for each data column
    g.map_diag(sns.kdeplot, lw=2)#, bw_method=0.3)
    #Calculate some statistics of the column data and add to diagonal plots
    g.map_diag(show_stats)
    
    #For each of the diagonal plots add vertical lines denoting the best value that was found
    for i in range(len(g.diag_axes.flat)):
        ax = g.diag_axes.flat[i]
#         ax.axvline(p[i],color='red',ls='--', label='Input')
        ax.axvline(best_x[i],color='green',ls='-', label='Best')
        ax.legend(frameon=False, loc=0) 
        
    #To get the name of the cfit file, removing the '.cfit' extension from the name
    filename = (str(datfile).rsplit('.', 1)[0]) 
    
    #Setting a title to the entire figure based on filename (for now) - change this to have a diff title? 
    g.fig.suptitle(filename+'_pairwise_hist')
    

    # Set the column entry as the title for each axes 
    for ax, title in zip(g.axes.flat, columns):
        ax.set(title=title) 
        
    #Plot
    plt.tight_layout()
    plt.show()
    
    #Save figure as a PDF with the same file name
    pdfname = filename + '_pairwise_hist.pdf'
    
#   Save this pdf file in a results_*_plots folder inside the current directory
    ## update to obtain current directory name to save plots to a folder : results_***stelib****_pairwise_plots 
    
    g.savefig(os.path.join('results_pairwise_plots', pdfname))
#     g.savefig(pdfname)
   
    
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not enough arguments. Please supply data file name. Exiting")
        sys.exit(1)

    #Read file name
    datfile = sys.argv[1]
   
    #Grab data with pandas and transpose so each row represents parameter values
    #NOTE 1: Here I take the log10 of the data since I know there are some small numbers and
    #scale it by 100 too. This just ensures we have larger values that are easier to compare
    #NOTE 2: Here I also exclude the fitness metrix cause it turns out they are all basically the same
    #data = 1e2*np.log10(np.genfromtxt(datfile).T[:,1:])
    
    data = np.genfromtxt(datfile).T[:,1:]
    cfitfile = datfile
    #Setting column names 
    col_names = ['Y','I1','I2', 'O']
    
    #Plot
    seaborn_pairwise(data, columns=col_names)
