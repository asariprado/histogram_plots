import sys
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy import stats
import time
import os #.path 

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
        NOTE: cfitfile must include the full path in order to find normalization files.
    """
    #  extracting the cfit file name prefix 
    file_prefix = cfitfile.split('_model_fits.cfit')[0]

    # assigning the prefix to new files to import the normalization files corresponding to our cfit file
    gal_norm_file = file_prefix+'_off_axis_normal_fact.dat'
    model_norm_file = file_prefix+'_model_normal_fact.dat'
    masstable_file = file_prefix+'_masstable.tab'
    

    #GREG: Try np.genfromtxt and retry masking using something like
    # met_mask = all_model_norms[0,:] == solar_met #Takes first column and selects metallicity = 0.02
    # N_i = all_model_norms[met_mask] 
    G = np.genfromtxt(gal_norm_file)
    all_model_norms = np.genfromtxt(model_norm_file)

    # Only select the normalization factors corresponding to solar metallicities ~ 0.2 
    solar_met = 0.02
    met_mask = (all_model_norms.transpose()[0,:] == solar_met) #Takes first column and selects metallicity = 0.02
     
    N_i = all_model_norms[met_mask].T[-1]
    
    print(N_i)
    # Stellar Mass Formed for each population:  a_i = b_i*G/N_i , where b_i are the cfit values
    normalization_factors = G/N_i
    
    print(normalization_factors)
    #Grab filename and full path
    return normalization_factors
    
    

def seaborn_pairwise(data, columns=None, output_name=None, output_dir='./', normalized=None, mass_frac=[]):
    
    """Plots pairwise scatterplot and histograms using seaborn, with the option 
        to plot normalized data with stellar mass fraction if 'normalized'=True, 
        otherwise the raw data will be plotted."""

    ##GREG: can probably revert this function back to NOT include the mass fraction, etc calculations
    # and let it be more general. We will want to check that our filename saving in new folder works.
    # I've taken out the datfile keyword and replaced it with output_dir to tell what directory to save file to
    # and I've set the default to save in whatever the current directory is and a output_name keyword
    # to capture what we want to call the output name
    
      
    df = pandas.DataFrame(data=data, columns=columns)
    
    # Added keyword parameter 'normalized' with default value None - if True, plot with mass fraction, else just plot data  
    if normalized != None: #  a_i = b_i*normalization_factors
        a_i = df * mass_frac  # a_i = b_i * G / N_i  // these are the values in masstable.tab
        df = a_i.div(np.sum(a_i, axis=1),axis=0) # mass fraction ~  a_i / sum(a_i)
        
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
        
    #Setting a title to the entire figure based on filename  
    g.fig.suptitle(output_name+'_pairwise_hist')
    
    # Set the column entry as the title for each axes 
    for ax, title in zip(g.axes.flat, columns):
        ax.set(title=title) 
        
    #Plot
    plt.tight_layout()
    plt.show()
   

    #Save figure as a PDF with the same file name
    #GREG: I updated this to include path to output
    pdfname = output_dir+output_name+'_pairwise_hist.pdf'
    print("Output PDF will be",pdfname)
#     pdfname = output_name+'_pairwise_hist.pdf'
    
    #Save this pdf file in a results_*_plots folder inside the current directory
    #Check if dir exists if not make it
    #Make directories for file if they don't exist
    os.makedirs(os.path.dirname(pdfname), exist_ok=True)
    #GREG: I uncommented this since we need to to automatically create the path to where we want the
    #new file to go.
#     if not os.path.exists(pdfname):
#         os.makedirs(pdfname)
 
    # FIX :  OSError: [Errno 30] Read-only file system: '/results_pairwise_plots'

    g.savefig(pdfname)
    
    

def plot_stellar_mass_fractions(cfitfile, nssps=4):
    """Read in model cfits file, plots with mass fraction conversion and save pairwise histograms."""

    #Grab filename and full path
    filename = cfitfile.split('/')[-1]
    #Define output name
    output_name = filename.split('.cfit')[0]+'_mass_fraction'
    #Use full path given in file if use_fullpath is True otherwise save in current directory
    fullpath = "/".join(cfitfile.split('/')[0:-1])
    #Let us know if we're not using fullpath
    if fullpath == '':
        print("PLease include full path to cfitfile. Exiting.")
        sys.exit(1)
    #Define output directory
    output_dir = fullpath+'/results_pairwise_plots/'

    #Reading in b_i's
    #NOTE: Excluding first row which is  metric
    #NOTE: Taking transpose for pandas
    data = np.genfromtxt(cfitfile).T[:,1:]
    #Setting column names 
    col_names = ['Y','I1','I2', 'O']

    
    #Convert to stellar mass fraction
    #This mass conversion function expects a cfitfilename that is a full path in order to find normalization files
    normalization_factors = mass_fraction_conversion(cfitfile)
    data = np.genfromtxt(cfitfile).T[:,1:]
    
    #Grab the SSP coefficients
    b_i = data[:,0:nssps]
    #Convert to solar masses
    a_i = b_i*normalization_factors
    #Convert to mass fractions
    massfrac_data = np.asarray([a_i[i]/np.sum(a_i,axis=1)[i] for i in range(a_i.shape[0])])
    # ^^^ I removed this conversion from this function bc it passed mass_fracdata to seaborn_pairwise
    # as an np.array and I had to convert it to a df again in seaborn func, thought it would
    # be better to just convert the data to df once in the seaborn func.
    #GREG: Here you don't need to convert to a pandas data frame, you can work with the data as numpy array
    #I use a list comprehension here to do the ssp divison by the sum as there was a brodcasting error in just doing the plain divison
    #Numpy doesn't let you divide something that has a 2-d shape 300,4 by something that's 1 d 300,
    
    
    #Plot
    #Plot pairwise histograms and save them to output directory
    #GREG: Since we're doing the conversion here now, I switch the normalized keyword to None
    seaborn_pairwise(data=massfrac_data, columns=col_names, output_name=output_name,output_dir=output_dir, normalized=None,mass_frac=normalization_factors)


def plot_cfit_data(cfitfile):
    """Read in cfit file and plot."""
    
    #Grab filename and full path
    filename = cfitfile.split('/')[-1]
    #Define output name
    output_name = filename.split('.cfit')[0]
    fullpath = "/".join(filename.split('/')[0:-1])
    # Don't define output directory -- just save in current directory
    
    #Reading in b_i's
    data = np.genfromtxt(cfitfile).T[:,1:]
   
    #Setting column names 
    col_names = ['Y','I1','I2', 'O']
    
    data = np.genfromtxt(cfitfile).T[:,1:]
    
    #Plot
    seaborn_pairwise(data, columns=col_names, output_name=output_name)

    
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not enough arguments. Please supply data file name. Exiting")
        sys.exit(1)

    #Read file name
    datfile = sys.argv[1]
   
    #Old original plots
    #Plot pairwise histogram for cfit data
    #plot_cfit_data(datfile) 

    #New stellar mass fraction plots
    #Plot pairwise histograms of stellar mass fractions from cfit data
    plot_stellar_mass_fractions(datfile)

    
