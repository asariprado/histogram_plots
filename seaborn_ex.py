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
    ax.annotate("Mean: {:.4f}\nMedian: {:.4f}\n$\sigma$: {:.4e}".format(mean,median,std), xy=(.6,.3),xycoords=ax.transAxes, fontsize=9)

    
def mass_fraction_conversion(cfitfile,nssps=4):
    """ Returns conversion coefficient to multiply b_i (cfit values) by to obtain stellar mass fraction.
        NOTE: cfitfile must include the full path in order to find normalization files.
    """
    #  extracting the cfit file name prefix 
    file_prefix = cfitfile.split('_model_fits.cfit')[0]

    # assigning the prefix to new files to import the normalization files corresponding to our cfit file
    gal_norm_file = file_prefix+'_off_axis_normal_fact.dat'
    model_norm_file = file_prefix+'_model_normal_fact.dat'
    masstable_file = file_prefix+'_masstable.tab'
    
    
    G = np.genfromtxt(gal_norm_file)
    all_model_norms = np.genfromtxt(model_norm_file)

    # Only select the normalization factors corresponding to solar metallicities ~ 0.2 
    solar_met = 0.02
#     met_mask = (all_model_norms.transpose()[0,:] == solar_met) #Takes first column and selects metallicity = 0.02
    
    
    #NEXT few masks are changed for creating the histogram plots for lega-c galaxy spectra using 3 ssps instead of 4ssps
    old_ssps = 9.7500000e+09 #Age of old SSPs
    met_mask = (all_model_norms.transpose()[0,:] == solar_met) #Takes first column and selects metallicity = 0.02
    age_mask = (all_model_norms.transpose()[1,:] < old_ssps) #Takes second column and selects all ages less than old_ssps age
    
    #Remove the element in met_mask that corresponds to old SSP (since we excluded that in the legac galaxy sspmodel plots)
    #Create a mask that is True at places where both the metallicity == solar_met AND the age < old_ssps
    met_and_age_mask = np.empty_like(met_mask, bool)
    for i in range(len(met_mask)):
        met_and_age_mask[i] = (met_mask[i] == True and age_mask[i] == True)
                
    
    if nssps == 3:
        ## Just adding this for legac plots for nasa ppt presentation (excluding old SSPs so we set nssps=3)
        N_i = all_model_norms[met_and_age_mask].T[-1] # if nssps = 3 and we are excluding the *OLD* SSP, apply metallicity and age mask to filter out all stars with nonsolar metallicity, as well as OLD stars with solar metallicity. So will just get the N_i for just the 3 ssps Y, I1, I2, 
    else:
        N_i = all_model_norms[met_mask].T[-1] #For general use (when nssps = 4 : Y,I1,I2,O), just apply metallicity mask for solar metallicity 


    # Stellar Mass Formed for each population:  a_i = b_i*G/N_i , where b_i are the cfit values
    normalization_factors = G/N_i
    
    ## Check if normalization_factors agree with those in the .masstable.tab files:
    masstable_factors = np.genfromtxt(masstable_file)
    print('Calculated mass conversion fraction: ', normalization_factors)
    
    return normalization_factors
    
    

def seaborn_pairwise(data, columns=None, output_name=None, output_dir='./', normalized=None, mass_frac=[]):
    
    """Plots pairwise scatterplot and histograms using seaborn. Saves plot to PDF and saves it in a results_*_plots folder in the same directory as the original file."""

    ##GREG: can probably revert this function back to NOT include the mass fraction, etc calculations
    # and let it be more general. We will want to check that our filename saving in new folder works.
    # I've taken out the datfile keyword and replaced it with output_dir to tell what directory to save file to
    # and I've set the default to save in whatever the current directory is and a output_name keyword
    # to capture what we want to call the output name
    
      
    df = pandas.DataFrame(data=data, columns=columns)

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
    #
    
    
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
    
    #Save this pdf file in a results_*_plots folder inside the current directory
    #Check if dir exists if not make it
    #Make directories for file if they don't exist
    os.makedirs(os.path.dirname(pdfname), exist_ok=True)
    
    g.savefig(pdfname)
    
    

def plot_stellar_mass_fractions(cfitfile, nssps=4):
    """Read in model cfits file, applies stellar mass fraction conversion, plots and saves pairwise histograms.
    nssps = number of ssps"""

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
    #NOTE: Excluding first row which is metric
    #NOTE: Taking transpose for pandas
    data = np.genfromtxt(cfitfile).T[:,1:] 
    print('Shape of cfit data: ', data.shape)
    
    
    #Setting column names
    #Size of column names list depends on how many ssps youre using
    if nssps == 4:
        col_names = ['Y','I1','I2', 'O']#,'Av']
    elif nssps ==3:
        col_names = ['Y','I1','I2']#O, 'Av'] # if excluding old stars, using 3 ssps, only need Y I1 I2 column names - for nasa presentation histogram plots for legac data
    
    
    #Convert to stellar mass fraction
    #This mass conversion function expects a cfitfilename that is a FULL PATH NAME in order to find normalization files
    normalization_factors = mass_fraction_conversion(cfitfile, nssps)
    
    
    #Grab the SSP coefficients
    b_i = data[:,0:nssps]
    #Convert to solar masses
    a_i = b_i*normalization_factors
    #Convert to mass fractions
    massfrac_data = np.asarray([a_i[i]/np.sum(a_i,axis=1)[i] for i in range(a_i.shape[0])])
    #GREG: Here you don't need to convert to a pandas data frame, you can work with the data as numpy array
    #I use a list comprehension here to do the ssp divison by the sum as there was a brodcasting error in just doing the plain divison
    #Numpy doesn't let you divide something that has a 2-d shape 300,4 by something that's 1 d 300,
    

    #Find the average star formation rate in solar masses per year 
    # avg star formation rate = (total mass)/(age of universe at that z)
    #Using z = 1.17, H0 = 70, omega_m = 0.3, omega_vac = 0.7    # from http://www.astro.ucla.edu/~wright/CosmoCalc.html
    #(this is for the LEGA-C galaxy spectra - legac_M13_230105_v2.0 -- for nasa ppt presentation )
    total_mass = np.sum(np.sum(massfrac_data,axis=1)) # np.sum(np.sum(a_i,axis=1))
    age_of_universe =  5.13000000e+09
    avg_star_formation = total_mass/age_of_universe
    print("Average star formation rate: ", avg_star_formation, " solar masses per year")
    

    #Plot
    #Plot pairwise histograms and save them to output directory
    seaborn_pairwise(data=massfrac_data, columns=col_names, output_name=output_name,output_dir=output_dir)

    
    

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
#     data = np.genfromtxt(cfitfile).T[:,:]

   
    #Setting column names 
    col_names = ['Y','I1','I2', 'O','Av']
    
#     data = np.genfromtxt(cfitfile).T[:,:]
    
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
#     plot_cfit_data(datfile) 

    #New stellar mass fraction plots
    #Plot pairwise histograms of stellar mass fractions from cfit data
#     plot_stellar_mass_fractions(datfile)
    plot_stellar_mass_fractions(datfile, nssps=3)

