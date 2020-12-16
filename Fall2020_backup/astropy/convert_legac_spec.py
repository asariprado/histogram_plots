#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from astropy.io import fits
import glob
import matplotlib.pyplot as plt



# CONVERTING LEGA-C SPECTRA to 1D spectra fits files

def convert_legac_spec(legac_file):
    """Converts LEGA-C Spectra FITS files to 1D Spectra fits files. 
    Takes in the name of a LEGA-C spectrum with the full path name """
    #spechdr for sspmodel fits files: need to have NAXIS = 2 (so the data we save needs to have 2 dimensions)
#CRVAL1 and CDELT1  are in meters but can also use angstroms


#We need keywords: NAXIS1, NAXIS2, CRVAL1 and CDELT1 -- to reconstruct the wavelength for each flux point without carrying the wavelength array values
#We'll use Angstroms for units
    
    
    
    
#     legafiles = glob.glob(legac_dir+'*fits')
    legac_spec_HDU = fits.open(legac_file)
    legac_spec_header = legac_spec_HDU[0].header
    

    
    #Find the WAVELMIN & WAVELMAX info in the legac file's header to get max & min values for wavelength
    wavel_min = legac_spec_header['WAVELMIN']
    wavel_max = legac_spec_header['WAVELMAX']
    spec_bin = legac_spec_header['SPEC_BIN']
    
    nwave = (wavel_max - wavel_min)/spec_bin + 1
    
    #Get actual wavelengths to convert from nm --> angstroms
    waves_nm = wavel_min + spec_bin*np.arange(nwave)
    waves_ang = waves_nm * 10
    
       
    
    #Setting header values to the expected header variable names for SSPMODEL
    #But first need to convert units to Angstroms (or check if they're already in Angstroms?? (maybe))
    cdelt = spec_bin * 10  #Use SPEC_BIN as the CDELT1 keyword
    crval1 = wavel_min * 10  #Use WAVELMIN as the CRVAL1 keyword in sspmodel / 1D spec files 
    
    
    
    #Go back to the legac_spec_HDU, grab the [1]st index entry (the binary fits table)
    #to grab the wavelength & flux info
    legac_spec_tab = legac_spec_HDU[1].data
    
    flux_tab = legac_spec_tab['FLUX']
    wave_tab = legac_spec_tab['WAVE']
    
    #Filter out flux values so that we only get the region of spectra where flux is nonzero
    wave_good = wave_tab[flux_tab!=0]
    flux_good = flux_tab[flux_tab!=0] * 1e-19
    nwave_good = len(wave_good)
    
    #WRITE THE LEGAC SPECTRUM TO A 1D SPECTRUM FITS FILE FOR SSPMODEL:
    #1) The SSPMODEL FITS style spectrum is saved to a single PrimaryHDU
    #2) Set the CRVAL1 & CDELT1 keywords
    #3) Since we need NAXIS=2, set the first index in NAXIS to 'None' when creating the PrimaryHDU (like PrimaryHDU(data=datafile[None, :])

    #Save the spectrum's flux to PrimaryHDU
    prim_HDU = fits.PrimaryHDU(data=flux_good[None,:])
    
    #Set the wavelength keywords, cdelt & crval1
    prim_HDU.header['CDELT1'] = cdelt
    prim_HDU.header['CRVAL1'] = crval1
    #prim_HDU.header
    
    
    #Write the FITS file & save the new file in the same directory as the original
    prim_HDU.writeto(legac_file.split('.fits')[0]+'_off_axis.fits', overwrite=True)
    
    print('Saved as: ' + str(legac_file.split('.fits')[0]+'_off_axis.fits'))
    
    
  
   
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not enough arguments. Please supply data file name. Exiting")
        sys.exit(1)

    #Read file name - input the lega-c spectra filename with full path to file
    legac_file = sys.argv[1]
   
  
    #Call the function
    convert_legac_spec(legac_file)

    

    
    

