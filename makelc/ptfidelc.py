from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.table import Table

def ptfide_light_curve(ideFile, hjd0, SNT = 3, SNU = 5, plotLC = False):
    """
    Produce calibrated mag measurements from PTFIDE forced photometry
    
    Parameters
    ----------
    ideFile : str, filename
        Full path to file containing output from PTFIDE forced photometry.
        Typically has a name like forcepsffitdiff_dFFFFFF_fB_cC.out, where
        FFFFFF is the 
    
    hjd0 : float
        The earliest date corresponding to a possible detection of the 
        transient. Observations taken before hjd0 are used to establish the 
        baseline flux in PTFIDE subtractions.
        
    SNT : float, optional (default = 3)
        Signal to Noise ratio threshold for an observation to be considered
        detected. WARNING - the default is an arbitrary selection.
    
    SNU : float, optional (default = 5)
        Signal to noise ratio used to report upper limits.
        
    plotLC : boolean, optional (default=False)
        Option to produce a plot of the corresponding light curve.
    
    Returns
    -------
    hjdDet : array-like
        Heliocentric Julian Date on epochs when the source is detected.
        
    magDet : array-like
        mag on epochs when the source is detected.
        
    magUncDet : array
        mag uncertainty on epochs when the source is detected.

    hjdLim : array
        Heliocentric Julian Date on epochs when the source is not detected.

    magLim : array
        Upper limits on epochs when the source is not detected.
    
    hjdFlux : array-like
        Heliocentric Julian Date on all epochs with reliable subtractions.
    
    flux : array-like
        flux on all epochs with reliable subtractions.
        
    fluxUnc : array
        flux uncertainty on all epochs with reliable subtractions.
    """

    lcDat = Table.read(ideFile, format="ipac")
    goodSubs = np.ones(len(lcDat)).astype(bool)
    early = np.zeros(len(lcDat)).astype(bool)

    goodSubs[np.where(lcDat["flux"] == 99999999)] = 0
    early[np.where((lcDat['HJD'] <= hjd0) & goodSubs)] = 1
    
    # determine baseline flux prior to transient
    baseline = np.median(lcDat['flux'][early])

    # first step - rescale by the Chi values as suggested by Frank
    chi = np.median(lcDat['chi'][early])
    if chi > 1:
        sigflux = lcDat['sigflux']*chi
    else:
        sigflux = lcDat['sigflux']

    # second step - check if scatter in flux measurements is similar to mean uncertainty
    s = np.std(lcDat['flux'][early], ddof = 1)/np.median(sigflux[early])
    if s > 1:
        sigflux = s * sigflux
    
    snr = (lcDat['flux']+baseline)/sigflux
    det = np.where((snr >= SNT) & goodSubs)
    lims = np.where((snr < SNT) & goodSubs)

    hjdDet = lcDat['HJD'][det]
    magDet = lcDat['zpmag'][det] - 2.5*np.log10(lcDat['flux'][det]+baseline)
    magUncDet = 1.0857*sigflux[det]/(lcDat['flux'][det]+baseline)

    hjdLim = lcDat['HJD'][lims]
    magLim = lcDat['zpmag'][lims] - 2.5*np.log10(SNU * sigflux[lims])

    hjdFlux = lcDat['HJD'][goodSubs]
    flux = lcDat['flux'][goodSubs]+baseline
    fluxUnc = sigflux[goodSubs]
    
    if plotLC and len(magDet) > 0:
        plt.errorbar( hjdDet, magDet, magUncDet, fmt = 'o')
        plt.plot(hjdLim, magLim, 'v')
        plt.ylim(max(magLim) + 0.1, min([min(magLim), min(magDet)]) - 0.1)
        plt.xlim(min([min(hjdLim), min(hjdDet)]), max([max(hjdLim), max(hjdDet)])+25)
    elif plotLC and len(magDet) == 0:
        print("Warning - cannot produce plots as the transient is not detected")
    
    return hjdDet, magDet, magUncDet, hjdLim, magLim, hjdFlux, flux, fluxUnc
