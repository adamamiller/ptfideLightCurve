from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.table import Table


class ide_lc(self):
    """
    Object to store light curve and associated properties
    """
    def __init__(self, name):
        self.name = name
    
    def read_raw_ide(self, ide_file):
        """
        Read in the data from the PTFIDE output
        
        Parameters
        ----------
        ide_file : str, filename
            Full path to file containing output from PTFIDE forced photometry.
            Typically has a name like forcepsffitdiff_dFFFFFF_fB_cC.out, where
            FFFFFF is the field id, B is the filter (1 = g, 2 = R), and C is 
            the ccd id.
        
        Attributes
        ----------
        hjd_ : array-like
            Heliocentric Julian Day for epochs of target observations
        
        flux_ : array-like
            PSF flux measurements at target position corresponding to hjd
        
        flux_unc_ : array-like
            Flux uncertainty measurements
        
        chi_ : array-like
            Chi^2 measurement of the PSF fit to the source
        
        zpmag_ : array-like
            ZP magnitude for every epoch
        
        zprms_ : array-like
            Scatter (uncertainty) in the ZP mag
        """

        lcDat = Table.read(self.ide_file, format="ipac")
        self.hjd_ = lcDat['HJD']
        self.flux_ = lcDat["flux"]
        self.flux_unc_ = lcDat["sigflux"]
        self.chi_ = lcDat["chi"]
        self.zpmag_ = lcDat["zpmag"]
        self.zprms_ = lcDat["zprms"]
    
    def read_ref_flux(self, ide_file):
        """
        Read the flux present in the reference image, which is needed for 
        stellar light curves
        
        Parameters
        ----------
        ide_file : str, filename
            Full path to file containing output from PTFIDE forced photometry.
            Typically has a name like forcepsffitdiff_dFFFFFF_fB_cC.out, where
            FFFFFF is the field id, B is the filter (1 = g, 2 = R), and C is 
            the ccd id.
        
        Attributes
        ----------
        ref_flux_ : float
            Flux (in DN) of the source in the reference image
        
        ref_flux_unc_ : float
            Flux uncertainty of the source in the reference image
        
        """
        with open(ide_file) as f:
            ll = f.readlines()
            ref_flux = ll[-2].split(" DN")[0].split(" ")[1]
            ref_flux_unc = ll[-2].split(" DN")[1].split(" ")[1]
        self.ref_flux_ = float(ref_flux)
        self.ref_flux_unc_ = float(ref_flux_unc)
        
        
def ptfide_light_curve(ideFile, hjd0, SNT = 3, SNU = 5, plotLC = False):
    """
    Produce calibrated mag measurements from PTFIDE forced photometry
    
    Parameters
    ----------
    ideFile : str, filename
        Full path to file containing output from PTFIDE forced photometry.
        Typically has a name like forcepsffitdiff_dFFFFFF_fB_cC.out, where
        FFFFFF is the field id, B is the filter (1 = g, 2 = R), and C is 
        the ccd id.
    
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
        Heliocentric Julian Date on epochs when the source is not 
        detected.

    magLim : array
        Upper limits on epochs when the source is not detected.
    
    hjdFlux : array-like
        Heliocentric Julian Date on epochs with reliable 
        subtractions.
    
    flux : array-like
        flux on all epochs with reliable subtractions under 
        assumption that the photometry is calibrated to the AB system. This 
        assumption is not correct, nevertheless it is useful to 
        propagate uncertainties associated with ZP
        
    fluxUnc : array
        flux uncertainty on epochs with reliable subtractions under 
        assumption that the photometry is calibrated to the AB system. This 
        assumption is not correct, nevertheless it is useful to 
        propagate uncertainties associated with ZP
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
    
    f_minus_b = lcDat['flux'] - baseline
    snr = (f_minus_b)/sigflux
    det = np.where((snr >= SNT) & goodSubs)
    lims = np.where((snr < SNT) & goodSubs)

    # calculate output terms
    hjdDet = lcDat['HJD'][det]
    magDet = lcDat['zpmag'][det] - 2.5*np.log10(f_minus_b[det])
    magUncDet = np.hypot(lcDat['zprms'][det],
                         1.0857*sigflux[det]/(f_minus_b[det]))

    hjdLim = lcDat['HJD'][lims]
    magLim = lcDat['zpmag'][lims] - 2.5*np.log10(SNU * sigflux[lims])

    # fluxes are in psuedo Jy
    hjdFlux = lcDat['HJD'][goodSubs]
    flux = 3631 * f_minus_b[goodSubs] * 10**(-lcDat['zpmag'][goodSubs]/2.5)
    fluxUnc = np.abs(flux) * \
              np.hypot( np.divide(sigflux[goodSubs], 
              f_minus_b[goodSubs], 
              where=f_minus_b[goodSubs] != 0, 
              out = sigflux[goodSubs]/lcDat['flux'][goodSubs]),
              np.log(10)/2.5*lcDat['zprms'][goodSubs] )
    
    if plotLC and len(magDet) > 0:
        plt.errorbar( hjdDet, magDet, magUncDet, fmt = 'o')
        plt.plot(hjdLim, magLim, 'v')
        plt.ylim(max(magLim) + 0.1, min([min(magLim), min(magDet)]) - 0.1)
        plt.xlim(min([min(hjdLim), min(hjdDet)]), max([max(hjdLim), max(hjdDet)])+25)
    elif plotLC and len(magDet) == 0:
        print("Warning - cannot produce plots as the transient is not detected")
    
    return hjdDet, magDet, magUncDet, hjdLim, magLim, hjdFlux, flux, fluxUnc
