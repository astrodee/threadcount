"""
NAME:
	calculate_star_formation_rate.py

FUNCTIONS INCLUDED:
    calc_hbeta_extinction
    calc_sfr

"""
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units

from astropy.io import fits


#-------------------------------------------------------------------------------
# EXTINCTION CALCULATIONS
#-------------------------------------------------------------------------------
def calc_hbeta_extinction(lamdas, z):
    """
    Calculates the H_beta extinction - corrects for the extinction caused by light
    travelling through the dust and gas of the original galaxy, using the
    Cardelli et al. 1989 curves and Av = E(B-V)*Rv.
    The value for Av ~ 2.11 x C(Hbeta) where C(Hbeta) = 0.24 from
    Lopez-Sanchez et al. 2006 A&A 449.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector
    z : float
        redshift

    Returns
    -------
    A_hbeta : float
        the extinction correction factor at the Hbeta line
    """
    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas**(-1) - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #define the constants
    Rv = 3.1
    #TO DO:
    #this Av is technically for IRAS08 - need a better definition from Hbeta/Hgamma
    Av = 2.11*0.24

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av

    #find A_hbeta
    #first redshift the hbeta wavelength and convert to micrometers
    hbeta = (4861.333*(1+z))/10000
    #then find in lamdas array
    index = (np.abs(lamdas - hbeta)).argmin()
    #use the index to find A_hbeta
    A_hbeta = A_lam[index]

    return A_hbeta

#-------------------------------------------------------------------------------
# FLUX ARRAYS
#-------------------------------------------------------------------------------

def get_arrays(galaxy_dictionary, var_string):
    """
    Uses the dictionary from threadcount to create arrays of the variable for the
    outflow and galaxy gaussians.

    Parameters
    ----------
    galaxy_dictionary : dictionary
        dictionary with the threadcount results

    var_string : str
        the variable to create arrays of (e.g. 'flux', 'height')

    Returns
    -------
    galaxy_var : :obj:'~numpy.ndarray'
        the variable for the galaxy component

    galaxy_var_error : :obj:'~numpy.ndarray'
        the variable error for the galaxy component

    outflow_var : :obj:'~numpy.ndarray'
        the variable for the outflow component

    outflow_var_error : :obj:'~numpy.ndarray'
        the variable error for the outflow component
    """
    #create arrays to save the fluxes into
    galaxy_var = np.full_like(galaxy_dictionary['choice'], np.nan, dtype=np.double)
    galaxy_var_error = np.full_like(galaxy_dictionary['choice'], 0.0, dtype=np.double)
    outflow_var = np.full_like(galaxy_dictionary['choice'], np.nan, dtype=np.double)
    outflow_var_error = np.full_like(galaxy_dictionary['choice'], 0.0, dtype=np.double)

    #create the mask of where outflows are
    flow_mask = (galaxy_dictionary['choice']==2)

    #use the flow mask to define the galaxy and outflow gaussians
    #g1 is the blue-est gaussian, so we need g2 where choice==2
    #and we need g1 where choice==1
    galaxy_var[~flow_mask] = galaxy_dictionary['avg_g1_'+var_string][~flow_mask]
    galaxy_var[flow_mask] = galaxy_dictionary['avg_g2_'+var_string][flow_mask]

    galaxy_var_error[~flow_mask] = galaxy_dictionary['avg_g1_'+var_string+'_err'][~flow_mask]
    galaxy_var_error[flow_mask] = galaxy_dictionary['avg_g2_'+var_string+'_err'][flow_mask]

    outflow_var[flow_mask] = galaxy_dictionary['avg_g1_'+var_string][flow_mask]
    outflow_var_error[flow_mask] = galaxy_dictionary['avg_g1_'+var_string+'_err'][flow_mask]

    #now we need all the places where there are two Gaussians, BUT the red one
    #has a lower height and lower flux than the blue one
    #then we swap them

    #flow_mask = (galaxy_dictionary['choice']==2) & (galaxy_dictionary['avg_g1_flux']>galaxy_dictionary['avg_g2_flux']) & (galaxy_dictionary['avg_g1_height']>galaxy_dictionary['avg_g2_height'])
    flow_mask = (galaxy_dictionary['choice']==2) & (galaxy_dictionary['avg_g1_height']>galaxy_dictionary['avg_g2_height'])

    galaxy_var[flow_mask] = galaxy_dictionary['avg_g1_'+var_string][flow_mask]
    galaxy_var_error[flow_mask] = galaxy_dictionary['avg_g1_'+var_string+'_err'][flow_mask]

    outflow_var[flow_mask] = galaxy_dictionary['avg_g2_'+var_string][flow_mask]
    outflow_var_error[flow_mask] = galaxy_dictionary['avg_g2_'+var_string+'_err'][flow_mask]


    return galaxy_var, galaxy_var_error, outflow_var, outflow_var_error


#-------------------------------------------------------------------------------
# SFR CALCULATIONS
#-------------------------------------------------------------------------------

def calc_sfr(galaxy_dictionary, z, wcs_step, include_outflow=False):
    """
    Calculates the star formation rate using Hbeta
    SFR = C_Halpha (L_Halpha / L_Hbeta)_0 x 10^{-0.4A_Hbeta} x L_Hbeta[erg/s]
    The Hbeta flux is calculated using the results from the threadcount fits.
    Assumes that the extinction has already been corrected.

    Parameters
    ----------
    galaxy_dictionary : dictionary
        dictionary with the threadcount results

    z : float
        redshift

    header : FITS header object
        the header from the fits file

    include_outflow : boolean
        if True, includes the broad outflow component in the flux calculation.
        If false, uses only the narrow component to calculate flux.  Default is
        False.

    Returns
    -------
    sfr : :obj:'~numpy.ndarray'
        the star formation rate found using Hbeta in M_sun/yr

    sfr_err : :obj:'~numpy.ndarray'
        the error of the star formation rate found using Hbeta in M_sun/yr

    total_sfr : float
        the total SFR of all the spectra input in M_sun/yr

    sfr_surface_density : :obj:'~numpy.ndarray'
        the star formation rate surface density found using Hbeta in M_sun/yr/kpc^2

    sfr_surface_density_err : :obj:'~numpy.ndarray'
        the error of the star formation rate surface density found using Hbeta
        in M_sun/yr/kpc^2
    """
    #first we need to define C_Halpha, using Hao et al. 2011 ApJ 741:124
    #From table 2, uses a Kroupa IMF, solar metallicity and 100Myr
    c_halpha = 10**(-41.257)

    #from Calzetti 2001 PASP 113 we have L_Halpha/L_Hbeta = 2.87
    lum_ratio_alpha_to_beta = 2.87

    #get the flux arrays
    flux_results, flux_error, outflow_flux, outflow_flux_err = get_arrays(galaxy_dictionary, var_string='flux')

    #add the outflow flux to the total flux if including the outflow in the SFR
    if include_outflow == True:
        flux_results = np.nansum((flux_results, out_flux_results), axis=0)
        flux_error = np.sqrt(np.nansum((flux_error**2, out_flux_error**2), axis=0))


    #give the flux units (10^-16 erg/s/cm^2)
    flux_results = flux_results*10**(-16)*units.erg/(units.s*(units.cm*units.cm))
    flux_error = flux_error*10**(-16)*units.erg/(units.s*(units.cm*units.cm))

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)

    #multiply by 4*pi*d^2 to get rid of the cm
    hbeta_luminosity = (flux_results*(4*np.pi*(dist**2))).to('erg/s')
    hbeta_luminosity_err = (flux_error*(4*np.pi*(dist**2))).to('erg/s')

    #calculate the star formation rate
    sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*0.0) * (hbeta_luminosity)
    sfr_err = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*0.0) * (hbeta_luminosity_err)

    total_sfr = np.nansum(sfr)

    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(units.kpc/units.arcsec)

    x = wcs_step[0] *(units.arcsec)
    y = wcs_step[1] *(units.arcsec)

    print('Spaxel Area:', (x*y)*(proper_dist)**2)
    print(' ')

    """try:
        x = header['CD1_2']*60*60
        y = header['CD2_1']*60*60

    except:
        x = (-header['CDELT2']*np.sin(header['CROTA2'])) *60*60
        y = (header['CDELT1']*np.sin(header['CROTA2'])) *60*60"""

    sfr_surface_density = sfr/((x*y)*(proper_dist**2))
    sfr_surface_density_err = sfr_err/((x*y)*(proper_dist**2))

    print(sfr.unit)

    return sfr.value, sfr_err.value, total_sfr.value, sfr_surface_density.value, sfr_surface_density_err.value
