"""
NAME:
	calculate_mass_outflow.py

FUNCTIONS INCLUDED:
    calc_outflow_mass
    calc_mass_outflow_rate
    calc_mass_outflow_flux
    calc_mass_loading_factor

"""
import numpy as np

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy.constants import m_p
from astropy import units as u

from . import calculate_outflow_velocity as calc_outvel
from . import calculate_star_formation_rate as calc_sfr

import importlib
importlib.reload(calc_sfr)



def calc_outflow_mass(galaxy_dictionary, z, n_e=100):
    """
    Calculates the mass outflow using the equation:
        M_out = (1.36m_H)/(gamma_Hbeta n_e) * L_Halpha,broad
    To convert from Halpha to Hbeta luminosities:
        L_Halpha/L_Hbeta = 2.87
    So:
        M_out = (1.36m_H)/(gamma_Hbeta n_e) * (L_Halpha,broad/L_Hbeta,broad) * L_Hbeta,broad

    Parameters
    ----------
    galaxy_dictionary : dictionary
        dictionary with the threadcount results

    z : float
        redshift

    n_e : float
        the electron density.  Default is 100, Davies+2019 used 380, so I used
        that in Reichardt Chu+2022.

    Returns
    -------
    mout : :obj:'~numpy.ndarray'
        mass of the outflow in units of solar masses
    """
    #from Calzetti 2001 PASP 113 we have L_Halpha/L_Hbeta = 2.87
    lum_ratio_alpha_to_beta = 2.87

    #m_H is the atomic mass of Hydrogen (in kg)
    m_H = m_p

    #gamma_Halpha is the Halpha emissivity at 10^4K (in erg cm^3 s^-1)
    #gamma_Halpha = 3.56*10**-25 * u.erg * u.cm**3 / u.s
    #actually we use Hbeta so:
    gamma_Hbeta = 1.24*10**-25 * u.erg * u.cm**3 / u.s

    #n_e is the local electron density in the outflow
    #give it some units
    n_e = n_e * (u.cm)**-3

    #L_Hbeta is the luminosity of the broad line of Hbeta (we want the outflow flux)
    flux_results, flux_error, outflow_flux, outflow_flux_err = calc_sfr.get_arrays(galaxy_dictionary, var_string='flux')

    #put the units in erg/s/cm^2
    outflow_flux = outflow_flux * 10**(-16) * u.erg / (u.s*(u.cm**2))
    outflow_flux_err = outflow_flux_err * 10**(-16) * u.erg / (u.s*(u.cm**2))

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)
    #multiply by 4*pi*d^2 to get rid of the cm
    L_Hbeta = (outflow_flux*(4*np.pi*(dist**2))).to('erg/s')
    L_Hbeta_err = (outflow_flux_err*(4*np.pi*(dist**2))).to('erg/s')

    #do the whole calculation
    mout = (1.36*m_H) / (gamma_Hbeta*n_e) * lum_ratio_alpha_to_beta * L_Hbeta
    mout_err = (1.36*m_H) / (gamma_Hbeta*n_e) * lum_ratio_alpha_to_beta * L_Hbeta_err

    #decompose the units to solar masses
    mout = mout.to(u.solMass)
    mout_err = mout_err.to(u.solMass)

    return mout, mout_err


def calc_mass_outflow_rate(galaxy_dictionary, z, n_e=100):
    """
    Calculates the mass outflow rate using the equation:
        dM_out = (1.36m_H)/(gamma_Hbeta n_e) * (v_out/R_out) * L_Halpha,broad
    To convert from Halpha to Hbeta luminosities:
        L_Halpha/L_Hbeta = 2.87
    So:
        dM_out = (1.36m_H)/(gamma_Hbeta n_e) * (v_out/R_out) *
                (L_Halpha,broad/L_Hbeta,broad) * L_Hbeta,broad

    Parameters
    ----------
    galaxy_dictionary : dictionary
        dictionary with the threadcount results

    z : float
        redshift

    n_e : float
        the electron density.  Default is 100, Davies+2019 used 380, so I used
        that in Reichardt Chu+2022.

    Returns
    -------
    M_out : :obj:'~numpy.ndarray'
        mass outflow rate in units of solar masses/year

    M_out_max : :obj:'~numpy.ndarray'
        maximum mass outflow rate in units of solar masses/year if R_min is 350pc

    M_out_min : :obj:'~numpy.ndarray'
        minimum mass outflow rate in units of solar masses/year if R_max is 2000pc
    """
    #from Calzetti 2001 PASP 113 we have L_Halpha/L_Hbeta = 2.87
    lum_ratio_alpha_to_beta = 2.87

    #m_H is the atomic mass of Hydrogen (in kg)
    m_H = m_p

    #gamma_Halpha is the Halpha emissivity at 10^4K (in erg cm^3 s^-1)
    #gamma_Halpha = 3.56*10**-25 * u.erg * u.cm**3 / u.s
    #actually we use Hbeta so:
    gamma_Hbeta = 1.24*10**-25 * u.erg * u.cm**3 / u.s

    #n_e is the local electron density in the outflow
    #give it some units
    n_e = n_e * (u.cm)**-3

    #v_out comes from whichever line is brightest that you fit with threadcount
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(galaxy_dictionary)
    #put in the units for the velocity
    vel_out = vel_out * u.km/u.s
    vel_out_err = vel_out_err * u.km/u.s

    #R_out is the radial extent of the outflow
    #use the 90% radius of the galaxy as the maximum radius - this is 5" or 2kpc
    #R_max = 2 * 1000 * u.parsec
    #use the resolution of the spaxels as the minimum radial extent - this is ~350pc
    #R_min = 350 * u.parsec
    #then use the average as R_out
    #R_out = (R_max + R_min)/2

    #use 500pc as the R_out
    R_out = 500 * u.parsec

    #we just want the flux for the outflow from Hbeta
    flux_results, flux_error, outflow_flux, outflow_flux_err = calc_sfr.get_arrays(galaxy_dictionary, var_string='flux')

    #put the units in erg/s/cm^2
    outflow_flux = outflow_flux * 10**(-16) * u.erg / (u.s*(u.cm**2))
    outflow_flux_err = outflow_flux_err * 10**(-16) * u.erg / (u.s*(u.cm**2))

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)
    #multiply by 4*pi*d^2 to get rid of the cm
    L_Hbeta = (outflow_flux*(4*np.pi*(dist**2))).to('erg/s')

    #do the whole calculation
    #dM_out_max = (1.36*m_H) / (gamma_Hbeta*n_e) * (vel_out/R_max) * lum_ratio_alpha_to_beta*L_Hbeta
    #dM_out_min = (1.36*m_H) / (gamma_Hbeta*n_e) * (vel_out/R_min) * lum_ratio_alpha_to_beta*L_Hbeta
    dM_out = (1.36*m_H) / (gamma_Hbeta*n_e) * (vel_out/R_out) * lum_ratio_alpha_to_beta*L_Hbeta

    #decompose the units to g/s
    dM_out = dM_out.to(u.solMass/u.yr)
    #dM_out_max = M_out_max.to(u.g/u.s)
    #dM_out_min = M_out_min.to(u.g/u.s)

    return dM_out #, dM_out_max, dM_out_min


def calc_mass_outflow_flux(galaxy_dictionary, z, wcs_step):
    """
    Calculates the mass outflow flux using the equation:
        Sigma_out = dM_out/area
    where
        dM_out = (1.36m_H)/(gamma_Hbeta n_e) * (v_out/R_out) * L_Halpha,broad
    To convert from Halpha to Hbeta luminosities:
        L_Halpha/L_Hbeta = 2.87

    Parameters
    ----------
    galaxy_dictionary : dictionary
        dictionary with the threadcount results

    z : float
        redshift

    header : FITS header object
        the header from the fits file

    Returns
    -------
    sigma_out : :obj:'~numpy.ndarray'
        mass outflow flux in units of solar masses / yr / kpc^2
    """
    #calculate the mass outflow rate
    dm_out = calc_mass_outflow_rate(galaxy_dictionary, z)

    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec)

    #get the bin area
    x = wcs_step[0]*(u.arcsec)
    y = wcs_step[1]*(u.arcsec)
    """try:
        x = header['CD1_2']*60*60
        y = header['CD2_1']*60*60

    except:
        x = (-header['CDELT2']*np.sin(header['CROTA2'])) *60*60
        y = (header['CDELT1']*np.sin(header['CROTA2'])) *60*60"""

    #divide by the area of the bin
    sigma_out = dm_out/((x*y)*(proper_dist**2))

    return sigma_out #, M_out_max, M_out_min



def calc_mass_loading_factor(galaxy_dictionary, z, wcs_step):
    """
    Calculates the mass loading factor
        eta = M_out/SFR
    Using the calc_sfr.calc_sfr_koffee and the calc_mass_outflow_rate functions

    Parameters
    ----------
    galaxy_dictionary : dictionary
        dictionary with the threadcount results

    z : float
        redshift

    header : FITS header object
        the header from the fits file

    Returns
    -------
    mlf_out : :obj:'~numpy.ndarray'
        mass loading factor

    mlf_max : :obj:'~numpy.ndarray'
        maximum mass loading factor if R_min is 350pc

    mlf_min : :obj:'~numpy.ndarray'
        minimum mass loading factor if R_max is 2000pc
    """
    #calculate the mass outflow rate (in solar masses/year)
    #m_out, m_out_max, m_out_min = calc_mass_outflow_rate(OIII_results, OIII_error, hbeta_results, hbeta_error, statistical_results, z)
    dm_out = calc_mass_outflow_rate(galaxy_dictionary, z)

    #calculate the SFR (I wrote this to give the answer without units...)
    #(I should probably change that!)
    sfr, sfr_err, total_sfr, sigma_sfr, sigma_sfr_err = calc_sfr.calc_sfr(galaxy_dictionary, z, wcs_step, include_outflow=False)

    #put the units back onto the sfr (M_sun/yr)
    sfr = sfr * (u.solMass/u.yr)

    #calculate mass loading factor
    mlf = dm_out/sfr

    #mlf_max = m_out_max/sfr
    #mlf_min = m_out_min/sfr

    return mlf #, mlf_max, mlf_min
