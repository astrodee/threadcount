"""
NAME:
	calculate_outflow_velocity.py

FUNCTIONS INCLUDED:
    calc_outflow_vel

"""
import math
import numpy as np
from astropy.io import fits

from . import calculate_star_formation_rate as calc_sfr



def calc_outflow_vel(galaxy_dictionary, sigma_lsf=41.9):
    """
    Calculates the outflow velocity

    Parameters
    ----------
    galaxy_dictionary : dictionary
        dictionary with the threadcount results

    sigma_lsf : float
        The velocity line spread function to convolve the velocity with, in km/s
        (Default is 40 km/s)

    Returns
    -------
    vel_disp : :obj:'~numpy.ndarray'
        Array with the dispersion of the outflow component in km/s, and np.nan
        where no velocity was found.

    vel_disp_err : :obj:'~numpy.ndarray'
        Array with the error for dispersion of the outflow component in km/s,
        and np.nan where no velocity was found.

    vel_diff : :obj:'~numpy.ndarray'
        Array with the mean difference between the outflow and systemic lines in
        km/s, and np.nan where no velocity was found.

    vel_diff_err : :obj:'~numpy.ndarray'
        Array with the error for the mean difference between the outflow and
        systemic lines in km/s, and np.nan where no velocity was found.

    vel_out : :obj:'~numpy.ndarray'
        Array with the outflow velocities in km/s, and np.nan where no velocity
        was found.

    vel_out_err : :obj:'~numpy.ndarray'
        Array with the outflow velocity errors in km/s, and np.nan where no
        velocity was found.
    """
    #get the results from threadcount
    systemic_mean, systemic_mean_err, flow_mean, flow_mean_err = calc_sfr.get_arrays(galaxy_dictionary, var_string='center')
    systemic_sigma, systemic_sigma_err, flow_sigma, flow_sigma_err = calc_sfr.get_arrays(galaxy_dictionary, var_string='sigma')

    #calculate the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff_calc = 299792.458*abs(systemic_mean - flow_mean)/systemic_mean

    #calculate the error on the velocity difference

    #do the numerator first (lam_gal-lam_out)
    num_err = np.sqrt(systemic_mean_err**2 + flow_mean_err**2)
    #now put that into the vel_diff error
    vel_diff_calc_err = vel_diff_calc * np.sqrt((num_err/(systemic_mean-flow_mean))**2 + systemic_mean_err**2/systemic_mean**2)

    #calculate the dispersion
    vel_disp_calc = flow_sigma*299792.458/systemic_mean

    #calculate the error on velocity dispersion
    vel_disp_calc_err = vel_disp_calc * np.sqrt((flow_sigma_err/flow_sigma)**2 + (systemic_mean_err/systemic_mean)**2)

    #convolve the dispersion with the lsf
    vel_disp_convolved = np.sqrt(vel_disp_calc**2 - sigma_lsf**2)

    #calculate the error on the convolved velocity dispersion
    vel_disp_convolved_err = np.sqrt((vel_disp_calc/vel_disp_convolved)**2 * vel_disp_calc_err**2 + (sigma_lsf/vel_disp_convolved)**2 * 0.0)

    #now doing 2*c*flow_sigma/lam_gal + vel_diff
    v_out = 2*vel_disp_convolved + vel_diff_calc

    #calculate the error on v_out
    v_out_err = np.sqrt(4*vel_disp_convolved_err**2 + vel_diff_calc_err**2)

    #and put it into the array
    vel_diff = vel_diff_calc
    vel_diff_err = vel_diff_calc_err
    vel_disp = vel_disp_calc
    vel_disp_err = vel_disp_calc_err
    vel_out = v_out
    vel_out_err = v_out_err

    return vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err
