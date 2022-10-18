"""
NAME:
	calculate_velocity_cuts.py

FUNCTIONS INCLUDED:


"""
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy import units
from astropy.constants import G
from astropy.cosmology import WMAP9 as cosmo

from threadcount import fit
from threadcount import models
from threadcount import lines
from threadcount.procedures import calculate_star_formation_rate as calc_sfr

import importlib
importlib.reload(fit)



#-------------------------------------------------------------------------------
# READ IN DATA
#-------------------------------------------------------------------------------

def fits_read_in(filename):
    """
    Reads in the data fits file
    """
    cube = fit.open_fits_cube(filename, data_hdu_index=0, var_filename=filename, var_hdu_index=1)

    return cube

def get_wave_vector(cube, z=None):
    """
    Gets the wavelength vector from the mpdaf cube object and deredshifts it
    """
    wave = cube.wave.coord()

    if z is not None:
        #wave = wave/(1+z)
        cube.wave.set_crval(cube.wave.get_crval()/(1+z))
        cube.wave.set_step(cube.wave.get_step()/(1+z))

    return cube



def read_in_threadcount_dict(filename):
    """
    Reads in the threadcount output as a dictionary
    """
    #read in the dictionary
    gal_dict = fit.ResultDict.loadtxt(filename)

    #get the comment lines
    comment_lines = gal_dict.comment.split('\n')

    #get the WCS from the comment lines
    wcs_step = extract_from_comments(comment_lines, 'wcs_step:')

    #get the redshift from the comment lines
    z = extract_from_comments(comment_lines, 'z_set:')

    return gal_dict, wcs_step, z


def extract_from_comments(comment_lines, search_string):
    """
    Extracts info from the comments in the threadcount output text file
    Copied from a extract_wcs() in analyze_outflow_extent.py
    """
    #search_string = "wcs_step:"
    wcs_line = [x for x in comment_lines if x.startswith(search_string)][0]
    return eval(wcs_line[len(search_string) :].strip().replace(" ", ","))

#-------------------------------------------------------------------------------
# SUBTRACT BASELINE
#-------------------------------------------------------------------------------

def create_subcube(cube, center_wavelength=lines.Hb4861, wavelength_range=(-150,150)):
    """
    Creates a subcube centred on the emission line

    Parameters
    ----------
    cube : :class:`mpdaf.obj.Cube`
        A datacube containing the wavelength range set in these parameters
    center_wavelength : float, optional
        The center wavelength of the emission line to fit, by default :const:`threadcount.lines.OIII5007`
    wavelength_range : array-like [float, float], optional
        The wavelength range to fit, in Angstroms. These are defined as a change
        from the `center_wavelength`, by default (-15, 15)
    """
    subcube = cube.select_lambda(
        center_wavelength + wavelength_range[0],
        center_wavelength + wavelengths[1])

    return subcube


def subtract_baseline(spec, this_baseline_range, baseline_fit_type):
    """
    Subtracts the baseline from the spectrum
    """
    #create the fit
    baseline_fit = fit.fit_baseline(
        spec,
        this_baseline_range=this_baseline_range,
        baseline_fit_type=baseline_fit_type)

    #subtract the best fit from the data
    new_spec = spec.data - baseline_fit.best_fit

    return baseline_fit.best_fit, new_spec




#-------------------------------------------------------------------------------
# SUBTRACT CENTRAL LINE
#-------------------------------------------------------------------------------

def subtract_gaussian(wave, spec, height, center, sigma, const=None):
    """
    Subtracts the fitted gaussian from the data
    """
    #get the gaussian
    gauss = models.gaussianH(wave, height=height, center=center, sigma=sigma)

    #add the constant
    if const is not None:
        gauss = gauss + const

    #subtract from the data
    residuals = spec - gauss

    return residuals




#-------------------------------------------------------------------------------
# CONVERT TO VELOCITY SPACE
#-------------------------------------------------------------------------------

def wave_to_vel(wave, center):
    """
    Converts the wavelength to the velocity

    Parameters
    ----------
    wave : :obj:'~numpy.ndarray'
        Vector of wavelengths
    center : float
        The central value fit for the narrow galaxy gaussian

    Returns
    -------
    vel_vector : :obj:'~numpy.ndarray'
        Vector of velocities
    """
    #minus the central wavelength off the wavelength vector
    wave = wave - center

    #do c*wave/center
    c = 299792.458 * (units.km/units.s)
    vel_vector = c * wave/center

    return vel_vector

#-------------------------------------------------------------------------------
# VELOCITY BANDS
#-------------------------------------------------------------------------------

def get_velocity_bands(vel_vec, residuals, gal_center, gal_sigma, v_esc):
    """
    Gets the flux in each velocity band
    """
    #convert the galaxy sigma to velocity space
    #do c*wave/center
    c = 299792.458 * (units.km/units.s)
    gal_sigma_vel = c * gal_sigma/gal_center

    #Disk Turbulence
    #add up everything between vel=0 and vel=gal_sigma
    disk_turb_mask = (vel_vec > -gal_sigma_vel.value) & (vel_vec < gal_sigma_vel.value)
    residuals_masked = ma.masked_where(~disk_turb_mask, residuals)
    disk_turb_flux = np.nansum(residuals_masked, axis=0)

    #Fountain Gas
    #add up everything between vel=gal_sigma and vel=escape vel
    fountain_mask = (vel_vec < -gal_sigma_vel.value) & (vel_vec > -v_esc)
    residuals_masked = ma.masked_where(~fountain_mask, residuals)
    fountain_flux = np.nansum(residuals_masked, axis=0)

    #Escaping gas
    #add up everything between the escape velocity and where the flux reaches the standard deviation
    #escape_mask = (vel_vec.value<-v_esc) & ()

    return disk_turb_flux, fountain_flux

#-------------------------------------------------------------------------------
# PLOTS
#-------------------------------------------------------------------------------

def plot_data_minus_gal(wave, cube, residuals, gal_dict, i, j):
    """
    Plot of the two gaussian fit, and the leftover data
    """
    #get the centre values
    gal_center, gal_center_err, flow_center, flow_center_err = calc_sfr.get_arrays(gal_dict, var_string='center')

    #get the height values
    gal_height, gal_height_err, flow_height, flow_height_err = calc_sfr.get_arrays(gal_dict, var_string='height')

    #get the sigma values
    gal_sigma, gal_sigma_err, flow_sigma, flow_sigma_err = calc_sfr.get_arrays(gal_dict, var_string='sigma')

    #get the constant values
    const, const_err = gal_dict['avg_c'], gal_dict['avg_c_err']

    #create the gaussians
    gal_gauss = models.gaussianH(wave, height=gal_height[i,j], center=gal_center[i,j], sigma=gal_sigma[i,j])

    flow_gauss = models.gaussianH(wave, height=flow_height[i,j], center=flow_center[i,j], sigma=flow_sigma[i,j])

    #create the interpolated model
    model_x = np.linspace(gal_center[i,j]-15, gal_center[i,j]+15,500)
    model_mask = (wave>gal_center[i,j]-15) & (wave<gal_center[i,j]+15)
    model_interp = np.interp(model_x, wave[model_mask], gal_gauss[model_mask]+flow_gauss[model_mask]+const[i,j])

    #plot the things
    plt.figure()

    plt.step(wave, cube.data[:,i,j], where='mid', c='k', label='data')

    plt.step(wave, gal_gauss+const[i,j], where='mid', c='g', ls='--', label='galaxy gaussian')
    plt.step(wave, flow_gauss+const[i,j], where='mid', c='b', ls='--', label='outflow gaussian')

    #plt.plot(wave, gal_gauss+flow_gauss+const[i,j], c='grey', ls=':', label='model fit')
    plt.plot(model_x, model_interp, c='grey', ls=':', label='total model fit')

    plt.step(wave, residuals[:,i,j], where='mid', c='r', label='data - galaxy')

    plt.xlim(gal_center[i,j]-10, gal_center[i,j]+10)

    plt.title('Data and Galaxy Gaussian-subtracted residual ('+str(i)+', '+str(j)+')')

    plt.legend()

    plt.show()






#-------------------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------------------

def main(data_filename, tc_filename):
    """
    Runs the whole thing
    """
    #read in the data file
    cube = fits_read_in(data_filename)

    #read in the threadcount results
    gal_dict, wcs_step, z = read_in_threadcount_dict(tc_filename)

    #deredshift the wavelength array
    cube = get_wave_vector(cube, z=z)

    #create an array to put the gaussian-subtracted data in
    #and the velocity vectors
    residuals = np.zeros_like(cube.data)
    vel_vecs = np.zeros((cube.data.shape[0], cube.data.shape[1], cube.data.shape[2]))

    #get the centre values
    gal_center, gal_center_err, flow_center, flow_center_err = calc_sfr.get_arrays(gal_dict, var_string='center')

    #get the height values
    gal_height, gal_height_err, flow_height, flow_height_err = calc_sfr.get_arrays(gal_dict, var_string='height')

    #get the sigma values
    gal_sigma, gal_sigma_err, flow_sigma, flow_sigma_err = calc_sfr.get_arrays(gal_dict, var_string='sigma')

    #get the constant values
    const, const_err = gal_dict['avg_c'], gal_dict['avg_c_err']

    #create a subcube with a shorter wavelength range
    subcube = create_subcube(cube)

    #iterating through the data array
    for i in np.arange(sub_cube.data.shape[1]):
        for j in np.arange(sub_cube.data.shape[2]):
            this_spec = sub_cube[:,i,j]

            #subtract the baseline from the data
            if baseline_fit_type is not None:
                baseline_fit, new_spec = subtract_baseline(this_spec, this_baseline_range, baseline_fit_type)
                this_spec = new_spec

            #subtract the gaussian from the data
            residuals[:,i,j] = subtract_gaussian(cube.wave.coord(), cube.data[:,i,j], gal_height[i,j], gal_center[i,j], gal_sigma[i,j], const=const[i,j])

            #transform from wavelength to velocity space
            vel_vecs[:,i,j] = wave_to_vel(cube.wave.coord(), gal_center[i,j])

    #do the flux calculation
    disk_turb_flux, fountain_flux = get_velocity_bands(vel_vecs, residuals, gal_center, gal_sigma, v_esc=300)


    #return residuals, vel_vecs
    return residuals, disk_turb_flux, fountain_flux
