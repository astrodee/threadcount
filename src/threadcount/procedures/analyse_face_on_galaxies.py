"""
NAME:
	analyse_face_on_galaxies.py

FUNCTIONS INCLUDED:


"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpdaf.obj import Image, gauss_image
from astropy import units
from astropy.constants import G
from astropy.cosmology import WMAP9 as cosmo

import cmasher as cmr

import threadcount as tc
from threadcount.procedures import calculate_star_formation_rate as calc_sfr
from threadcount.procedures import calculate_velocity_cuts as calc_vc
from threadcount.procedures import set_rcParams
from threadcount.procedures import open_cube_and_deredshift



def run(user_settings):
    #the inputs that will be used if there are no command line arguments
    default_settings = {
        # need the continuum subtracted data fits file for the velocity cuts method
        "data_filename" : "ex_gal_fits_file.fits",
        "two_gauss_mc_input_file" : "ex_4861_mc_best_fit.txt",
        "gal_name" : "example_galaxy_name", #used for labelling plots
        "z" : 0.03, # redshift
        # now for some default fitting details
        "line" : tc.lines.L_Hb4861,
        "baseline_subtract" : None, # baseline can be "quadratic" or "linear" or None
        # the range of wavelengths to use in the baseline subtraction
        "baseline_fit_range" : [
                            ], # a list of: [[left_begin, left_end],[right_begin, right_end]], one for each line
        # also need the stellar mass of the galaxy
        "stellar_mass" : 10**11.21, # MUST BE INCLUDED
        # we need the escape velocity for the data
        # either this is a given parameter, or threadcount can work it out if
        # you give it data to use to calculate the effective radius
        "escape_velocity" : 456 * units.km/units.s, # put as None if you don't know
        # either give the effective radius, or the threadcount output will
        # be used to find an effective radius, which assumes that the entire
        # galaxy is within the field of view for the IFU data
        "effective_radius" : None, # in arcseconds, used to calculate the escape
        # velocity, so doesn't technically need to be v_50
        # alternatively, you can give another data image (e.g. PANSTARRs) to use
        # to find the effective radius
        "image_data_filename" : "ex_image_file.fits", #or put as None
        # escape_velocity MUST BE NONE IF YOU WANT TO CALCULATE IT FROM THE
        # IMAGE DATA FILE GIVEN ABOVE
        "average_disk_sigma" : None, # if None will use the threadcount fits to find average_disk_sigma.
        # output options
        "output_base_name" : "ex_velocity_cuts_results", # saved files will begin with this
        "plot_results" : True, #boolean
        "crop_data" : None, # Use to define how much of the data goes into the maps in the
        # format: [axis1_begin, axis1_end, axis2_begin, axis2_end] or None
        # e.g. [2, -1, 3, -2] will map data[2:-1, 3:-2]
    }
    #test if the cube has been opened.  If not, open cube and deredshift
    if "cube" not in user_settings.keys():
        user_settings = open_cube_and_deredshift.run(user_settings)
        set_rcParams.set_params({"image.aspect" : user_settings["image_aspect"]})
    # s for settings
    s = tc.fit.process_settings_dict(default_settings, user_settings)

    #print('data filename:', s.data_filename)
    print('tc data filename:', s.two_gauss_mc_input_file)

    # calculate the escape velocity if it wasn't given
    if s.escape_velocity is None:
        # check for a given effective radius
        if s.effective_radius is None:
            #need to calculate the effective radius
            if s.image_data_filename is None:
                #use the threadcount output to find the effective radius
                tc_data, _, _ = calc_vc.read_in_threadcount_dict(s.two_gauss_mc_input_file)

                _, _, rad = data_coords(tc_data, s.z_set, s.wcs_step)

                s.effective_radius = calc_effective_radius_tc(tc_data, rad, flux_percentage=50)

                #get rid of stuff we don't need out of memory
                del tc_data, rad

            else:
                #use the given image data to find the effective radius
                s.effective_radius = calc_effective_radius_fits(s.image_data_filename, fits_ext='COMPRESSED_IMAGE', flux_percentage=50)

        #now use that to calculate the escape velocity
        #this uses the assumed redshift from settings, or if you've used tc_data
        #to calculate the effective_radius it uses the z found in threadcount
        print("redshift:", s.z_set)
        s.escape_velocity = calc_vc.calculate_escape_velocity(s.effective_radius, s.stellar_mass, s.z_set)

    if s.average_disk_sigma is None:
        #use the threadcount output to find the average disk sigma
        tc_data, _, _ = calc_vc.read_in_threadcount_dict(s.two_gauss_mc_input_file)

        #get the fitted galaxy sigma
        gal_sigma, _, _, _ = calc_sfr.get_arrays(tc_data, var_string='sigma')

        gal_center, _, _, _ = calc_sfr.get_arrays(tc_data, var_string='center')

        #the sigma is in Angstroms, need to convert to km/s
        gal_sigma_vel = calc_vc.sigma_to_vel_disp(gal_sigma, gal_center)

        #take the average
        avg_gal_sigma_vel = np.nanmean(gal_sigma_vel)

        s.average_disk_sigma = avg_gal_sigma_vel

        #get rid of stuff we don't need out of memory
        del tc_data, gal_sigma, gal_center, gal_sigma_vel

    print('escape velocity:', s.escape_velocity)
    print('average disk sigma:', s.average_disk_sigma)

    residuals, vel_cuts_dict = calc_vc.main(
        data_filename = s.data_filename,
        tc_filename = s.two_gauss_mc_input_file,
        baseline_fit_range = s.baseline_fit_range,
        baseline_fit_type = s.baseline_subtract,
        v_esc = s.escape_velocity, disk_sigma=s.average_disk_sigma, line=s.line)

    #print some stuff
    print('Residuals type', type(residuals))
    print('Residuals shape', residuals.shape)
    print('Fountain flux shape', vel_cuts_dict['low_velocity_outflow'].shape)
    print('Escape flux shape', vel_cuts_dict['high_velocity_outflow'].shape)

    #save the results - OVERWRITES ANY EXISTING FILES
    residuals.write(s.output_base_name+'_'+str(s.line.center)+'_residuals.fits')

    vel_cuts_dict.savetxt(s.output_base_name+'_'+str(s.line.center)+'_vel_cuts_dict.txt')

    #run through the plotting scripts
    if s.plot_results == True:
        #read in the threadcount results
        tc_data, wcs_step, z = calc_vc.read_in_threadcount_dict(s.two_gauss_mc_input_file)

        #read in the data file
        cube = calc_vc.fits_read_in(s.data_filename)

        #calculate the noise cube
        #get the spectral pixel indexes of wavelengths
        k1, k2 = cube.wave.pixel([4700, 4800], nearest=True)
        noise = np.nanstd(cube.data[k1:k2+1, :, :], axis=0)

        fig = plot_velocity_cut_maps(vel_cuts_dict['low_velocity_outflow'], vel_cuts_dict['high_velocity_outflow'], tc_data, noise, v_esc=s.escape_velocity, disk_sigma=s.average_disk_sigma, title=s.gal_name+' '+s.line.label, wcs_step=wcs_step, crop_data=s.crop_data)

        plt.show(block=False)



def run_plots_only(user_settings):
    #the inputs that will be used if there are no command line arguments
    default_settings = {
        # need the continuum subtracted data fits file for the velocity cuts method
        "data_filename" : "ex_gal_fits_file.fits",
        "two_gauss_mc_input_file" : "ex_4861_mc_best_fit.txt",
        "gal_name" : "example_galaxy_name", #used for labelling plots
        "z" : 0.03, # redshift
        # now for some default fitting details
        "line" : tc.lines.L_Hb4861,
        "baseline_subtract" : None, # baseline can be "quadratic" or "linear" or None
        # the range of wavelengths to use in the baseline subtraction
        "baseline_fit_range" : [
                            ], # a list of: [[left_begin, left_end],[right_begin, right_end]], one for each line
        # also need the stellar mass of the galaxy
        "stellar_mass" : 10**11.21, # MUST BE INCLUDED
        # we need the escape velocity for the data
        # either this is a given parameter, or threadcount can work it out if
        # you give it data to use to calculate the effective radius
        "escape_velocity" : 456 * units.km/units.s, # put as None if you don't know
        # either give the effective radius, or the threadcount output will
        # be used to find an effective radius, which assumes that the entire
        # galaxy is within the field of view for the IFU data
        "effective_radius" : None, # in arcseconds, used to calculate the escape
        # velocity, so doesn't technically need to be v_50
        # alternatively, you can give another data image (e.g. PANSTARRs) to use
        # to find the effective radius
        "image_data_filename" : "ex_image_file.fits", #or put as None
        # escape_velocity MUST BE NONE IF YOU WANT TO CALCULATE IT FROM THE
        # IMAGE DATA FILE GIVEN ABOVE
        "average_disk_sigma" : None, # if None will use the threadcount fits to find average_disk_sigma.
        # output options
        "output_base_name" : "ex_velocity_cuts_results", # saved files will begin with this
        "plot_results" : True, #boolean
        "crop_data" : None, # Use to define how much of the data goes into the maps in the
        # format: [axis1_begin, axis1_end, axis2_begin, axis2_end] or None
        # e.g. [2, -1, 3, -2] will map data[2:-1, 3:-2]
    }
    # s for settings
    s = tc.fit.process_settings_dict(default_settings, user_settings)

    #read in the data file
    cube = calc_vc.fits_read_in(s.data_filename)

    #calculate the noise cube
    #get the spectral pixel indexes of wavelengths
    k1, k2 = cube.wave.pixel([4700, 4800], nearest=True)
    noise = np.nanstd(cube.data[k1:k2+1, :, :], axis=0)

    #read in velocity cuts results
    #disk_turb_flux = np.loadtxt(s.output_base_name+'_'+str(s.line.center)+'_disk_turbulence_flux.txt')
    #fountain_flux = np.loadtxt(s.output_base_name+'_'+str(s.line.center)+'_fountain_gas_flux.txt')
    #escape_flux = np.loadtxt(s.output_base_name+'_'+str(s.line.center)+'_escaping_gas_flux.txt')

    vel_cuts_dict = tc.fit.ResultDict.loadtxt(s.output_base_name+'_'+str(s.line.center)+'_vel_cuts_dict.txt')

    #read in the threadcount results
    tc_data, wcs_step, z = calc_vc.read_in_threadcount_dict(s.two_gauss_mc_input_file)


    #run through the plotting scripts
    fig = plot_velocity_cut_maps(vel_cuts_dict['low_velocity_outflow'], vel_cuts_dict['high_velocity_outflow'], tc_data, noise, v_esc=s.escape_velocity, disk_sigma=s.average_disk_sigma, title=s.gal_name+' '+s.line.label, wcs_step=wcs_step, crop_data=s.crop_data)

    plt.show(block=False)



def data_coords(gal_dict, z, wcs_step, shiftx=None, shifty=None):
    """
    Takes the data cube and creates coordinate arrays that are centred on the
    galaxy.  The arrays can be shifted manually.  If this is not given to
    the function inputs, the function finds the centre using the maximum continuum
    value.

    Parameters
    ----------
    gal_dict : dictionary
        dictionary with the threadcount results

    z : float
        redshift

    wcs_step : list of floats
        the step for the wcs size of the spaxels in arcseconds

    shiftx : float or None
        the hardcoded shift in the x direction for the coord arrays (in arcseconds).
        If this is none, it finds the maximum point of the median across a section
        of continuum, and makes this the centre.  Default is None.

    shifty : float or None
        the hardcoded shift in the y direction for the coord arrays (in arcseconds).
        If this is none, it finds the maximum point of the median across a section
        of continuum, and makes this the centre.  Default is None.

    Returns
    -------
    xx : :obj:'~numpy.ndarray'
        2D x coordinate array

    yy : :obj:'~numpy.ndarray'
        2D y coordinate array

    rad : :obj:'~numpy.ndarray'
        2D radius array
    """
    #get the data shape
    s = gal_dict['choice'].shape

    #create x and y ranges
    x = np.arange(s[0]) #RA
    y = np.arange(s[1]) #DEC

    #multiply through by wcs_step values
    x = x*wcs_step[0]
    y = y*wcs_step[1]

    print("x shape, y shape:", x.shape, y.shape)

    #shift the x and y
    if None not in (shiftx, shifty):
        x = x + shiftx
        y = y + shifty

    #otherwise use the flux fits to find the centre of the galaxy
    else:
        flux_results, flux_error, outflow_flux, outflow_flux_err = calc_sfr.get_arrays(gal_dict, var_string='flux')

        i, j = np.unravel_index(np.nanargmax(flux_results), flux_results.shape)

        shiftx = i*wcs_step[0]
        shifty = j*wcs_step[1]

        print("shiftx, shifty:", shiftx, shifty)
        x = x - shiftx
        y = y - shifty

    #create x and y arrays
    xx, yy = np.meshgrid(x,y, indexing='ij')

    print("xx shape, yy shape", xx.shape, yy.shape)

    #create radius array
    rad = np.sqrt(xx**2+yy**2)

    return xx, yy, rad

def calc_effective_radius_tc(gal_dict, radius_array, flux_percentage=50):
    """
    Calculates the effective radius (default) of the galaxy using the fitted
    flux, but can also be used to calculate e.g. r_75 or r_90
    """
    #get the galaxy flux array
    gal_flux, gal_flux_err, flow_flux, flow_flux_err = calc_sfr.get_arrays(gal_dict, var_string='flux')

    #get the total flux
    total_flux = np.nansum(gal_flux)

    #get the half flux (or whatever percentage of the flux you wanted)
    effective_flux = total_flux * (flux_percentage/100)
    print('Looking for effective flux:', effective_flux)

    #get the unique radii from the radius array so we can iterate through them
    unique_rad = np.unique(radius_array)

    #iterate through the available radii and add up the flux
    for i, radius in enumerate(unique_rad):
        #calcualte the enclosed flux
        enclosed_flux = np.nansum(gal_flux[radius_array<=radius])

        #calculate the percentage of the total flux enclosed
        percentage_total_flux = (enclosed_flux/total_flux) * 100

        if percentage_total_flux < flux_percentage:
            continue

        elif percentage_total_flux > flux_percentage:
            print('radius:', radius, 'enclosed flux', enclosed_flux)
            print('percentage of total flux', percentage_total_flux)

            #the previous radius is the one we want, since this one gives too
            #much flux
            effective_radius = unique_rad[i-1]
            enclosed_flux = np.nansum(gal_flux[radius_array<=effective_radius])
            percentage_total_flux = (enclosed_flux/total_flux) * 100

            print('effective radius:', effective_radius, 'enclosed flux', enclosed_flux)
            print('final percentage of total flux', percentage_total_flux)

            return effective_radius


def calc_effective_radius_fits(fits_filename, fits_ext='COMPRESSED_IMAGE', flux_percentage=50):
    """
    Reads in an image fits file (e.g. PANSTARRs) and then calculates the
    effective radius (default) of the galaxy using the summed flux, but can
    also be used to calculate e.g. r_75 or r_90
    """
    #read in the fits file
    gal = Image(fits_filename, ext=fits_ext)

    #find the centre of the galaxy by fitting a 2D gaussian model
    gfit = gal.gauss_fit(plot=False)
    gal_center = gfit.center

    #get the data shape
    s = gal.shape

    #create x and y ranges
    x = np.arange(s[0]) #RA
    y = np.arange(s[1]) #DEC

    #multiply through by wcs_step values
    x = x*gal.get_step(unit=units.arcsec)[0]
    y = y*gal.get_step(unit=units.arcsec)[1]

    print("x shape, y shape:", x.shape, y.shape)

    #shift the x and y by the galaxy centre value
    x = x - gal_center[0]
    y = y - gal_center[1]

    #create x and y arrays
    xx, yy = np.meshgrid(x,y, indexing='ij')

    print("xx shape, yy shape", xx.shape, yy.shape)

    #create radius array
    radius_array = np.sqrt(xx**2+yy**2)

    #get the total flux
    total_flux = np.nansum(gal.data)

    #get the half flux (or whatever percentage of the flux you wanted)
    effective_flux = total_flux * (flux_percentage/100)
    print('Looking for effective flux:', effective_flux)

    #get the unique radii from the radius array so we can iterate through them
    #unique_rad = np.unique(radius_array)

    #get the maximum radius
    max_rad = np.nanmax(radius_array)

    #create an array of radii to iterate through
    unique_rad = np.linspace(0, max_rad, 100)

    #iterate through the available radii and add up the flux
    for i, radius in enumerate(unique_rad):
        #calcualte the enclosed flux
        enclosed_flux = np.nansum(gal.data[radius_array<=radius])

        #calculate the percentage of the total flux enclosed
        percentage_total_flux = (enclosed_flux/total_flux) * 100

        if percentage_total_flux < flux_percentage:
            continue

        elif percentage_total_flux > flux_percentage:
            print('radius:', radius, 'enclosed flux', enclosed_flux)
            print('percentage of total flux', percentage_total_flux)

            #the previous radius is the one we want, since this one gives too
            #much flux
            effective_radius = unique_rad[i-1]
            enclosed_flux = np.nansum(gal.data[radius_array<=effective_radius])
            percentage_total_flux = (enclosed_flux/total_flux) * 100

            print('effective radius:', effective_radius, 'enclosed flux', enclosed_flux)
            print('final percentage of total flux', percentage_total_flux)

            return effective_radius

def calc_stellar_mass_surface_density(stellar_mass, radius_eff, z):
    """
    Calculates the stellar mass surface density of a galaxy given a stellar mass
    and an effective radius
    """
    #give mass units
    stellar_mass = stellar_mass * units.solMass

    #give the radius units
    radius_eff = radius_eff * units.arcsec

    #convert to kpc
    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(units.kpc/units.arcsec)
    radius_eff = radius_eff * proper_dist

    #calcuate the stellar mass surface density
    sigma_star = stellar_mass/(4*np.pi*radius_eff**2)

    return sigma_star


def calc_average_disk_height(gal_dict, sigma_star):
    """
    Calculates the average disk height using the velocity dispersion from the
    threadcount fits
    """
    #get the fitted galaxy sigma
    gal_sigma, gal_sigma_err, flow_sigma, flow_sigma_err = calc_sfr.get_arrays(gal_dict, var_string='sigma')

    gal_center, gal_center_err, flow_center, flow_center_err = calc_sfr.get_arrays(gal_dict, var_string='center')

    #the sigma is in Angstroms, need to convert to km/s
    gal_sigma_vel = calc_vc.sigma_to_vel_disp(gal_sigma, gal_center)

    #take the average
    avg_gal_sigma_vel = np.nanmean(gal_sigma_vel)

    #calculate the average disk height
    avg_disk_height = avg_gal_sigma_vel**2/(np.pi*G*sigma_star)

    return avg_disk_height.to('pc')




#-------------------------------------------------------------------------------
# PLOTS
#-------------------------------------------------------------------------------

def plot_velocity_cut_maps(mid_velocity_array, high_velocity_array, gal_dict, noise, v_esc=300, disk_sigma=60, title='Gal Name Line', wcs_step=[1,1], crop_data=None):
    """
    Makes maps of the velocity cuts

    Parameters
    ----------

    crop_data : list of ints or None
        Use to define how much of the data goes into the maps in the format:
        [axis1_begin, axis1_end, axis2_begin, axis2_end]
        e.g. [2, -1, 3, -2] will map data[2:-1, 3:-2]
    """
    #get the flux values
    gal_flux, gal_flux_err, flow_flux, flow_flux_err = calc_sfr.get_arrays(gal_dict, var_string='flux')

    #low_vel_masked = ma.masked_where(low_velocity_array<0, low_velocity_array)
    try:
        mid_vel_masked = ma.masked_where(mid_velocity_array<0, mid_velocity_array.value)
    except AttributeError:
        mid_vel_masked = ma.masked_where(mid_velocity_array<0, mid_velocity_array)
    try:
        high_vel_masked = ma.masked_where(high_velocity_array<0, high_velocity_array.value)
    except AttributeError:
        high_vel_masked = ma.masked_where(high_velocity_array<0, high_velocity_array)

    #crop the data
    if crop_data:
        noise = noise[crop_data[0]:crop_data[1], crop_data[2]:crop_data[3]]
        gal_flux = gal_flux[crop_data[0]:crop_data[1], crop_data[2]:crop_data[3]]
        flow_flux = flow_flux[crop_data[0]:crop_data[1], crop_data[2]:crop_data[3]]
        mid_vel_masked = mid_vel_masked[crop_data[0]:crop_data[1], crop_data[2]:crop_data[3]]
        high_vel_masked = high_vel_masked[crop_data[0]:crop_data[1], crop_data[2]:crop_data[3]]

    #do a S/N check
    #mid_vel_masked



    #fig, ax = plt.subplots(1, 3, figsize=(9,3), constrained_layout=True)
    fig = plt.figure(figsize=(10,2.5), constrained_layout=True)

    #ax1 = plt.subplot(131)
    #im = ax1.imshow(np.log10(gal_flux.T), origin='lower', aspect=wcs_step[1]/wcs_step[0], cmap='viridis', vmax=4)
    #ax1.contour(np.log10(gal_flux).T, levels=7, origin='lower', colors='k', alpha=0.8)
    #ax1.set_title('Galaxy Component Flux', fontsize='small')
    #cbar = plt.colorbar(im, ax=ax1, shrink=0.6)
    #cbar.set_label('Log Flux [10$^{-16}$ erg/(cm2 s)]', fontsize='small')

    ax1 = plt.subplot(131)
    #im = ax1.imshow(np.log10(flow_flux.T), origin='lower', aspect=wcs_step[1]/wcs_step[0], cmap=cmr.cosmic, vmax=2.2)
    im = ax1.imshow(np.log10(flow_flux.T), origin='lower', aspect=wcs_step[1]/wcs_step[0], cmap='viridis', vmax=2.2, vmin=-0.5)
    ax1.contour(np.log10(gal_flux).T, levels=[0.6, 1.0, 1.5], origin='lower', colors='k', alpha=0.8)
    ax1.invert_xaxis()
    ax1.set_title('Broad Component Flux', fontsize='small')
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Log Flux [10$^{-16}$ erg/(cm2 s)]', fontsize='small')


    ax2 = plt.subplot(132)
    im = ax2.imshow(np.log10(mid_vel_masked.T), origin='lower', aspect=wcs_step[1]/wcs_step[0], cmap='viridis', vmax=2.2, vmin=-0.5)
    ax2.contour(np.log10(gal_flux).T, levels=[0.6, 1.0, 1.5], origin='lower', colors='k', alpha=0.8)
    ax2.invert_xaxis()
    ax2.set_title('$\sigma_{disk}$ < v < $v_{esc}$', fontsize='small')
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Log Flux [10$^{-16}$ erg/(cm2 s)]', fontsize='small')


    ax3 = plt.subplot(133)
    im = ax3.imshow(np.log10(high_vel_masked.T), origin='lower', aspect=wcs_step[1]/wcs_step[0], cmap='viridis', vmax=2.2, vmin=-0.5)
    ax3.contour(np.log10(gal_flux).T, levels=[0.6, 1.0, 1.5], origin='lower', colors='k', alpha=0.8)
    ax3.invert_xaxis()
    ax3.set_title('v > $v_{esc}$', fontsize='small')
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Log Flux [10$^{-16}$ erg/(cm2 s)]', fontsize='small')


    plt.suptitle(title)

    return fig
