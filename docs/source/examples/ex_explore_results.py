import threadcount as tc
from threadcount.procedures import (  # noqa: F401
    explore_results,
)


#example file to run explore_results.py
#It will display a figure with a map on the left where you can click on a pixel or move arround with keyboard arrows and it will 
#display an emission line with its corresponding fit from threadcount on the right figure
#It assumes you already run ex2.py script, but you can run it with any threadcount output file 


# This is to create a numpy array with the values of the map you want to display. It can be anything as long as is the same shape of threadcount results.
# Here I am just reading directly from threadcount output file 
# If you dont input any value for the map, the explore_results script will show a map with the flux of the highes gaussian fit in each spaxel 

input_data = tc.fit.ResultDict.loadtxt('ex3_full_5007_mc_best_fit.txt')
map = input_data["avg_g1_flux"]


#these are the necessary inputs to run the explore_results script

my_settings = {
    "input_file": 'ex3_full_5007_mc_best_fit.txt', # Threadcount output file, can be any threadcout output txt file
    "data_filename": None, # file to read the spectra, should be the same fits file you use for the threadcount fit. if set to None it will get the file from the header of the input_file.
    "data_hdu_index": 0, # the hdu index from where to read the spectral data of data_filename
    "line": tc.lines.L_OIII5007, #the line you want to plot, should be a tc.lines object, the sabe you used for the fit 
    "plot_map": map, # an array with values of the map to display, must have the same spatial shape as the data cube. 
    #If None it will show a map of the highest fitted gaussian in each spaxel.
    "plot_map_log": True # True to show the map in log scale, False to show it in linear scale 
}


explore_results.run(my_settings)