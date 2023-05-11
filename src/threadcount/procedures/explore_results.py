import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import threadcount as tc


def run(user_settings):
    '''
    This funtion put everything together.
    '''
    default_settings = {
        
        "input_file": "ex3_full_5007_mc_best_fit.txt",
        "data_filename": None,
        "data_hdu_index": 0,
        "line": tc.lines.L_OIII5007,
        "plot_map": None, 
        "plot_map_log": True
    }
    s = tc.fit.process_settings_dict(default_settings, user_settings)  # s for settings.

    #read txt file, output of threadcount 
    input_data = tc.fit.ResultDict.loadtxt(s.input_file)

    #open the header of the txt file and print
    with open(s.input_file) as input_file:
        head = [next(input_file) for _ in range(14)]
    print(*head, sep = "\n")

    print('Click on a spaxel to plot a new Spectrum or move with the keyboard arrows')

    #if data_filename is none, read file phath from the header of txt file
    if s.data_filename == None:
        s.data_filename = head[0].split()[-1]

    #get z and region average radius from txt file
    s.z_set = float(head[1].split()[-1])
    s.region_averaging_radius = float(head[6].split()[-1])
    
    #from the colum names in txt file get how many gaussians was fitted and if it is mc results or not
    columns = head[-1]
    nr_gauss = columns.count('flux_err')
    mc = columns.count('avg')
    
    # if plot_image is None, get the flux of the highes gaussian in each spaxel to do the map 
    if s.plot_map is None:
        if mc > 0 :
            flux = np.empty(input_data["avg_g1_flux"].shape)
            flux[:] = np.nan
            for i in range(nr_gauss):
                flux_i = input_data["avg_g%i_flux" %(i+1) ]
                flux = np.fmax(flux, flux_i) #selects the higher flux of all fitted gaussiand to do the flux map 

        else:
            flux = np.empty(input_data["g1_flux"].shape)
            flux[:] = np.nan
            for i in range(nr_gauss):
                flux_i = input_data["g%i_flux" %(i+1) ]
                flux = np.fmax(flux, flux_i)

        s.plot_map = np.asarray(flux)
  
    #smooth the data and select only the wavelenght range of the wanted line
    region_pixels = tc.fit.get_region(s.region_averaging_radius)
    k = tc.fit.get_reg_image(region_pixels)
    data_cube = tc.fit.open_fits_cube(s.data_filename, data_hdu_index=s.data_hdu_index)
    subcube = data_cube.select_lambda(s.line.low*(1+s.z_set), s.line.high*(1+s.z_set))
    subcube_av = tc.fit.spatial_average(subcube, k)

    #defin p and q, these variables tells us the position x,y in the map
    p_max, q_max = s.plot_map.shape[1], s.plot_map.shape[0]
    global p, q
    p, q = 0, 0

    #create the figure and plot the map

    fig = plt.figure(figsize=(15, 5))
    gs0 = gridspec.GridSpec(1,5, figure=fig)
    ax1 = fig.add_subplot(gs0[0:3])
    ax2 = fig.add_subplot(gs0[3:])
    if s.plot_map_log:
        map = ax1.imshow(s.plot_map, norm=LogNorm(), aspect='auto', origin='lower')
    else:
        map = ax1.imshow(s.plot_map, aspect='auto', origin='lower')
    plt.colorbar(map, ax=ax1,location = 'bottom')
    
    #conect the figure with click_func and arrow_func. These functions define how p and q change when click on a spaxel or using the arrows in keyboard 

    fig.canvas.mpl_connect('button_press_event', lambda event: click_func(event, ax1, ax2, input_data, s, subcube_av, nr_gauss, mc))
    fig.canvas.mpl_connect('key_press_event', lambda event: arrow_func(event, ax1, ax2, input_data, s, subcube_av, p_max, q_max, nr_gauss, mc))
    fig.tight_layout()
    plt.show()
    plt.draw()
    
    


def click_func(event, ax1, ax2, input_data, s, scube,  nr_gauss, mc):
    #defines what happens when you click on a spaxel
    try: # use try/except in case we are not using Qt backend
        zooming_panning = (fig.canvas.cursor().shape() != 0 ) # 0 is the arrow, which means we are not zooming or panning.
    except:
        zooming_panning = False
    if zooming_panning:
        print("Zooming or panning")
        return
    if event.inaxes is not None:
        ax = event.inaxes
        if ax == ax1:
            global p, q
            p = round(event.xdata)
            q = round(event.ydata)
            plot_map(ax1, s, p,q)
            plot_spec(ax2, input_data, scube, p, q, nr_gauss,mc, s.line, s.z_set)


          

def arrow_func(event, ax1, ax2, input_data, s, scube, p_max, q_max, nr_gauss,mc):
    if event.key == 'up' or 'down' or 'left' or 'right': #check that you pressed and arrow on the keyboard
        global p, q
        if event.key == 'up' and q < q_max-1:
            q += 1
        elif event.key == 'up' and q >= q_max-1:
            q = 0

        elif event.key == 'down' and q >= 1:
            q -= 1 
        elif event.key == 'down' and q < 1:
            q = q_max -1

        elif event.key == 'right' and p < p_max-1 :
            p += 1
        elif event.key == 'right' and p >= p_max-1 :
            p = 0

        elif event.key == 'left' and p>= 1:
            p -= 1
        elif event.key == 'left' and p < 1:
            p = p_max -1
        
        plot_map(ax1,s, p,q)
        plot_spec(ax2, input_data, scube, p, q, nr_gauss,mc, s.line, s.z_set)


def plot_map(ax1, s, p,q): 
    ax1.cla() 
    if s.plot_map_log:
        map = ax1.imshow(s.plot_map, norm=LogNorm(), aspect='auto', origin='lower')
    else:
        map = ax1.imshow(s.plot_map, aspect='auto', origin='lower')
    plt.draw()
    ax1.scatter(p,q, color='r')
    plt.draw()

def plot_spec(ax2, input_data, scube, p, q, nr_gauss,mc, line, z_set ):
    #function that does the ploting 
    ax2.cla()
    plt.draw()
    a = p
    b = q           
    spe = scube[:,b,a]
    lam = np.linspace(line.low, line.high, len(spe.data))
    lam_plot = lam*(1+z_set)
    fit = []
    fluxes = []

    if mc > 0:
        const = input_data["avg_c"][b,a]
        snr = input_data["snr"][b,a]
        for i in range(nr_gauss):

            sigma_i = input_data["avg_g%i_sigma" %(i+1) ][b,a]
            center_i = input_data["avg_g%i_center" %(i+1) ][b,a]
            height_i = input_data["avg_g%i_height" %(i+1) ][b,a]
            flux_i = input_data["avg_g%i_flux" %(i+1) ][b,a]
            fit_i = tc.models.gaussianH(lam, height_i, center_i, sigma_i)
            fit.append(fit_i)
            fluxes.append(flux_i)
    else:
        const = input_data["c"][b,a]
        snr = input_data["snr"][b,a]
        for i in range(nr_gauss):
            sigma_i = input_data["g%i_sigma" %(i+1) ][b,a]
            center_i = input_data["g%i_center" %(i+1) ][b,a]
            height_i = input_data["g%i_height" %(i+1) ][b,a]
            flux_i = input_data["g%i_flux" %(i+1) ][b,a]
            fit_i = tc.models.gaussianH(lam, height_i, center_i, sigma_i)
            fit.append(fit_i)
            fluxes.append(flux_i)
    
    ax2.text(0.05, 0.95, 'pixel: ' + str(input_data["row"][b,a]) + ',' + str(input_data["col"][b,a]), transform=ax2.transAxes)
    ax2.text(0.05, 0.9, 'snr: ' + str(snr), transform=ax2.transAxes)

    ax2.step(lam_plot, spe.data, c='black', where='pre')
    for i in range(len(fit)):
        ax2.plot(lam_plot, fit[i]+const, linewidth=3, alpha=0.8)
    #ax2.plot(lam2, fit2+const, linewidth=3, alpha=0.8)
            
    plt.draw()