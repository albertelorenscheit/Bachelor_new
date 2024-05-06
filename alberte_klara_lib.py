import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from chaosmagpy.model_utils import design_gauss, power_spectrum
from mpl_toolkits.axes_grid1 import AxesGrid


# Global variables
n_int_max = 13
n_ext_max = 1
radius_Earth = 6378
radius_CMB = 3485

theta_grid = np.arange(0.5, 180.5, 1)
theta_grid = np.tile(theta_grid, 40320 // len(theta_grid) + 1)[:40320]
phi_grid = np.linspace(-179.4, 180.5, 180)
phi_grid = np.repeat(phi_grid, 224)

SAA_extent = [[80, 140], [-90, 60]] # SAA in coodinates (colat) (lat, lon)

fig_path = 'figs/'

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Author: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# constructing int and ext design matrices combined
def do_design(radius, colat, lon, n_int_max, n_ext_max):
    G_radius, G_theta, G_phi = design_gauss(radius, colat, lon, nmax=n_int_max)

    if (n_ext_max != False) and (n_ext_max != 0):
        G_ext_radius, G_ext_theta, G_ext_phi = design_gauss(radius, colat, lon, nmax=n_ext_max, source = 'external')
        G_radius = np.hstack((G_radius, G_ext_radius))
        G_theta = np.hstack((G_theta, G_ext_theta))
        G_phi = np.hstack((G_phi, G_ext_phi))

    return G_radius, G_theta, G_phi


def do_design_timetrend(radius, colat, lon, n_int_max, n_ext_max, time_grad_array):
    """
    Takes (r, theta, phi)-coordinates and truncation degrees to construct design matrices including a linear time trend.

    radius int or array
    colat array
    lon array
    n_int_max int
    n_ext_max int
    time_grad array of temporal gradient at every synthetic data point. can be constructed by np.linspace(min(data['time_grad'].values), max(data['time_grad'].values), n_evals)

    """
    nm_internal = (n_int_max + 1)**2 - 1
    n_evals = len(colat)

    G_int_radius, G_int_theta, G_int_phi = design_gauss(radius, colat, lon, nmax=n_int_max)

    if (n_ext_max != False) and (n_ext_max != 0):
        G_ext_radius, G_ext_theta, G_ext_phi = design_gauss(radius, colat, lon, nmax=n_ext_max, source = 'external')
        G_radius = np.hstack((G_int_radius, G_ext_radius))
        G_theta = np.hstack((G_int_theta, G_ext_theta))
        G_phi = np.hstack((G_int_phi, G_ext_phi))
    else:
        G_radius = G_int_radius
        G_theta = G_int_theta
        G_phi = G_int_phi
    
    
    G_radius = np.hstack((G_radius, G_int_radius))
    G_theta = np.hstack((G_theta, G_int_theta))
    G_phi = np.hstack((G_phi, G_int_phi))

    time_grad = np.ones_like(G_radius)
    time_grad[:, -nm_internal:] = time_grad_array.reshape(n_evals, 1)

    G_radius *= time_grad
    G_theta *= time_grad
    G_phi *= time_grad

    return G_radius, G_theta, G_phi


# plotting on map with equal Earth and polar views (three views)
def plot_map_three(map_data, lon, colat, sat: str, title: str, label: str, scatter_size = 70, colors = 'jet', clim = None, save = False, filename = None):
    """
    Scatters points in x and y on Earth map by corresponding marker colors in c
                                                                                                                            # Credits to Chris (?)
    
    lon longitunidal cooridnates
    colat co-latitudinal coordinates
    sat satllite (either 'Swarm' or 'Oersted')
    title title of plot
    label label on color bar
    scatter_size scatter size
    colors color scheme
    clim limit on color bar, list [start, end], default is None


    In latex, trim as:
        \includegraphics[width=01\textwidth, trim = {6cm 5cm 0cm 3cm},clip]
    """

    # Ensure satellite declaration
    sat = sat.lower()  # Convert to lowercase for case-insensitive comparison
    if sat not in ['swarm', 'oersted', 'ørsted']:
        raise ValueError('Satellite must be defined: either \'Swarm\' or \'Ørsted\'')

    fig = plt.figure(figsize=(25, 25))
    gs = fig.add_gridspec(2, 2)

    axes = []
    axes.append(plt.subplot(gs[1, 0], projection=ccrs.Orthographic(0,90)))
    axes.append(plt.subplot(gs[1, 1], projection=ccrs.Orthographic(180,-90)))
    axes.append(plt.subplot(gs[0, :], projection=ccrs.EqualEarth()))

    # Titles 
    if (sat == 'Swarm') or (sat == 'swarm'):
        plt.title(title + ' \n Swarm data January-December 2023 \n \n', weight='bold', fontsize=20)

    elif (sat == 'Oersted') or (sat == 'oersted') or (sat == 'Ørsted') or (sat == 'ørsted'):
        plt.title(title + ' \n Ørsted data January-December 2001 \n \n', weight='bold', fontsize=20)


    for ax in axes:
            pc = ax.scatter(x = lon, y = 90 - colat, s = scatter_size, c = map_data, edgecolors='none', cmap = colors, transform = ccrs.PlateCarree())
            ax.coastlines(linewidth=1.1)
    
    cax = inset_axes(axes[-1], width="3%", height="100%", loc='right', borderpad=-11)
    clb = plt.colorbar(pc, cax=cax)
    clb.ax.tick_params(labelsize=20)
    clb.set_label(label, rotation=270, labelpad=25, fontsize=20)

    # Adjust color bar if limits are given
    if clim != None:
        pc.set_clim(vmin = clim[0], vmax = clim[1])
        if min(map_data) >= 0:
            clb = plt.colorbar(pc, cax=cax, extend = 'max')
        else:
            clb = plt.colorbar(pc, cax=cax, extend = 'both')

    if save:
        if filename == None:
            raise ValueError('File title must be defined')
        
        plt.savefig(fig_path + filename + '.png', format = 'png', dpi = 200)

    plt.show()

    return


# plotting on map with equal Earth (one view)
def plot_map_one(map_data, lon, colat, sat, title, label, scatter_size = 70, colors = 'jet', clim = None, save = False, filename = None):
    """
    Scatters points in x and y on Earth map by corresponding marker colors in c
                                                                                                                            # Credits to Chris (?)
    
    lon longitunidal cooridnates
    colat co-latitudinal coordinates
    sat satllite (either 'Swarm' or 'Oersted')
    title title of plot
    label label on color bar
    scatter_size scatter size
    colors color scheme
    clim limit on color bar, list [start, end], default is None
    """

    # Ensure satellite declaration
    sat = sat.lower()  # Convert to lowercase for case-insensitive comparison
    if sat not in ['swarm', 'oersted', 'ørsted']:
        raise ValueError('Satellite must be defined: either \'Swarm\' or \'Ørsted\'')

    fig = plt.figure(figsize=(25, 25))

    ax = plt.subplot(projection=ccrs.EqualEarth())
    
    # Titles
    if (sat == 'Swarm') or (sat == 'swarm'):
        plt.title(title + ' \n Swarm data January-December 2023 \n \n', weight='bold', fontsize=20)

    elif (sat == 'oersted') or (sat == 'ørsted'):
        plt.title(title + ' \n Ørsted data January-December 2001 \n \n', weight='bold', fontsize=20) 
    

    pc = ax.scatter(x = lon, y = 90 - colat, s = scatter_size, c = map_data, edgecolors='none', cmap = colors, transform = ccrs.PlateCarree())
    ax.coastlines(linewidth=1.1)

    cax = inset_axes(ax, width="3%", height="100%", loc='right', borderpad=-11)
    clb = plt.colorbar(pc, cax=cax)
    clb.ax.tick_params(labelsize=20)
    clb.set_label(label, rotation=270, labelpad=25, fontsize=20)

    # Adjust color bar if limits are given
    if clim != None:
        pc.set_clim(vmin = clim[0], vmax = clim[1])
        if min(map_data) >= 0:
            clb = plt.colorbar(pc, cax=cax, extend = 'max')
        else:
            clb = plt.colorbar(pc, cax=cax, extend = 'both')

    if save:
        if filename == None:
            raise ValueError('File title must be defined')
        
        plt.savefig(fig_path + filename + '.png', format = 'png', dpi = 200, bbox_inches='tight')


    plt.show()

    return

# compute Gauss coefficients
def get_gausscoeff_timetrend(dataframe, n_int_max, n_ext_max):
    data = dataframe.copy()
    n_obs = len(data)
    nm = ((n_int_max + 1)**2 - 1)*2 + (n_ext_max + 1)**2 - 1

    # lhs and rhs of lst sq prb will have to be constructed bit by bit
    # allocating these in memory
    lhs = np.zeros((nm, nm)) # G.T@G
    rhs = np.zeros((nm,)) # G.T@d

    # taking 10k data point at a time (sub matrix, chunk of data, G_sub)
    i = 0
    while (i <= n_obs - 10*10**3):
        i10k = i + 10*10**3

        G_int_static = np.vstack((design_gauss(data['radius'].iloc[i:i10k], data['colat'].iloc[i:i10k], data['lon'].iloc[i:i10k], nmax=n_int_max)))
        G_ext = np.vstack((design_gauss(data['radius'].iloc[i:i10k], data['colat'].iloc[i:i10k], data['lon'].iloc[i:i10k], nmax=n_ext_max, source = 'external')))
        G_int_sec = ((np.vstack((design_gauss(data['radius'].iloc[i:i10k], data['colat'].iloc[i:i10k], data['lon'].iloc[i:i10k], 
                                            nmax=n_int_max))).T)*np.tile(data['time_grad'].iloc[i:i10k], 3)).T
        G = np.hstack((G_int_static, G_ext, G_int_sec))
        lhs += G.T@G

        d = np.hstack((data['B_radius'].iloc[i:i10k], data['B_theta'].iloc[i:i10k], data['B_phi'].iloc[i:i10k]))
        rhs += G.T@d
        i += 10*10**3

    G_int_static = np.vstack((design_gauss(data['radius'].iloc[i:], data['colat'].iloc[i:], data['lon'].iloc[i:], nmax=n_int_max)))
    G_ext = np.vstack((design_gauss(data['radius'].iloc[i:], data['colat'].iloc[i:], data['lon'].iloc[i:], nmax=n_ext_max, source = 'external')))
    G_int_sec = ((np.vstack((design_gauss(data['radius'].iloc[i:], data['colat'].iloc[i:], data['lon'].iloc[i:], nmax=n_int_max))).T)*np.tile(data['time_grad'].iloc[i:], 3)).T
    G = np.hstack((G_int_static, G_ext, G_int_sec))
    lhs += G.T@G

    d = np.hstack((data['B_radius'].iloc[i:], data['B_theta'].iloc[i:], data['B_phi'].iloc[i:]))
    rhs += G.T@d

    m = np.linalg.solve(lhs, rhs)
    return m


# plot histogram of residuals
def plot_hist_withzoom(residuals_r, residuals_t, residuals_p, r_ylim, t_ylim, p_ylim, sat, additional_title = None, save = False, filename = None):
    """
    Plots histograms of residuals for radial, co-latitudinal and longitudinal components.
    Residuals are constrained to a certain range for better visualization.
    """


    sat = sat.lower()  # Convert to lowercase for case-insensitive comparison
    if sat not in ['swarm', 'oersted', 'ørsted']:
        raise ValueError('Satellite must be defined: either \'Swarm\' or \'Ørsted\'')


    # Plot histogram of residuals
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # Plot histogram of residuals_r
    axs[0, 0].hist(residuals_r, bins=100, color='blue', edgecolor='black', alpha=0.7)
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Radial component residuals \nRMS = ' + str(np.sqrt(np.mean(residuals_r**2))))
    axs[0, 0].grid(True)

    # Plot histogram of residuals_t
    axs[0, 1].hist(residuals_t, bins=100, color='green', edgecolor='black', alpha=0.7)
    axs[0, 1].set_title('Co-latitudinal component residuals \nRMS = ' + str(np.sqrt(np.mean(residuals_t**2))))
    axs[0, 1].grid(True)

    # Plot histogram of residuals_p
    axs[0, 2].hist(residuals_p, bins=100, color='red', edgecolor='black', alpha=0.7)
    axs[0, 2].set_title('Longitudinal component residuals \nRMS = ' + str(np.sqrt(np.mean(residuals_p**2))))
    axs[0, 2].grid(True)

    # Plot histogram of residuals_r (constrained)
    axs[1, 0].hist(residuals_r, bins=100, color='blue', edgecolor='black', alpha=0.7)
    axs[1, 0].set_xlabel('Residuals')
    axs[1, 0].set_ylabel('Frequency (constrained)')
    axs[1, 0].set_ylim(0, r_ylim)
    axs[1, 0].grid(True)

    # Plot histogram of residuals_t (constrained)
    axs[1, 1].hist(residuals_t, bins=100, color='green', edgecolor='black', alpha=0.7)
    axs[1, 1].set_xlabel('Residuals')
    axs[1, 1].set_ylim(0, t_ylim)
    axs[1, 1].grid(True)

    # Plot histogram of residuals_p (constrained)
    axs[1, 2].hist(residuals_p, bins=100, color='red', edgecolor='black', alpha=0.7)
    axs[1, 2].set_xlabel('Residuals')
    axs[1, 2].set_ylim(0, p_ylim)
    axs[1, 2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    if additional_title != None:
        if (sat == 'ørsted') or (sat == 'oersted'):
            fig.suptitle(additional_title + '\nResiduals between measured field and predicted field by component\n Ørsted data January-December 2001', weight = 'bold', fontsize = 15)
        elif (sat == 'swarm'):
            fig.suptitle(additional_title + '\nResiduals between measured field and predicted field by component\n Swarm data January-December 2023', weight = 'bold', fontsize = 15)
    else:
        if (sat == 'ørsted') or (sat == 'oersted'):
            fig.suptitle('Residuals between measured field and predicted field by component\n Ørsted data January-December 2001', weight = 'bold', fontsize = 15)
        elif (sat == 'swarm'):
            fig.suptitle('Residuals between measured field and predicted field by component\n Swarm data January-December 2023', weight = 'bold', fontsize = 15)

    if save:
        if filename == None:
            raise ValueError('File title must be defined')
        
        plt.savefig(fig_path + filename + '.png', format = 'png', dpi = 200)

    plt.show()

    return


def plot_difference_F(dataframe, r, theta, phi, m, n_int_max, n_ext_max, sat, external_source = False, save = False, filename = None):

    sat = sat.lower()  # Convert to lowercase for case-insensitive comparison
    if sat not in ['swarm', 'oersted', 'ørsted']:
        raise ValueError('Satellite must be defined: either \'Swarm\' or \'Ørsted\'')

    # nm = ((n_int_max + 1)**2 - 1)*2 + (n_ext_max + 1)**2 - 1
    data = dataframe.copy()
    theta_grid = theta.copy()
    phi_grid = phi.copy()

    nm_internal = (n_int_max + 1)**2 - 1

    if external_source: print('External source not implemented yet')
    else:
        Gr_start, Gt_start, Gp_start = do_design_timetrend(r, theta_grid, phi_grid, n_int_max, 0, np.tile(min(data['time_grad'].values), len(theta_grid)))
        Gr_end, Gt_end, Gp_end = do_design_timetrend(r, theta_grid, phi_grid, n_int_max, 0, np.tile(max(data['time_grad'].values), len(theta_grid)))

        br_start = Gr_start @ np.hstack((m[:nm_internal], m[-nm_internal:])) # Only internal parameters
        bt_start = Gt_start @ np.hstack((m[:nm_internal], m[-nm_internal:]))
        bp_start = Gp_start @ np.hstack((m[:nm_internal], m[-nm_internal:]))

        F_start = np.sqrt(br_start**2 + bt_start**2 + bp_start**2)

        br_end = Gr_end @ np.hstack((m[:nm_internal], m[-nm_internal:]))
        bt_end = Gt_end @ np.hstack((m[:nm_internal], m[-nm_internal:]))
        bp_end = Gp_end @ np.hstack((m[:nm_internal], m[-nm_internal:]))

        F_end = np.sqrt(br_end**2 + bt_end**2 + bp_end**2)

        F = F_end - F_start

        if r == (radius_Earth + 450):
            plot_title = '450km altitude'
        elif r == radius_Earth:
            plot_title = 'surface of Earth'
        elif r == radius_CMB:
            plot_title = 'CMB'

        if save:
            if filename == None:
                raise ValueError('File title must be defined')
            
            plot_map_one(F, phi_grid, theta_grid, sat, 'Difference in field intensity (F(end)-F(start)) at ' + plot_title + ' \nn = ' + str(n_int_max), '[nT]', save = True, filename = filename)
    
        else:
            plot_map_one(F, phi_grid, theta_grid, sat, 'Difference in field intensity (F(end)-F(start)) at ' + plot_title + ' \nn = ' + str(n_int_max), '[nT]')

    return


def plot_power_spectrum(m, n_int_max, r_eval, sat, static = True, save = False, filename = None):

    sat = sat.lower()  # Convert to lowercase for case-insensitive comparison
    if sat not in ['swarm', 'oersted', 'ørsted']:
        raise ValueError('Satellite must be defined: either \'Swarm\' or \'Ørsted\'')

    nm_internal = (n_int_max + 1)**2 - 1

    if r_eval == (radius_Earth + 450):
        plot_title = '450km altitude'
    elif r_eval == radius_Earth:
        plot_title = 'surface of Earth'
    elif r_eval == radius_CMB:
        plot_title = 'CMB'

    if static:
        plt.semilogy(range(n_int_max), power_spectrum(m[:nm_internal], radius = r_eval))
        plt.scatter(range(n_int_max), power_spectrum(m[:nm_internal], radius = r_eval))

        title = 'Power spectrum of internal, static coefficients truncated at n = '

        # Titles
        if (sat == 'swarm'):
            plt.title(title + str(n_int_max) + ' at ' + plot_title + ' \n Swarm data January-December 2023')

        elif (sat == 'oersted') or (sat == 'ørsted'):
            plt.title(title + str(n_int_max) + ' at ' + plot_title + ' \n Ørsted data January-December 2001')

        if r_eval == radius_Earth:
            plt.ylabel('Wn(a) [nT^2]')
        elif r_eval == radius_Earth + 450:
            plt.ylabel('Wn(a + 450km) [nT^2]')
        elif r_eval == radius_CMB:
            plt.ylabel('Wn(CMB) [nT^2]')
        else: 
            plt.ylabel('Wn(' + str(r_eval) + 'km) [nT^2]')

    else:
        plt.semilogy(range(n_int_max), power_spectrum(m[-nm_internal:]*365.25, radius = r_eval))
        plt.scatter(range(n_int_max), power_spectrum(m[-nm_internal:]*365.25, radius = r_eval))

        title = 'Power spectrum of secular coefficients truncated at n = '

        # Titles
        if (sat == 'swarm'):
            plt.title(title + str(n_int_max) + ' at ' + plot_title + ' \n Swarm data January-December 2023')

        elif (sat == 'oersted') or (sat == 'ørsted'):
            plt.title(title + str(n_int_max) + ' at ' + plot_title + ' \n Ørsted data January-December 2001')

        if r_eval == radius_Earth:
            plt.ylabel('Wn(a) [nT^2/yr]')
        elif r_eval == radius_Earth + 450:
            plt.ylabel('Wn(a + 450km) [nT^2/yr]')
        elif r_eval == radius_CMB:
            plt.ylabel('Wn(CMB) [nT^2/yr]')
        else: 
            plt.ylabel('Wn(' + str(r_eval) + 'km) [nT^2/yr]')
    
 
    plt.xlabel('n')
    plt.xticks(range(n_int_max), range(n_int_max + 1)[1:])
    plt.grid()

    if save:
        if filename == None:
            raise ValueError('File title must be defined')
        
        plt.savefig(fig_path + filename + '.png', format = 'png', dpi = 200)

    plt.show()

    return


