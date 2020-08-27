from matplotlib import pyplot as plt
import numpy as np
import bjlib.lib_project as lib
from matplotlib import cm
from numpy import inf
import matplotlib.ticker as mticker
from cycler import cycler
import matplotlib as mpl


def cl_comparison(cl_unchanged, cl_rot, angle):

    ls = np.arange(cl_unchanged.shape[0])
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))

    ax[0, 0].plot(ls, cl_unchanged[:, 0], color='k',
                  label='original power spectra')
    ax[0, 0].plot(ls, cl_rot[:, 0], color='b',
                  label='power spectra rotated by {}'.format(angle))
    ax[0, 0].set_title('TT')
    ax[0, 0].legend(borderaxespad=0., loc='upper center',
                    bbox_to_anchor=(0.5, 1.5))

    ax[0, 1].plot(ls, cl_unchanged[:, 1], color='k', linewidth=2)
    ax[0, 1].plot(ls, cl_rot[:, 1], color='b')
    ax[0, 1].set_title(r'$EE$')

    ax[1, 0].plot(ls, cl_unchanged[:, 2], color='k', linewidth=2)
    ax[1, 0].plot(ls, cl_rot[:, 2], color='b')
    ax[1, 0].set_title(r'$BB$')

    ax[1, 1].plot(ls, cl_unchanged[:, 3], color='k', linewidth=2)
    ax[1, 1].plot(ls, -cl_unchanged[:, 3], '--', color='k', linewidth=2)
    ax[1, 1].plot(ls, cl_rot[:, 3], color='b')
    ax[1, 1].plot(ls, -cl_rot[:, 3], '--', color='b')
    ax[1, 1].set_title(r'$TE$')

    ax[2, 0].plot(ls, cl_rot[:, 4], color='b')
    ax[2, 0].set_title(r'$EB$')

    ax[2, 1].plot(ls, cl_rot[:, 5], color='b')
    ax[2, 1].plot(ls, -cl_rot[:, 5],  '--', color='b')
    ax[2, 1].set_title(r'$TB$')
    for ax_ in ax.reshape(-1):
        ax_.set_xlim([2, cl_unchanged.shape[0]])
        ax_.set_xscale('log')
        ax_.set_yscale('log')
        return fig, ax


def new_angle(cl_rot, angle, fig, ax):
    ls = np.arange(cl_rot.shape[0])
    ax[0, 0].plot(ls, cl_rot[:, 0],
                  label='power spectra rotated by {}'.format(angle))
    ax[0, 0].legend(borderaxespad=0., loc='upper center',
                    bbox_to_anchor=(0.5, 1.5))
    ax[0, 1].plot(ls, cl_rot[:, 1])
    ax[1, 0].plot(ls, cl_rot[:, 2])
    color = next(ax[1, 1]._get_lines.prop_cycler)['color']
    ax[1, 1].plot(ls, cl_rot[:, 3], color=color)
    ax[1, 1].plot(ls, -cl_rot[:, 3], '--', color=color)
    ax[2, 0].plot(ls, cl_rot[:, 4])
    ax[2, 1].plot(ls, cl_rot[:, 5], color=color)
    ax[2, 1].plot(ls, -cl_rot[:, 5],  '--', color=color)
    for ax_ in ax.reshape(-1):
        ax_.set_xlim([2, cl_rot.shape[0]])
        ax_.set_xscale('log')
        ax_.set_yscale('log')
        return fig, ax


def error_on_angle_wrt_scale(fisher_element, label=None):
    ls = np.arange(2, len(fisher_element))
    print('1 / np.sqrt(fisher_element[2:]', 1/np.sqrt(fisher_element[2:]))
    plt.plot(ls, 1 / np.sqrt(fisher_element[2:]), label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\ell$', fontsize=16)
    plt.ylabel(r'$ \frac{1}{\sqrt{F_{\ell}}} $', fontsize=16)
    plt.legend()

    return 0


def cumulative_error(fisher_element, label=None, l_min=2, color=None,
                     dotted=False):
    ls = np.arange(l_min, len(fisher_element))  # 2)

    cumulative = np.array([1 / np.sqrt(sum(fisher_element[l_min:k]))
                           for k in range(l_min, len(fisher_element))])  # 2)])

    if color is None:
        if dotted is False:
            plt.plot(ls, cumulative, label=label)
        else:
            plt.plot(ls, cumulative, '--', label=label)

    else:
        if dotted is False:
            plt.plot(ls, cumulative, label=label, color=color)
        else:
            plt.plot(ls, cumulative, '--', label=label, color=color)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\ell$', fontsize=16)
    plt.ylabel(r'$ \frac{1}{\sqrt{ \sum_{\ell_{min} = 2}^{\ell}{F_{\ell}}}} $',
               fontsize=16)
    # plt.legend()
    return 0


def white_noise_vs_lensing(nl_spectra, powers):
    fig = plt.figure()
    ls = np.arange(powers['lensed_scalar'].shape[0])

    plt.plot(ls, powers['lensed_scalar'][:, 2],
             label=r'$C_{\ell}^{BB}$ lensed')
    plt.plot(ls, nl_spectra[:, 1], label=r'$N_{\ell}$')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\frac{ \ell (\ell +1) C_{\ell} }{2 \pi}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(2, 1000)
    plt.ylim(1E-6, 1E5)
    plt.legend()
    return fig


def spectra(cl_dict, linear_cross=False, linear_all=False, lw_list=1.5):

    if type(lw_list) == float:
        print('In spectra() default linewidtth has be chosen')
        lw_list = [lw_list] * len(cl_dict)
    if linear_all:
        print('WARNING: linear_all =', linear_all)
    if linear_cross:
        print('WARNING: linear_cross =', linear_cross)

    if 1-isinstance(cl_dict, dict):
        print("WARNING: in spectra() cl_dict is not a dictionnary")
        cl_dict = {'WARRNING:one_spectrum': cl_dict}

    # TODO: maybe take min of all entry if not all spectra have same ell max
    ls = np.arange(cl_dict[list(cl_dict.keys())[0]].shape[0])
    spectra_number = cl_dict[list(cl_dict.keys())[0]].shape[1]
    first_key = list(cl_dict.keys())[0]  # TODO: ugly
    if spectra_number == 4:
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    if spectra_number == 6:
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))

    ax[0, 0].set_title('TT')
    ax[0, 1].set_title(r'$EE$')
    ax[1, 0].set_title(r'$BB$')
    ax[1, 1].set_title(r'$TE$')
    if spectra_number == 6:
        ax[2, 0].set_title(r'$EB$')
        ax[2, 1].set_title(r'$TB$')

    spectrum_counter = 0
    for key, value in cl_dict.items():

        color = next(ax[1, 1]._get_lines.prop_cycler)['color']
        specific_spectra_number = cl_dict[key].shape[1]

        ax[0, 0].plot(ls, cl_dict[key][:, 0], label='{}'.format(
            key), color=color, lw=lw_list[spectrum_counter])
        ax[0, 1].plot(ls, cl_dict[key][:, 1], label='{}'.format(
            key), color=color, lw=lw_list[spectrum_counter])
        ax[1, 0].plot(ls, cl_dict[key][:, 2], label='{}'.format(
            key), color=color, lw=lw_list[spectrum_counter])
        ax[1, 1].plot(ls, cl_dict[key][:, 3], label='{}'.format(
            key), color=color, lw=lw_list[spectrum_counter])
        if linear_cross is False:
            ax[1, 1].plot(ls, -cl_dict[key][:, 3], '--', color=color,
                          lw=lw_list[spectrum_counter])
        if linear_all is False:
            ax[0, 0].plot(ls, -cl_dict[key][:, 0], '--', color=color,
                          lw=lw_list[spectrum_counter])
            ax[0, 1].plot(ls, -cl_dict[key][:, 1], '--', color=color,
                          lw=lw_list[spectrum_counter])
            ax[1, 0].plot(ls, -cl_dict[key][:, 2], '--', color=color,
                          lw=lw_list[spectrum_counter])
            ax[1, 1].plot(ls, -cl_dict[key][:, 3], '--', color=color,
                          lw=lw_list[spectrum_counter])

        # if spectra_number==6:
        if specific_spectra_number == 6:
            ax[2, 0].plot(ls, cl_dict[key][:, 4], label='{}'.format(
                key), color=color, lw=lw_list[spectrum_counter])
            ax[2, 1].plot(ls, cl_dict[key][:, 5], label='{}'.format(
                key), color=color, lw=lw_list[spectrum_counter])
            if linear_cross is False:
                ax[2, 0].plot(ls, -cl_dict[key][:, 4], '--',
                              color=color, lw=lw_list[spectrum_counter])
                ax[2, 1].plot(ls, -cl_dict[key][:, 5], '--',
                              color=color, lw=lw_list[spectrum_counter])
            if linear_all is False:
                ax[2, 0].plot(ls, -cl_dict[key][:, 4], '--',
                              color=color, lw=lw_list[spectrum_counter])
                ax[2, 1].plot(ls, -cl_dict[key][:, 5], '--',
                              color=color, lw=lw_list[spectrum_counter])
        if specific_spectra_number != spectra_number:
            print(' WARNING: in spectra() \"{}\" doesn t have the same',
                  'spectra number as \"{}\"'.format(key, first_key))
        spectrum_counter += 1
    for ax_ in ax.reshape(-1):
        ax_.set_xlim([2, cl_dict[list(cl_dict.keys())[0]].shape[0]])
        ax_.set_xscale('log')
        ax_.set_yscale('log')
        ax_.set_xlabel(r'$\ell$')
        ax_.set_ylabel(r'$C_{\ell} \frac{\ell (\ell +1)}{2 \pi}$')
    if linear_cross is True:
        for ax_ in ax.reshape(-1)[3:]:
            ax_.set_xscale('linear')
            ax_.set_yscale('linear')
    if linear_all is True:
        for ax_ in ax.reshape(-1):
            ax_.set_xscale('linear')
            ax_.set_yscale('linear')
    ax[0, 0].legend(borderaxespad=0., loc='upper left',
                    bbox_to_anchor=(-0.2, 1.5))

    return 0


def relative_derivative(spectra_dict, key_rot):

    cl_normalised = lib.cl_normalisation(spectra_dict[key_rot])
    cl_derivative = lib.cl_rotation_derivative(cl_normalised, key_rot)

    ls = np.arange(cl_normalised.shape[0])

    label_list = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
    for i in range(cl_normalised.shape[1]):
        dcl_o_cl = cl_derivative[:, i] / cl_normalised[:, i]

        plt.plot(ls, dcl_o_cl, label=label_list[i])
    plt.xlabel(r'$\ell$', fontsize=16)
    plt.ylabel(r'$\frac{\partial_{\alpha}C_{\ell}}{C_{\ell}}$', fontsize=16)
    plt.title(
        r'$ \frac{ \partial C_{\ell}} {\partial \alpha} . \frac{1}{C_{\ell}}$',
        ' for $\alpha =$ %s as a function of $\ell$' % key_rot)

    plt.legend()
    return 0


def truncated_fisher_cumulative(fisher_trunc_array_dict, key):
    count = 0
    label_list = ['T', 'E', 'B', 'T,E', 'E,B', 'T,B', 'T,E,B']
    for fisher_element in fisher_trunc_array_dict[key]:
        cumulative_error(fisher_element, label=label_list[count])
        count += 1
    if key[1] == 'foregrounds':
        if key[0][1] == 'no noise':
            plt.title(
                'cumulative error for rotation {}, NO noise,',
                'and with foregrounds'.format(key[0][0]))
        else:
            plt.title('cumulative error for rotation {},',
                      'noise beam {} and with foregrounds'.format(
                          key[0][0], key[0][1][1]))
    elif key[1] == 'no noise':
        plt.title('cumulative error for rotation {},',
                  'and NO noise'.format(key[0]))
    else:
        plt.title('cumulative error for rotation {},',
                  'and noise beam {}'.format(key[0], key[1][1]))
    return 0


def cumulative_error_3D(fisher_element, key, l_max=None):
    if l_max is None:
        l_max = len(fisher_element)
    l_min_array = np.arange(2, l_max)
    l_max_array = np.arange(2, l_max)

    slice_list = [[sum(fisher_element[_l_min:_l_max+1])
                   for _l_max in l_max_array]
                  for _l_min in l_min_array]
    l_min_array, l_max_array = np.meshgrid(l_min_array, l_max_array,
                                           indexing='ij')

    cumulative = np.array(1/np.sqrt(slice_list))
    cumulative[cumulative == inf] = 0
    cumulative = np.array(cumulative)

    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')

    log_cumu = np.log10(cumulative)
    log_cumu[log_cumu == inf] = None
    log_cumu[log_cumu == -inf] = None
    log_cumu = np.array(log_cumu)

    surface = ax.plot_surface(np.log10(l_min_array), np.log10(l_max_array),
                              log_cumu, cmap=cm.viridis, vmin=np.nanmin(
        log_cumu), vmax=np.nanmax(log_cumu), rcount=50, ccount=50)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

    ax.set_xlabel(r'$\ell_{min}$', fontsize=16)
    ax.set_ylabel(r'$\ell_{max}$', fontsize=16)
    ax.set_zlabel(
        r'$ \frac{1}{\sqrt{ \sum_{ \ell_{min} }^{\ell_{max}}{F_{\ell}}}} $',
        fontsize=16)
    cbar = fig1.colorbar(surface)
    cbar.ax.set_ylabel(
        r'$ \frac{1}{\sqrt{ \sum_{ \ell_{min} }^{\ell_{max}}{F_{\ell}}}} $',
        fontsize=16)

    fig2, ax2 = plt.subplots()
    contour = ax2.contourf(l_min_array, l_max_array, log_cumu, cmap=cm.viridis,
                           zdir='z', offset=np.nanmin(log_cumu)-1)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\ell_{min}$', fontsize=16)
    ax2.set_ylabel(r'$\ell_{max}$', fontsize=16)
    cbar = fig2.colorbar(contour)
    cbar.ax.set_ylabel(
        r'$ \frac{1}{\sqrt{ \sum_{ \ell_{min} }^{\ell_{max}}{F_{\ell}}}} $',
        fontsize=16)

    if key[1] == 'lensed_scalar':
        if key[0][1] == 'no noise':
            ax.set_title(
                'surface plot of cumulative error for rotation {},',
                'and NO noise, with lensing'.format(key[0][0]))
            ax2.set_title(
                'heatmap of cumulative error for rotation {},',
                'and NO noise, with lensing'.format(key[0][0]))
        else:
            ax.set_title('surface plot of cumulative error for rotation {},',
                         'and noise beam {}, with lensing'.format(
                             key[0][0], key[0][1][1]))
            ax2.set_title('heatmap of cumulative error for rotation {},',
                          'and noise beam {}, with lensing'.format(
                              key[0][0], key[0][1][1]))
    elif key[1] == 'foregrounds':
        if key[0][1] == 'no noise':
            ax.set_title(
                'surface plot of cumulative error for rotation {},',
                'and NO noise, with foregrounds'.format(key[0][0]))
            ax2.set_title(
                'heatmap of cumulative error for rotation {}, and NO noise,',
                'with foregrounds'.format(key[0][0]))
        else:
            ax.set_title('surface plot of cumulative error for rotation {},',
                         'and noise beam {}, with foregrounds'.format(
                             key[0][0], key[0][1][1]))
            ax2.set_title('heatmap of cumulative error for rotation {},',
                          'and noise beam {}, with foregrounds'.format(
                              key[0][0], key[0][1][1]))
    else:
        if key[1] == 'no noise':
            ax.set_title(
                'surface plot of cumulative error for rotation {},',
                'and NO noise'.format(key[0]))
            ax2.set_title(
                'heatmap of cumulative error for rotation {},',
                'and NO noise'.format(key[0]))
        else:
            ax.set_title(
                'surface plot of cumulative error for rotation {},',
                'and noise beam {}'.format(key[0], key[1][1]))
            ax2.set_title(
                'heatmap of cumulative error for rotation {},',
                'and noise beam {}'.format(key[0], key[1][1]))

    return fig1, fig2


def log_tick_formatter(val, pos=None):
    return "{:.2e}".format(10**val)


def one_spectrum(cl_dict, spectrum_name, ax, linear_cross=False,
                 linear_all=False):
    plotting_spectrum_dict = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3, 'EB': 4,
                              'TB': 5}
    print('linear_cross', linear_cross)
    print('linear_all', linear_all)

    ls = np.arange(cl_dict[list(cl_dict.keys())[0]].shape[0])
    print('ls', ls)
    # spectra_number = cl_dict[list(cl_dict.keys())[0]].shape[1]

    colors = plt.cm.YlOrBr_r(np.linspace(0, 1, len(cl_dict)))

    plt.gca().set_prop_cycle(cycler('color', colors))

    for key, value in cl_dict.items():
        ax.plot(ls, cl_dict[key][:, plotting_spectrum_dict[spectrum_name]],
                linewidth=2.0)  # ,label='{}'.format(key)) #'{}'.format(key)

    # ax.set_xlim([2,cl_dict[ list(cl_dict.keys())[0] ].shape[0]])

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.xlabel(r'$\ell$', fontsize= 14)
    # ax.ylabel(r'$ \frac{\ell (\ell +1)}{2 \pi} C_{\ell} \quad (\mu K^{2})$',
    # fontsize= 14)

    colors = plt.cm.YlOrBr_r(np.linspace(0, 1, len(cl_dict)))
    # (vmin=list(cl_dict.keys())[0], vmax=list(cl_dict.keys())[-1])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.YlOrBr_r)
    cmap.set_array([])
    cbar = plt.colorbar(cmap)
    cbar.set_label('rotation angle in degrees', fontsize=20)
    return 0
