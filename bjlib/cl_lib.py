import bjlib.class_faraday as cf
import numpy as np
import healpy as hp
from pysm import convert_units
import bjlib.lib_project as lib
import copy
# import pysm
import pymaster as nmt
from astropy import units as u
from scipy.optimize import minimize
import numdifftools as nd
import bjlib.mk_noise_map2 as mknm
from mpi4py import MPI


def binning_definition(nside, lmin=2, lmax=200, nlb=[], custom_bins=False):
    if custom_bins:
        ells = np.arange(3*nside, dtype='int32')  # Array of multipoles
        weights = (1.0/nlb)*np.ones_like(ells)  # Array of weights
        bpws = -1+np.zeros_like(ells)  # Array of bandpower indices
        i = 0
        while (nlb+1)*(i+1)+lmin < lmax:
            bpws[(nlb+1)*i+lmin:(nlb+1)*(i+1)+lmin] = i
            i += 1
        # adding a trash bin 2<=ell<=lmin
        # bpws[lmin:(nlb+1)*i+lmin] += 1
        # bpws[2:lmin] = 0
        # weights[2:lmin]= 1.0/(lmin-2-1)
        b = nmt.NmtBin(nside, bpws=bpws, ells=ells, weights=weights)
    else:
        b = nmt.NmtBin(nside, nlb=int(1./self.config['fsky']))
    return b


def get_field(mp_q, mp_u, mask_apo, purify_e=False, purify_b=True):
    # This creates a spin-2 field with both pure E and B.
    f2y = nmt.NmtField(mask_apo, [mp_q, mp_u], purify_e=purify_e, purify_b=purify_b)
    return f2y


def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def error_68(x_grid, y_grid, x_max_likelihood=0):
    """Evaluates the boundaries of the integral of a function such as it is
    equal to 68% of the total integral. error_68() then returns the value of
    boundarie which can then be interpreted as a 1sigma error. Warning the
    function must be symmetric arround a point x_max_likelihood.

    Parameters
    ----------
    x_grid : 1D numpy array
        Points along the x axis along which the function is evaluated.
    y_grid : 1D numpy array
        Values of the function corresponding to the points in the x_grid.
    x_max_likelihood : float
        Position of the maximum of the likelihood around which the boundaries
        will be centered.

    Returns
    -------
    float
        Position on the x axis of the upper boundary of the integral such that
        integral of the function from x = - error + x_max_likelihood to
        x = + error + x_max_likelihood equals 0.68 of the total integral.
        aka 1 sigma error.

    """

    total_integral = np.trapz(y_grid, x=x_grid)
    if np.isnan(total_integral):
        print('nan in x_grid or y_grid impossible to compute integral')
        return np.nan
    positive_index = np.where(x_grid >= 0 + x_max_likelihood)
    negative_index = np.where(x_grid < 0 + x_max_likelihood)
    partial_integral = 0
    i = 0
    y_array = np.array([])
    x_array = np.array([])
    while partial_integral <= 0.682689*total_integral:
        y_array = np.append(y_grid[negative_index[0][-(i+1)]], y_array)
        y_array = np.append(y_array, y_grid[positive_index[0][i]])
        x_array = np.append(
            x_grid[negative_index[0][-(i+1)]], x_array)
        x_array = np.append(x_array, x_grid[positive_index[0][i]])
        partial_integral = np.trapz(y_array, x_array)

        # partial_integral += np.exp(in_exp[positive_index[0][i]]) +
        #     np.exp(in_exp[negative_index[0][-i]])
        i += 1
    error = x_grid[positive_index[0][i]] - x_max_likelihood
    return error


def L_gridding(spectra_model, data_matrix, alpha_grid,
               bin, nside, f_sky_SAT, spectra_used='all'):

    grid_logL = []
    # grid_model_matrix = []
    # grid_data_matrix = []
    # grid_Cm1D = []
    # likelihood_element_grid = []
    for alpha in alpha_grid:
        a = cf.likelihood_for_hessian_a(alpha, spectra_model, data_matrix,
                                        bin, nside, f_sky_SAT, spectra_used)
        grid_logL.append(a)
        # grid_model_matrix.append(b.spectra)
        # grid_data_matrix.append(c.spectra)
        # grid_Cm1D.append(d)
        # likelihood_element_grid.append(e)
        # grid_logL.append(cf.likelihood_for_hessian_a(
        # alpha, spectra_model, data_matrix,
        # b, nside, f_sky_SAT, spectra_used))

    grid_logL = np.array(grid_logL)
    max_logL = max(-grid_logL[~np.isnan(grid_logL)])

    # grid_LogL_woMax = grid_logL[~np.array(-grid_logL == max(-grid_logL))]
    # in_exp_2 = -grid_LogL_woMax - max(-grid_LogL_woMax)
    # grid_alpha2 = alpha_grid[~np.array(-grid_logL == max(-grid_logL))]
    in_exp = -np.array(grid_logL) - max_logL  # max(-np.array(grid_logL))

    # , np.array(grid_model_matrix), np.array(grid_data_matrix), np.array(grid_Cm1D), np.array(likelihood_element_grid), (grid_alpha2, np.exp(in_exp_2))
    return np.exp(in_exp)


def grid_error(spectra_model, data_matrix, alpha_grid, bin, nside, f_sky_SAT,
               spectra_used='all', x_max_likelihood=0, output_grid=False):
    likelihood_grid = L_gridding(spectra_model, data_matrix, alpha_grid, bin,
                                 nside, f_sky_SAT, spectra_used=spectra_used)
    error = error_68(alpha_grid, likelihood_grid, x_max_likelihood)
    if output_grid:
        return error, likelihood_grid
    return error


def jac_n2logL(angle, spectra_model, data_matrix, bin, nside, f_sky_SAT, spectra_used='all'):
    angle = angle * u.rad
    model = copy.deepcopy(spectra_model)
    model.spectra_rotation(angle)
    model.l_min_instru = 0
    model.l_max_instru = 3*nside
    model.get_noise()
    model.get_instrument_spectra()
    model_spectra = bin.bin_cell(model.instru_spectra.spectra[:3*nside].T).T

    deriv = cf.power_spectra_obj(bin.bin_cell(lib.cl_rotation_derivative(
        spectra_model.spectra.spectra, angle)[:3*nside].T).T, bin.get_effective_ells())

    if spectra_used == 'all':
        model_matrix = cf.power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, bin.get_effective_ells())
        model_inverse = np.linalg.inv(model_matrix.spectra)
        Cm1D = np.einsum('kij,kjl->kil', model_inverse, data_matrix.spectra)

        deriv_matrix = cf.power_spectra_obj(np.array(
            [[deriv.spectra[:, 1], deriv.spectra[:, 4]],
             [deriv.spectra[:, 4], deriv.spectra[:, 2]]]).T, bin.get_effective_ells())
        Cm1dC = np.einsum('kij,kjl->kil', model_inverse, deriv_matrix.spectra)
        Cm1dCCm1D = np.einsum('kij,kjl->kil', Cm1dC, Cm1D)
        jac_trace = np.trace(-Cm1dCCm1D + Cm1dC, axis1=1, axis2=2)

    elif spectra_used == 'EE':
        model_matrix = cf.power_spectra_obj(np.array(
            model_spectra[:, 1]).T, bin.get_effective_ells())
        model_inverse = 1/model_matrix.spectra
        Cm1D = model_inverse * data_matrix.spectra

        deriv_matrix = cf.power_spectra_obj(np.array(
            deriv.spectra[:, 1]).T, bin.get_effective_ells())
        Cm1dC = model_inverse * deriv_matrix.spectra
        Cm1dCCm1D = Cm1dC * Cm1D
        jac_trace = -Cm1dCCm1D + Cm1dC

    elif spectra_used == 'BB':
        model_matrix = cf.power_spectra_obj(np.array(
            model_spectra[:, 2]).T, bin.get_effective_ells())
        model_inverse = 1/model_matrix.spectra
        Cm1D = model_inverse * data_matrix.spectra

        deriv_matrix = cf.power_spectra_obj(np.array(
            deriv.spectra[:, 2]).T, bin.get_effective_ells())
        Cm1dC = model_inverse * deriv_matrix.spectra
        Cm1dCCm1D = Cm1dC * Cm1D
        jac_trace = -Cm1dCCm1D + Cm1dC

    elif spectra_used == 'EB':
        model_matrix = cf.power_spectra_obj(np.array(
            model_spectra[:, 4]).T, bin.get_effective_ells())
        model_inverse = 1/model_matrix.spectra
        Cm1D = model_inverse * data_matrix.spectra

        deriv_matrix = cf.power_spectra_obj(np.array(
            deriv.spectra[:, 4]).T, bin.get_effective_ells())
        Cm1dC = model_inverse * deriv_matrix.spectra
        Cm1dCCm1D = Cm1dC * Cm1D
        jac_trace = -Cm1dCCm1D + Cm1dC
    jac = 0
    ell_counter = 0
    jac_element = []
    for l in bin.get_effective_ells():
        jac_ell = f_sky_SAT*(2*l + 1)*0.5 * jac_trace[ell_counter]
        jac += jac_ell
        jac_element.append(jac_ell)
        ell_counter += 1
    jac = np.array([jac])
    print(jac)
    return jac


def noise_maps_simulation(sensitivity, knee_mode, ny_lf, nside,
                          norm_hits_map_path, no_inh, frequencies_index):
    nhits, noise_maps_, nlev = mknm.get_noise_sim(
        sensitivity=sensitivity, knee_mode=knee_mode,
        ny_lf=ny_lf, nside_out=nside,
        norm_hits_map=hp.read_map(norm_hits_map_path), no_inh=no_inh)
    noise_maps = []

    # noise_maps_masked = []

    for i in frequencies_index:
        noise_maps.append([noise_maps_[i*3],
                           noise_maps_[i*3+1],
                           noise_maps_[i*3+2]])
        # noise_maps_masked.append([noise_maps_[i*3] * mask,
        #                           noise_maps_[i*3+1] * mask,
        #                           noise_maps_[i*3+2] * mask])
    noise_maps = np.array(noise_maps)
    # noise_maps_masked = np.array(noise_maps_masked)

    del noise_maps_
    return noise_maps, nhits


def get_foreground_maps_and_cl(sky_obj, miscalibration_angle, frequencies2use,
                               mask, mask_apo, wsp, purify_e, purify_b,
                               return_dust=True, return_synchrotron=True,
                               return_maps=False):
    dust_map_freq = []
    sync_map_freq = []
    Cl_dust_freq = []
    Cl_sync_freq = []

    for f in frequencies2use:
        if return_dust:
            dust_maps_ = sky_obj.sky.dust(f) * \
                convert_units('K_RJ', 'K_CMB', f)
            dust_maps = lib.map_rotation(dust_maps_, miscalibration_angle)
            dust_map_freq.append(dust_maps)
            f2_dust = get_field(mask*dust_maps[1, :],
                                mask*dust_maps[2, :], mask_apo,
                                purify_e=purify_e, purify_b=purify_b)
            Cl_dust_freq.append(compute_master(f2_dust, f2_dust, wsp))

        if return_synchrotron:
            sync_maps_ = sky_obj.sky.synchrotron(f) * \
                convert_units('K_RJ', 'K_CMB', f)
            sync_maps = lib.map_rotation(sync_maps_, miscalibration_angle)
            sync_map_freq.append(sync_maps)

            f2_sync = get_field(mask*sync_maps[1, :],
                                mask*sync_maps[2, :], mask_apo,
                                purify_e=purify_e, purify_b=purify_b)
            Cl_sync_freq.append(compute_master(f2_sync, f2_sync, wsp))

    if return_dust:
        dust_map_freq = np.array(dust_map_freq)
        Cl_dust_freq = np.array(Cl_dust_freq)

    if return_synchrotron:
        sync_map_freq = np.array(sync_map_freq)
        Cl_sync_freq = np.array(Cl_sync_freq)
    if return_maps:
        if return_dust and return_synchrotron:
            return Cl_dust_freq, Cl_sync_freq, dust_map_freq, sync_map_freq
        elif return_dust:
            return Cl_dust_freq, dust_map_freq
        elif return_synchrotron:
            return Cl_sync_freq, sync_map_freq
    else:
        return Cl_dust_freq, Cl_sync_freq


def get_nsim_freq_cl(input_map, nsim, frequencies_index, mask, mask_apo,
                     purify_e, purify_b, wsp):

    Cl_nsim_freq = []
    for i in range(nsim):
        Cl_freq = []
        for f in range(len(frequencies_index)):
            f2 = get_field(mask*input_map[i, f, 1, :],
                           mask*input_map[i, f, 2, :], mask_apo,
                           purify_e=purify_e, purify_b=purify_b)
            Cl_freq.append(compute_master(f2, f2, wsp))
        Cl_nsim_freq.append(Cl_freq)
    Cl_nsim_freq = np.array(Cl_nsim_freq)

    return Cl_nsim_freq


def min_and_error_nsim_freq(
        model, data_cl, nsim, frequencies_index, bin, nside, f_sky_SAT,
        spectra_used='all', spectra_indexation='NaMaster',
        minimisation_init=0.001,
        compute_error68=False, step_size=1e-4, output_grid=False):
    if spectra_indexation == 'NaMaster':
        indexation = {'EE': 0, 'BB': 3, 'EB': 1}
    elif spectra_indexation == 'CAMB':
        indexation = {'EE': 1, 'BB': 2, 'EB': 4}
    minimisation_nsim_freq = []
    H_nsim_freq = []
    if compute_error68:
        error_nsim_freq = []
        if output_grid:
            grid_nsim_freq = []

    for i in range(nsim):
        minimisation_freq = []
        H_freq = []
        if compute_error68:
            error_freq = []
            if output_grid:
                grid_freq = []

        for f in range(len(frequencies_index)):
            if spectra_used == 'all':
                data_matrix = cf.power_spectra_obj(np.array(
                    [[data_cl[i, f, indexation['EE']], data_cl[i, f, indexation['EB']]],
                     [data_cl[i, f, indexation['EB']], data_cl[i, f, indexation['BB']]]]).T,
                    bin.get_effective_ells())
            else:
                data_matrix = cf.power_spectra_obj(np.array(
                    data_cl[i, f, indexation[spectra_used]]).T,
                    bin.get_effective_ells())

            minimisation_likelihood_results = minimize(
                cf.likelihood_for_hessian_a, minimisation_init,
                (model, data_matrix, bin, nside, f_sky_SAT, spectra_used),
                jac=jac_n2logL)
            H = nd.Hessian(cf.likelihood_for_hessian_a)(
                minimisation_likelihood_results.x,
                model, data_matrix, bin,
                nside, f_sky_SAT, spectra_used)
            minimisation_freq.append(minimisation_likelihood_results.x[0])
            H_freq.append(H[0][0])
            if compute_error68:
                min_alpha = minimisation_likelihood_results.x[0] - 6/np.sqrt(H[0][0])
                max_alpha = minimisation_likelihood_results.x[0] + 6/np.sqrt(H[0][0])
                alpha_grid = np.arange(min_alpha, max_alpha, step_size)
                error_ = grid_error(
                    model, data_matrix, alpha_grid, bin, nside, f_sky_SAT,
                    spectra_used=spectra_used,
                    x_max_likelihood=minimisation_likelihood_results.x[0],
                    output_grid=output_grid)
                if output_grid:
                    error_freq.append(error_[0])
                    grid_freq.append(error_[1])
                else:
                    error_freq.append(error_)

        minimisation_nsim_freq.append(minimisation_freq)
        H_nsim_freq.append(H_freq)
        del minimisation_freq
        del H_freq

        if compute_error68:
            error_nsim_freq.append(error_freq)
            del error_freq
            if output_grid:
                grid_nsim_freq.append(grid_freq)
                del grid_freq

    minimisation_nsim_freq = np.array(minimisation_nsim_freq)
    H_nsim_freq = np.array(H_nsim_freq)
    if compute_error68:
        error_nsim_freq = np.array(error_nsim_freq)
        if output_grid:
            grid_nsim_freq = np.array(grid_nsim_freq)
    if compute_error68 and output_grid:
        return minimisation_nsim_freq, H_nsim_freq, error_nsim_freq, grid_nsim_freq
    elif compute_error68:
        return minimisation_nsim_freq, H_nsim_freq, error_nsim_freq

    return minimisation_nsim_freq, H_nsim_freq


def get_nsim_freq_cl_mpi(input_map, nsim, frequencies_index, mask, mask_apo,
                         purify_e, purify_b, wsp):

    # Cl_nsim_freq = []
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()
    # print('SIZE = ', size)
    # print('nside =', nsim)
    # root = 0
    Cl_freq = []
    for f in range(len(frequencies_index)):
        f2 = get_field(mask*input_map[f, 1, :],
                       mask*input_map[f, 2, :], mask_apo,
                       purify_e=purify_e, purify_b=purify_b)
        Cl_freq.append(compute_master(f2, f2, wsp))
    print('shape Cl_freq=', np.shape(Cl_freq))
    Cl_freq = np.array(Cl_freq)

    return Cl_freq


def min_and_error_nsim_freq_mpi(
        model, data_cl, nsim, frequencies_index, bin, nside, f_sky_SAT,
        spectra_used='all', spectra_indexation='NaMaster',
        minimisation_init=0.001,
        compute_error68=False, step_size=1e-4, output_grid=False):
    comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    root = 0
    if spectra_indexation == 'NaMaster':
        indexation = {'EE': 0, 'BB': 3, 'EB': 1}
    elif spectra_indexation == 'CAMB':
        indexation = {'EE': 1, 'BB': 2, 'EB': 4}
    minimisation_nsim_freq = []
    H_nsim_freq = []
    if compute_error68:
        error_nsim_freq = []
        if output_grid:
            grid_nsim_freq = []

    minimisation_freq = []
    H_freq = []
    if compute_error68:
        error_freq = []
        if output_grid:
            grid_freq = []

    for f in range(len(frequencies_index)):
        if spectra_used == 'all':
            data_matrix = cf.power_spectra_obj(np.array(
                [[data_cl[f, indexation['EE']], data_cl[f, indexation['EB']]],
                 [data_cl[f, indexation['EB']], data_cl[f, indexation['BB']]]]).T,
                bin.get_effective_ells())
        else:
            data_matrix = cf.power_spectra_obj(np.array(
                data_cl[f, indexation[spectra_used]]).T,
                bin.get_effective_ells())

        minimisation_likelihood_results = minimize(
            cf.likelihood_for_hessian_a, minimisation_init,
            (model, data_matrix, bin, nside, f_sky_SAT, spectra_used),
            jac=jac_n2logL)
        H = nd.Hessian(cf.likelihood_for_hessian_a)(
            minimisation_likelihood_results.x,
            model, data_matrix, bin,
            nside, f_sky_SAT, spectra_used)
        minimisation_freq.append(minimisation_likelihood_results.x[0])
        H_freq.append(H[0][0])
        if compute_error68:
            min_alpha = minimisation_likelihood_results.x[0] - 6/np.sqrt(H[0][0])
            max_alpha = minimisation_likelihood_results.x[0] + 6/np.sqrt(H[0][0])
            alpha_grid = np.arange(min_alpha, max_alpha, step_size)
            error_ = grid_error(
                model, data_matrix, alpha_grid, bin, nside, f_sky_SAT,
                spectra_used=spectra_used,
                x_max_likelihood=minimisation_likelihood_results.x[0],
                output_grid=output_grid)
            if output_grid:
                error_freq.append(error_[0])
                grid_freq.append(error_[1])
            else:
                error_freq.append(error_)

    minimisation_freq = np.array(minimisation_freq)
    H_freq = np.array(H_freq)

    if compute_error68:
        error_nsim_freq = comm.gather(error_freq, root)
        del error_freq
        if output_grid:
            grid_nsim_freq = comm.gather(grid_freq, root)
            del grid_freq

    if compute_error68:
        error_nsim_freq = comm.bcast(error_nsim_freq, root)
        error_nsim_freq = np.array(error_nsim_freq)

        if output_grid:
            grid_nsim_freq = comm.bcast(grid_nsim_freq, root)
            grid_nsim_freq = np.array(grid_nsim_freq)
    if compute_error68 and output_grid:
        return minimisation_nsim_freq, H_nsim_freq, error_nsim_freq, grid_nsim_freq
    elif compute_error68:
        return minimisation_nsim_freq, H_nsim_freq, error_nsim_freq

    return minimisation_freq, H_freq
