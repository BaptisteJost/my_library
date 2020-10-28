
import numpy as np
import healpy as hp
import camb
from astropy import units as u
import pysm
import copy
from fgbuster import MixingMatrix
from fgbuster.component_model import CMB, Dust, Synchrotron
from . import V3calc as V3


def get_basics(l_max=5000, raw_cl=False, lens_potential=False, ratio=0.07):
    # l_max = 5000
    pars = camb.CAMBparams(WantTensors=True, max_l_tensor=l_max,
                           parameterization="tensor_param_indeptilt")
    # , tensor_parameterization=3)
    # This function sets up CosmoMC-like settings,
    # swith one massive neutrino and helium set using BBN consistency
    # pars.set_accuracy(AccuracyBoost=2)
    pars.Accuracy.AccurateBB = True
    pars.Accuracy.AccuratePolarization = True

    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0,
                       tau=0.06)
    # camb.initialpower.InitialPowerLaw(tensor_parameterization=1)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=ratio,
                              parameterization="tensor_param_indeptilt",
                              nt=0, ntrun=0)
    pars.set_for_lmax(l_max, lens_potential_accuracy=1)  # !! why 1 ??
    pars.max_eta_k_tensor = l_max + 100  # 15000  # 100
    # if lmax=5000 max_eta_k_tensor=4981 works
    # too, maybe there is some improvement possible

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, lmax=l_max, CMB_unit='muK',
                                           raw_cl=raw_cl)
    # print(pars)
    if lens_potential:
        lens_potential_spectrum = results.get_lens_potential_cls(l_max)
        return pars, results, powers, lens_potential_spectrum
    return pars, results, powers


def cl_normalisation(cl):  # TODO: DUMB NAME VERY CONFUSING !!
    ls = np.arange(cl.shape[0])
    ls[0] = 1
    cl_normalised = np.array(cl.T[:] * 2*np.pi / ((ls+1)*ls))
    cl_normalised[:, 0] = 0
    return cl_normalised.T


def get_normalised_cl(cl):  # WARNING: cl is of shape [cl,spectrum]
    ls = np.arange(cl.shape[0])
    # ls[0]=1
    cl_normalised = np.array(cl.T[:] * ((ls+1)*ls) / (2*np.pi))
    # cl_normalised[:,0] = 0
    return cl_normalised.T


def cl_rotation(cl, angle_in):
    """Performs a rotation of the power spectrum polarisation.

    Parameters
    ----------
    cl : numpy array [number of l, number of spectra]
        Numpy array of the different power spectra in order : TT, EE, BB, TE,
        and optionnally EB, TB
        Takes both raw and l*(l+1)cl power spectra
    angle : float or numpy array
        angle of rotation IN RADIAN, can be isotropic is float or anisotropic
        if numpy array. In the latter case the angle is discomposed as fourier
        transmorm on the sphere 'l', and the length of the array should be the
        same as the length of a given cl element (eg the EE spectrum)
    EB : bool
        If EB spectrum in the given power spectra array 'cl' put as 'True'
    TB : bool
        If TB spectrum in the given power spectra array 'cl' put as 'True'

    Returns
    -------
    Numpy array
        Same shape as the given power spectra array 'cl' but with EB and TB
        contributions in addition if not present already.
        i.e. : TT, EE, BB, TE, EB, TB

    """
    angle = angle_in.to(u.rad).value

    cl_EE_rot = cl[:, 1] * (np.cos(2 * angle)**2) + cl[:, 2] * \
        (np.sin(2 * angle)**2)
    cl_BB_rot = cl[:, 1] * (np.sin(2 * angle)**2) + cl[:, 2] * \
        (np.cos(2 * angle)**2)
    cl_EB_rot = (cl[:, 1] - cl[:, 2]) * np.sin(4 * angle) * 0.5
    cl_TE_rot = cl[:, 3] * np.cos(2 * angle)
    cl_TB_rot = cl[:, 3] * np.sin(2 * angle)
    if cl.shape[0] == 6:

        cl_EE_rot -= cl[:, 4] * np.sin(4 * angle)
        cl_BB_rot += cl[:, 4] * np.sin(4 * angle)
        cl_EB_rot += cl[:, 4] * np.cos(4 * angle)

        cl_TE_rot -= cl[:, 5] * np.sin(2 * angle)
        cl_TB_rot += cl[:, 5] * np.cos(2 * angle)
    cl_rot = np.array([np.array([cl[k, 0], cl_EE_rot[k], cl_BB_rot[k],
                                 cl_TE_rot[k], cl_EB_rot[k], cl_TB_rot[k]])
                       for k in range(len(cl))])
    return cl_rot


def cl_rotation_derivative(cl, angle_in):
    """Returns de derivative of the rotated power spectra w.r.t. the rotation angle
        Useful for the computation of Fisher.

    Parameters
    ----------
    cl : numpy array [number of l, number of spectra]
        Numpy array of the different power spectra in order : TT, EE, BB, TE,
        and optionnally EB, TB
    angle : float or numpy array [number of l]
        angle of rotation, can be isotropic is float or anisotropic if numpy
        array. In the latter case the angle is discomposed as fourier transmorm
        on the sphere 'l', and the length of the array should be the same as
        the length of a given cl element (eg the EE spectrum)
    EB : bool
        If EB spectrum in the given power spectra array 'cl' put as 'True'
    TB : bool
        If TB spectrum in the given power spectra array 'cl' put as 'True'

    Returns
    -------
    type
        Same shape as the given power spectra array 'cl' but with EB and TB
        contributions in addition if not present already.
        i.e. : TT, EE, BB, TE, EB, TB

    """
    print('Warning ! cl_rotation_derivative() is an outdated version, sign error may occur')
    print('Be particularly cautious about the use of TB,TE terms')
    angle = angle_in.to(u.rad).value
    cl_EE_rot_da = np.sin(4 * angle) * (cl[:, 1] - cl[:, 2])
    cl_BB_rot_da = np.sin(4 * angle) * (-cl[:, 1] + cl[:, 2])
    cl_EB_rot_da = np.cos(4 * angle) * (-cl[:, 1] + cl[:, 2])
    cl_TE_rot_da = cl[:, 3] * np.sin(2 * angle)
    cl_TB_rot_da = cl[:, 3] * (-np.cos(2 * angle))

    if cl.shape[0] == 6:
        cl_EE_rot_da += cl[:, 4] * 2 * np.cos(4 * angle)
        cl_BB_rot_da -= cl[:, 4] * 2 * np.cos(4 * angle)
        cl_EB_rot_da += cl[:, 4] * 2 * np.sin(4 * angle)

        cl_TE_rot_da += cl[:, 5] * np.cos(2 * angle)
        cl_TB_rot_da += cl[:, 5] * np.sin(2 * angle)

    cl_rot_da = np.array([np.array([0, 2 * cl_EE_rot_da[k],
                                    2 * cl_BB_rot_da[k], -2 * cl_TE_rot_da[k],
                                    2 * cl_EB_rot_da[k],
                                    -2 * cl_TB_rot_da[k]])
                          for k in range(len(cl))])
    return cl_rot_da


def fisher_angle(cl, angle, cl_rot=None, f_sky=1,
                 return_elements=False, raw_cl=False, raw_cl_rot=False):
    """Returns the value of Fisher for the rotation angle parameter for a given
        angle.

    Parameters
    ----------
    cl : numpy array [number of l, number of spectra]
        Numpy array of the different power spectra in order : TT, EE, BB, TE,
        and optionnally EB, TB
    angle : float or numpy array [number of l]
        angle of rotation, can be isotropic is float or anisotropic if numpy
        array. In the latter case the angle is decomposed as fourier transmorm
        on the sphere 'l', and the length of the array should be the same as
        the length of a given cl element (eg the EE spectrum)
    cl_rot : numpy array [number of l, number of spectra]
        Rotated power spectra by the angle 'angle' using cl_rotation. If not
        given it will be computed using cl_rotation(cl, angle)
    f_sky : float
        percentage of the observed sky
    EB : bool
        If EB spectrum in the given power spectra array 'cl' put as 'True'
    TB : bool
        If TB spectrum in the given power spectra array 'cl' put as 'True'

    Returns
    -------
    float
        The Fisher value for the rotation angle 'angle'

    Notes
    -------
    the two first multipole (l=0, l=1) are ignored throughout due to camb
    glitches that might be a problem ?
    """
    if raw_cl is False:
        cl = cl_normalisation(cl)

    if np.all(cl_rot is None):
        cl_rot = cl_rotation(cl, angle)
    else:
        if raw_cl_rot is False:
            cl_rot = cl_normalisation(cl_rot)

    cl_rot_da = cl_rotation_derivative(cl, angle)
    cov_matrix_da = np.array(
        [[cl_rot_da[:, 0], cl_rot_da[:, 3], cl_rot_da[:, 5]],
         [cl_rot_da[:, 3], cl_rot_da[:, 1], cl_rot_da[:, 4]],
         [cl_rot_da[:, 5], cl_rot_da[:, 4], cl_rot_da[:, 2]]])

    cov_matrix = np.array(
        [[cl_rot[:, 0], cl_rot[:, 3], cl_rot[:, 5]],
         [cl_rot[:, 3], cl_rot[:, 1], cl_rot[:, 4]],
         [cl_rot[:, 5], cl_rot[:, 4], cl_rot[:, 2]]])

    cov_matrix_inv = np.linalg.inv(cov_matrix.T[2:])
    cov_matrix_inv = cov_matrix_inv.T

    sq_in_trace = np.array([np.dot(cov_matrix_inv[:, :, k], cov_matrix_da[:, :,
                                                                          k+2])
                            for k in range(cov_matrix_inv.shape[2])])

    in_trace = np.array([np.dot(sq_in_trace[k, :, :], sq_in_trace[k, :, :])
                         for k in range(sq_in_trace.shape[0])])

    trace_fisher = np.trace(in_trace, axis1=1, axis2=2)

    fisher_element = [0., 0.]

    fisher = 0
    for l in range(2, len(cl)):
        fisher += (2*l + 1) * 0.5 * f_sky * trace_fisher[l-2]
        fisher_element.append((2*l + 1) * 0.5 * f_sky * trace_fisher[l-2])
    fisher_element = np.array(fisher_element)
    if return_elements:
        return fisher, fisher_element
    return fisher


def cl_to_map(cl, nside, raw_cl=False):
    # TODO: seems fishy, triple check
    if raw_cl is False:
        cl = cl_normalisation(cl).T
    else:
        cl = cl.T

    map = hp.sphtfunc.synfast(cl, nside, new=True)

    return map


def fisher_manual(cl, f_sky=1, raw_cl=False):

    if raw_cl is False:
        cl = cl_normalisation(cl)

    fisher = 0
    if cl.shape[1] == 6:
        for l in range(2, len(cl)):
            fisher += (2*l + 1) * 0.5 * f_sky *\
                (-16 + 8 * ((-cl[l, 0] * (2*cl[l, 4]**2 + cl[l, 2]**2 +
                                          cl[l, 1]**2) +
                             cl[l, 1] * cl[l, 3]**2 +
                             2*cl[l, 3]*cl[l, 5]*cl[l, 4] +
                             cl[l, 5]**2 * cl[l, 2]) /
                            ((cl[l, 0] * (cl[l, 4]**2 - cl[l, 1] * cl[l, 2]) +
                              cl[l, 3]**2 * cl[l, 2] -
                              2*cl[l, 3]*cl[l, 5]*cl[l, 4]
                              + cl[l, 1] * cl[l, 5]**2))
                            )
                 )
    if cl.shape[1] == 4:
        for l in range(2, len(cl)):
            fisher += (2*l + 1) * 0.5 * f_sky *\
                (-16 + 8 * ((-cl[l, 0] * (cl[l, 2]**2 + cl[l, 1]**2) +
                             cl[l, 1] * cl[l, 3]**2)
                            /
                            ((cl[l, 0] * (- cl[l, 1] * cl[l, 2]) +
                              cl[l, 3]**2 * cl[l, 2]))
                            )
                 )
    return fisher


def get_noise_spectrum(w_inv, theta_fwhm, l_max):
    theta_fwhm = theta_fwhm.to(u.rad).value

    w_inv = w_inv.to(u.uK*u.uK * u.rad*u.rad).value

    el = np.arange(l_max+1)

    nl = w_inv * np.exp((el * (el+1) * theta_fwhm * theta_fwhm) /
                        (8 * np.log(2)))

    return nl


def get_noise_spectra(w_inv, theta_fwhm, l_max, raw_output=False):
    nl_raw = get_noise_spectrum(w_inv, theta_fwhm, l_max)
    el = np.arange(len(nl_raw))
    nl = nl_raw * el * (el+1) / (2*np.pi)

    nl_spectra = np.array([np.array([nl[k] / 2., nl[k], nl[k],
                                     nl[k]/np.sqrt(2),
                                     nl[k], nl[k]/np.sqrt(2)])
                           for k in range(len(nl))])

    if raw_output:
        nl_spectra = cl_normalisation(nl_spectra)
    return nl_spectra


def get_spectra_dict(cl_unchanged, angle_array, include_unchanged=True):
    if include_unchanged:
        spectra_dict = {0*u.deg: cl_rotation(cl_unchanged, 0*u.deg)}
    else:
        spectra_dict = {}

    for k in angle_array:
        spectra_dict[k] = cl_rotation(cl_unchanged, k)
    return spectra_dict


def spectra_addition(spectra_dict, key1, key2):
    spectra_dict[(key1, key2)] = copy.deepcopy(
        spectra_dict[key1]) + copy.deepcopy(spectra_dict[key2])
    return 0


def get_fisher_dict(spectra_dict, angle_array, w_inv, beam_array,
                    lensing=False, foregrounds=False, fisher_dict={}):
    # TODO: same as get_truncated_fisher_dict
    if type(angle_array[0].shape) == ():
        angle_array = [angle_array]
    if type(beam_array[0].shape) == ():
        beam_array = [beam_array]
    if fisher_dict == {}:
        for key_rot in angle_array:
            fisher, fisher_element = fisher_angle(
                spectra_dict[(0*u.deg, 'unlensed')],
                key_rot, cl_rot=spectra_dict[(key_rot, 'unlensed')],
                return_elements=True)
            fisher_dict[(key_rot, 'no noise')] = fisher_element

    for key_rot in angle_array:
        for k in beam_array:
            fisher, fisher_element = fisher_angle(
                spectra_dict[(0*u.deg, 'unlensed')], key_rot,
                cl_rot=spectra_dict[(key_rot, (w_inv, k*u.arcmin))],
                return_elements=True)
            fisher_dict[(key_rot, (w_inv, k*u.arcmin))] = fisher_element
    if lensing:
        for key_rot in angle_array:
            for k in beam_array:
                fisher, fisher_element = fisher_angle(
                    spectra_dict[(0*u.deg, 'unlensed')], key_rot,
                    cl_rot=spectra_dict[(key_rot, (w_inv, k*u.arcmin))],
                    return_elements=True)
                fisher_dict[((key_rot, (w_inv, k*u.arcmin)),
                             'lensed_scalar')] = fisher_element
    if foregrounds:
        for key_rot in angle_array:
            for k in beam_array:
                fisher, fisher_element = fisher_angle(
                    spectra_dict[(0*u.deg, 'unlensed')], key_rot,
                    cl_rot=spectra_dict[((key_rot, (w_inv, k*u.arcmin)),
                                         'foregrounds')], return_elements=True)
                fisher_dict[((key_rot, (w_inv, k*u.arcmin)),
                             'foregrounds')] = fisher_element
    return fisher_dict


def get_truncated_fisher_dict(spectra_dict, angle_array, w_inv, beam_array,
                              foregrounds=False, fisher_trunc_array_dict={}):
    """Short summary.

    Parameters
    ----------
    spectra_dict : type
        Description of parameter `spectra_dict`.
    angle_array : type
        Description of parameter `angle_array`.
    w_inv : type
        Description of parameter `w_inv`.
    beam_array : type
        Description of parameter `beam_array`.
    foregrounds : type
        Description of parameter `foregrounds`.
    fisher_trunc_array_dict : type
        Description of parameter `fisher_trunc_array_dict`.

    Returns
    -------
    type
        Description of returned object.
        TODO : if fisher_trunc given diff than {} check for already created
         entries in dic so that not to create them twice loop over keys in
         spectra dict to create fisher_trunc_array_dict (no need for extrea
         if statements)
    """

    if type(angle_array[0].shape) == ():
        angle_array = [angle_array]

    if fisher_trunc_array_dict == {}:
        for key_rot in angle_array:
            truncated_fisher_array, truncated_fisher_element_array = \
                truncated_fisher_angle(spectra_dict[0*u.deg], key_rot,
                                       cl_rot_noise=spectra_dict[key_rot],
                                       return_elements=True)
            fisher_trunc_array_dict[(key_rot, 'no noise')] = \
                truncated_fisher_element_array

    for key_rot in angle_array:
        for k in beam_array:
            truncated_fisher_array, truncated_fisher_element_array = \
                truncated_fisher_angle(
                    spectra_dict[0*u.deg], key_rot,
                    cl_rot_noise=spectra_dict[(key_rot, (w_inv, k*u.arcmin))],
                    return_elements=True)
            fisher_trunc_array_dict[(key_rot, (w_inv, k*u.arcmin))] = \
                truncated_fisher_element_array

    if foregrounds:
        for key_rot in angle_array:
            for k in beam_array:
                truncated_fisher_array, truncated_fisher_element_array = \
                    truncated_fisher_angle(
                        spectra_dict[0*u.deg], key_rot,
                        cl_rot_noise=spectra_dict[
                            ((key_rot, (w_inv, k*u.arcmin)), 'foregrounds')],
                        return_elements=True)
                fisher_trunc_array_dict[((key_rot, (w_inv, k*u.arcmin)),
                                         'foregrounds')] = \
                    truncated_fisher_element_array

    return fisher_trunc_array_dict


def get_truncated_covariance_matrix(cl):
    TT = cl[:, 0]
    EE = cl[:, 1]
    BB = cl[:, 2]
    TE = np.array([[cl[:, 0], cl[:, 3]],
                   [cl[:, 3], cl[:, 1]]])
    EB = np.array([[cl[:, 1], cl[:, 4]],
                   [cl[:, 4], cl[:, 2]]])
    TB = np.array([[cl[:, 0], cl[:, 5]],
                   [cl[:, 5], cl[:, 2]]])
    matrix_array = np.array([TT, EE, BB, TE, EB, TB])
    return matrix_array


def truncated_fisher_angle(cl_orig_for_deriv, angle, cl_rot_noise, f_sky=1,
                           return_elements=False, raw_cl=False,
                           raw_cl_rot=False):
    if raw_cl is False:
        cl_orig = cl_normalisation(cl_orig_for_deriv)
    else:
        cl_orig = cl_orig_for_deriv

    if raw_cl_rot is False:
        cl_rot_noise = cl_normalisation(cl_rot_noise)

    cl_da = cl_rotation_derivative(cl_orig, angle)

    cl_rot_noise_truncated_array = \
        get_truncated_covariance_matrix(cl_rot_noise)

    cl_da_truncated_array = get_truncated_covariance_matrix(cl_da)
    # print('cl_da_truncated_array[4]=',cl_da_truncated_array[4])
    truncated_fisher_list = []
    truncated_fisher_element_list = []

    for i in range(len(cl_da_truncated_array)):

        if np.shape(np.shape(cl_rot_noise_truncated_array[i].T[2:]))[0] == 1:
            cov_matrix_inv = 1/(cl_rot_noise_truncated_array[i].T[2:])

        else:
            cov_matrix_inv = np.linalg.inv(
                cl_rot_noise_truncated_array[i].T[2:])

        cov_matrix_inv = cov_matrix_inv.T

        sq_in_trace = np.array([np.dot(cov_matrix_inv.T[k],
                                       cl_da_truncated_array[i].T[k+2])
                                for k in range(cl_orig.shape[0]-2)])
        # print('sum sq in trance = ', sum(sq_in_trace))

        in_trace = np.array([np.dot(sq_in_trace[k], sq_in_trace[k])
                             for k in range(sq_in_trace.shape[0])])
        # print('sum in trace =', sum(in_trace))

        if np.shape(np.shape(cl_rot_noise_truncated_array[i].T[2:]))[0] == 1:
            trace_fisher = in_trace
        else:
            trace_fisher = np.trace(in_trace, axis1=1, axis2=2)

        fisher_element = [0., 0.]

        fisher = 0
        for l in range(2, len(cl_orig)):
            fisher += (2*l + 1) * 0.5 * f_sky * trace_fisher[l-2]
            fisher_element.append((2*l + 1) * 0.5 * f_sky * trace_fisher[l-2])

        fisher_element = np.array(fisher_element)
        truncated_fisher_list.append(fisher)
        truncated_fisher_element_list.append(fisher_element)

    total_fisher, total_fisher_element = fisher_angle(
        cl_orig, angle, cl_rot=cl_rot_noise, f_sky=f_sky,
        return_elements=True, raw_cl=True, raw_cl_rot=True)

    truncated_fisher_list.append(total_fisher)
    truncated_fisher_element_list.append(total_fisher_element)

    truncated_fisher_array = np.array(truncated_fisher_list)
    truncated_fisher_element_array = np.array(truncated_fisher_element_list)

    if return_elements:
        return truncated_fisher_array, truncated_fisher_element_array
    return truncated_fisher_array


def get_foreground_spectrum(nside, nu_u,
                            mask_file="HFI_Mask_GalPlane-apo2_2048_R2.00.fits",
                            mask_field=2):
    nu = nu_u.to(u.GHz).value
    l_max = nside * 3
    sky_config = {'dust': pysm.nominal.models(
        'd0', nside), 'synchrotron': pysm.nominal.models('s0', nside)}
    sky = pysm.Sky(sky_config)
    dust_signal = sky.dust(nu)
    synchrotron_signal = sky.synchrotron(nu)

    mask = np.array(hp.ud_grade(hp.read_map(mask_file,
                                            field=mask_field).astype(np.bool),
                                nside))

    proper_dust_masked = hp.ma(dust_signal)
    proper_dust_masked.mask = np.logical_not(mask)

    proper_sync_masked = hp.ma(synchrotron_signal)
    proper_sync_masked.mask = np.logical_not(mask)

    cl_dust_masked = np.array(
        [hp.anafast(proper_dust_masked, lmax=l_max).T[l, :] *
         l*(l+1)/(2*np.pi) for l in range(l_max)])
    cl_synchrotron_masked = np.array(
        [hp.anafast(proper_sync_masked, lmax=l_max).T[l, :] *
         l*(l+1)/(2*np.pi) for l in range(l_max)])

    return {'dust': cl_dust_masked, 'synchrotron': cl_synchrotron_masked}


def likelihood(spectra_cov, spectra_data, raw_spectra=False, f_sky=1):

    if raw_spectra is False:
        spectra_cov = cl_normalisation(spectra_cov)
        spectra_data = cl_normalisation(spectra_data)
    # TODO : covariance matrix function (put power spectra output function)
    cov_matrix = np.array(
        [[spectra_cov[:, 0], spectra_cov[:, 3], spectra_cov[:, 5]],
         [spectra_cov[:, 3], spectra_cov[:, 1], spectra_cov[:, 4]],
         [spectra_cov[:, 5], spectra_cov[:, 4], spectra_cov[:, 2]]])

    data_matrix = np.array(
        [[spectra_data[:, 0], spectra_data[:, 3], spectra_data[:, 5]],
         [spectra_data[:, 3], spectra_data[:, 1], spectra_data[:, 4]],
         [spectra_data[:, 5], spectra_data[:, 4], spectra_data[:, 2]]])

    cov_matrix_inv = np.linalg.inv(cov_matrix.T[2:]).T
    cov_dot_data = np.array([np.dot(cov_matrix_inv[:, :, k-2],
                                    data_matrix[:, :, k])
                             for k in range(2, len(spectra_cov))])
    # in_trace = np.log(cov_matrix.T[2:]) + cov_dot_data
    # trace_likelihood = np.trace( in_trace, axis1=1, axis2=2 )
    in_trace = cov_dot_data
    trace_likelihood = np.log(np.linalg.det(
        cov_matrix.T[2:])) + np.trace(in_trace, axis1=1, axis2=2)

    likelihood = 0

    for l in range(2, len(spectra_cov)):
        # for l in range(900, 1200):
        likelihood += (2*l + 1)*0.5 * f_sky * trace_likelihood[l-2]
        if likelihood == np.inf:
            # TODO : exception/error
            print('likelihood = np.inf !!')
            break

    return likelihood


def likelihood_normalisation(likelihood):
    min_likelihood = min(likelihood)
    # max_likelihood = max(likelihood)
    return (likelihood - min_likelihood)
    # / (max_likelihood)# - min_likelihood)


def likelihood_sweep(dict, cov_key, data_key, angle_array):
    # TODO: WHAT IS THIS DOING ?
    likelihood_list = []
    for angle in angle_array:
        likelihood_list.append(likelihood(covariance, data[angle]))

    return 0


def myindex1(lst, target):
    for index, item in enumerate(lst):
        if item == target:
            return [index]
        if isinstance(item, (list, tuple)):
            path = myindex(item, target)
            if path:
                return [index] + path
    return []


def myindex(lst, target):
    for index, item in enumerate(lst):
        if item == target:
            return [index]
        if isinstance(item, str):  # or 'basestring' in 2.x
            return []
        try:
            path = myindex(item, target)
        except TypeError:
            pass
        else:
            if path:
                return [index] + path
    return []


def get_cl_noise(nl, instrument='SAT'):
    nl_inv = 1/nl
    if instrument == 'SAT':
        frequencies = V3.so_V3_SA_bands()
    if instrument == 'LAT':
        frequencies = V3.so_V3_LA_bands()

    components = [CMB(), Dust(150.), Synchrotron(20.)]
    A = MixingMatrix(*components)
    A_ev = A.evaluator(frequencies)
    res = [1.59, 20, -3]
    A_ = A_ev(res)
    AtNA = np.einsum('fi, fl, fj -> lij', A_, nl_inv, A_)
    inv_AtNA = np.linalg.inv(AtNA)

    return inv_AtNA.swapaxes(-3, -1)


def map_rotation(map, rotation_angle):
    """Rotates the Q,U maps given a certain angle. Handles multiple frequecies
    and frequency dependent rotation angle.
    Parameters
    ----------
    map : numpy array, [frequency, I/Q/U, pixels] OR [I/Q/U, pixels]
        sky map typically given by fgbuster/pysm. Contains temperature and
        polarisation maps. Can contain frequency maps also (optionnal)
    rotation_angle : float or numpy array
        Angle by which to rotate the map. Might be an array with same length as
        number of frequencies if frequency dependent rotation.

    Returns
    -------
    numpy array
        new map array of same shape as argument but with now mixed Q & U
        component according to the rotation.

    """
    len_map = len(np.shape(map))
    frequency_dependent = isinstance(rotation_angle, np.ndarray)

    if 1-frequency_dependent and len_map == 3:  # TODO: why ?
        print('Check map rotation !!!')
        rotation_angle = np.ones(len(map[:, 0, 0])) * rotation_angle
    if len_map == 3:
        print('if map rot')
        map_rotated = np.empty(np.shape(map))

        for i in range(len(map[:, 0, 0])):
            Qrot = np.cos(2*rotation_angle[i])*map[i, 1, :] + \
                np.sin(2 * rotation_angle[i])*map[i, 2, :]
            Urot = - np.sin(2*rotation_angle[i])*map[i, 1, :] + \
                np.cos(2 * rotation_angle[i])*map[i, 2, :]
            map_rotated[i, 0] = map[i, 0]
            map_rotated[i, 1] = Qrot
            map_rotated[i, 2] = Urot

    else:
        print('else map rot')
        map_rotated = np.empty(np.shape(map))

        Qrot = np.cos(2*rotation_angle)*map[1, :] - \
            np.sin(2 * rotation_angle)*map[2, :]
        Urot = np.sin(2*rotation_angle)*map[1, :] + \
            np.cos(2 * rotation_angle)*map[2, :]
        for i in range(len(map[0][:])):
            map_rotated[0, i] = map[0, i]
        map_rotated[1] = Qrot
        map_rotated[2] = Urot

    return map_rotated


def get_dr_cov_bir_EB(cl_prim_r1, alpha):
    dr_cov_matrix = np.array(
        [[cl_prim_r1[:, 2]*(np.sin(2 * alpha)**2), -cl_prim_r1[:, 2]*np.sin(4*alpha)*0.5],
         [-cl_prim_r1[:, 2]*np.sin(4*alpha)*0.5, cl_prim_r1[:, 2]*(np.cos(2 * alpha)**2)]])
    return dr_cov_matrix


def fisher(cov, deriv, f_sky, cov2=None, deriv2=None, return_elements=False):
    if cov2 is None:
        cov2 = cov
    if deriv2 is None:
        deriv2 = deriv
    print('cov_matrix 48', cov[:, :, 18])
    cov_matrix_inv1 = np.linalg.inv(cov.T[2:])
    cov_matrix_inv1 = cov_matrix_inv1.T
    print('cov_matrix_inv1 46', cov_matrix_inv1[:, :, 16])

    cov_matrix_inv2 = np.linalg.inv(cov2.T[2:])
    cov_matrix_inv2 = cov_matrix_inv2.T

    sq_in_trace1 = np.array(
        [np.dot(cov_matrix_inv1[:, :, k], deriv[:, :, k+2])
         for k in range(cov_matrix_inv1.shape[2])])
    # print('sum sq in trace', sum(sq_in_trace1))
    print('sq_in_trace1 16', sq_in_trace1[16])

    sq_in_trace2 = np.array(
        [np.dot(cov_matrix_inv2[:, :, k], deriv2[:, :, k+2])
         for k in range(cov_matrix_inv2.shape[2])])

    in_trace = np.array([np.dot(sq_in_trace1[k, :, :], sq_in_trace2[k, :, :])
                         for k in range(sq_in_trace1.shape[0])])
    print('sum in trace', sum(in_trace))
    print('in_trace 16 = ', in_trace[16])

    trace_fisher = np.trace(in_trace, axis1=1, axis2=2)
    print('trace_fisher 16 = ', trace_fisher[16])
    fisher_element = [0., 0.]

    fisher = 0
    for l in range(2, cov.shape[2]):
        fisher += (2*l + 1) * 0.5 * f_sky * trace_fisher[l-2]
        fisher_element.append((2*l + 1) * 0.5 * f_sky * trace_fisher[l-2])
    fisher_element = np.array(fisher_element)
    if return_elements:
        return fisher, fisher_element
    return fisher


"""""
----------------------------Function purgatory-------------------------------
"""""


def test_likelihood(log_cov_matrix, data_matrix, inv_matrix, f_sky):
    # cov_matrix_inv = np.linalg.inv(cov_matrix.T[2:]).T
    cov_matrix_inv = inv_matrix

    cov_dot_data = np.array([np.dot(cov_matrix_inv[:, :, k],
                                    data_matrix[:, :, k])
                             for k in range(np.shape(log_cov_matrix)[-1])])
    # in_trace = np.log(cov_matrix.T[2:]) + cov_dot_data
    # trace_likelihood = np.trace( in_trace, axis1=1, axis2=2 )
    in_trace = cov_dot_data
    # np.log(np.linalg.det(log_cov_matrix.T[2:]))
    trace_likelihood = log_cov_matrix + np.trace(in_trace, axis1=1, axis2=2)

    likelihood = 0

    for l in range(2, np.shape(log_cov_matrix)[-1]):
        # for l in range(900, 1200):
        likelihood += (2*l + 1)*0.5 * f_sky * trace_likelihood[l-2]
        if likelihood == np.inf:
            # TODO : exception/error
            print('likelihood = np.inf !!')
            break

    return likelihood
