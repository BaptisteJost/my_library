# import numdifftools as nd
from scipy.linalg import logm
import IPython
# import sys
# sys.path.insert(1, '/home/baptiste/Documents/stage_npac')
import bjlib.lib_project as lib
# import plot_project as plotpro
# import math
from fgbuster import visualization as visu

from astropy import constants as c
from astropy import units as u
import matplotlib.pyplot as plt
# import healpy as hp
import numpy as np
import V3calc as V3
import copy
import time
# << << << < HEAD
# == == == =
# >>>>>> > parent of af9e50b... save


class power_spectra_obj:
    def __init__(self, spectra, ell, normalisation=False):
        self._spectra = spectra
        if type(ell) == list:
            self._ell = np.array(ell)
        else:
            self._ell = ell
        self._normalisation = normalisation

    @property
    def spectra(self):
        return self._spectra

    @spectra.setter
    def spectra(self, new_spectra):
        # print('Modifying spectra...')
        self._spectra = new_spectra

    @property
    def ell(self):
        return self._ell

    @property
    def normalisation(self):
        return self._normalisation

    @normalisation.setter
    def normalisation(self, new_normalisation):
        if self.normalisation == new_normalisation:
            if self.normalisation:
                print('The spectra are already normalised by ell(ell + 1)/(2 pi)')
            else:
                print('The spectra are already raw (not normalised)')
        else:
            if new_normalisation:
                print('Spectra are being normalised')
                if len(np.shape(self.spectra)) == 1:
                    # print('blou')
                    new_cl = np.array(
                        [self.spectra.T[i] *
                            (((self.ell[i]+1)*self.ell[i])/(2*np.pi))
                            for i in range(len(self.ell))])
                else:
                    new_cl = np.array(
                        [self.spectra.T[:, i] *
                            (((self.ell[i]+1)*self.ell[i])/(2*np.pi))
                            for i in range(len(self.ell))])
            else:
                print('Spectra are being un-normalised')
                if len(np.shape(self.spectra)) == 1:
                    new_cl = np.array(
                        [self.spectra.T[i] /
                            (((self.ell[i]+1)*self.ell[i])/(2*np.pi))
                            for i in range(len(self.ell))])
                else:
                    new_cl = np.array(
                        [self.spectra.T[:, i] /
                            (((self.ell[i]+1)*self.ell[i])/(2*np.pi))
                            for i in range(len(self.ell))])

            self.spectra = new_cl
            # TODO: comment faire pour ne modifier spectra que dans ce cas?
            self._normalisation = new_normalisation

    @ell.setter
    def ell(self, new_ell):
        if len(new_ell) > len(self.ell):
            print('Cannot go to a bigger set of ell')

        else:
            test_ = copy.deepcopy(self.ell)
            for i in range(len(new_ell)):
                test_[i] = copy.deepcopy(self.ell[i])-copy.deepcopy(new_ell[i])
            test = sum(test_)
            if test < 0:
                print('ERROR: new_ell must be a subset of ell')
                # TODO: check viability of this test
            else:
                # print('Ell are being modified')
                new_ell_index = [self.ell.tolist().index(element) for element in new_ell]
                # self.new_ell_index = new_ell_index

                new_cl = self.spectra[new_ell_index]
                self.spectra = new_cl
                self._ell = np.array(new_ell)

        # print('WARNING: spectra normalisation changed while spectra modification')
        # print('Be sure to check consistency')


class power_spectra_operation:
    def __init__(self, l_max=5000, instrument='SAT', r=0.0,
                 rotation_angle=0*u.rad, powers_name='total',
                 Bfield_strength=45 * 10**-13 * u.T,
                 e_density_cmb=10**8 / u.m**3,
                 traveled_length_at_recom=(100 * 10**3 * u.year).to(u.s)*c.c,
                 ):
        self.l_max = l_max
        self._instrument = instrument
        self.r = r
        self.rotation_angle = rotation_angle
        self.powers_name = powers_name
        self.Bfield_strength = Bfield_strength
        self.e_density_cmb = e_density_cmb
        self.traveled_length_at_recom = traveled_length_at_recom

        if self.instrument == 'SAT':
            if self.l_max <= 500:
                print('WARNING: l_max for power spectra generation',
                      'is too low for SAT ')
            self.l_max_instru = 300
            self.l_min_instru = 30
            self.fsky = 0.1
            self.obs_time = 1  # in years

        if self.instrument == 'LAT':
            if self.l_max <= 5500:
                print('WARNING: l_max for power spectra generation',
                      'is too low for LAT ')
            self.l_max_instru = 5000
            self.l_min_instru = 100
            self.fsky = 0.4

    def _get_instrument(self):
        return self._instrument

    instrument = property(_get_instrument)

    def get_spectra(self, r=None):
        if r is None:
            pars, results, powers = lib.get_basics(l_max=self.l_max, ratio=self.r,
                                                   raw_cl=True)
        else:
            pars, results, powers = lib.get_basics(l_max=self.l_max, ratio=r,
                                                   raw_cl=True)
            self.r = r
        del pars, results
        ell = np.array([i for i in range(self.l_max)])
        self.spectra = power_spectra_obj(powers[self.powers_name], ell=ell,
                                         normalisation=0)

    def get_frequencies(self):
        if self.instrument == 'LiteBIRD':
            frequencies = np.array(
                [40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9,
                    140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1]) * u.GHz
            self.frequencies = frequencies
        elif self.instrument == 'SAT':
            frequencies = V3.so_V3_SA_bands() * u.GHz
            self.frequencies = frequencies
        elif self.instrument == 'LAT':
            frequencies = V3.so_V3_LA_bands() * u.GHz
            self.frequencies = frequencies
        else:
            print('Instrument not recognized, allowed instruments are:')
            print('LiteBIRD \n SAT \n LAT')
            print('By default LiteBIRD has been chosen')
            frequencies = np.array(
                [40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9,
                 140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1]) * u.GHz
            self.frequencies = frequencies

    def spectra_rotation(self, angle=None):
        if angle is None:
            angle = self.rotation_angle
        cl_rot = lib.cl_rotation(self.spectra.spectra, angle)
        self.cl_rot = power_spectra_obj(cl_rot, self.spectra.ell)

    def get_faraday_angles(self):
        fd_cst = (c.e.si**3) / (8 * (np.pi**2) * c.eps0 * (c.m_e**2) *
                                (c.c**3))

        fd_fct = (fd_cst * self.Bfield_strength * self.e_density_cmb *
                  self.traveled_length_at_recom *
                  (c.c / self.frequencies)**2).decompose() % (2*np.pi)
        self.faraday_angles = fd_fct

    def split_spectra(self):
        spectra_number = len(self.frequencies)
        split_spectra = np.array([self.spectra.spectra / spectra_number
                                  for k in range(spectra_number)])
        self.split_spectra = split_spectra

    def get_faraday_spectra(self):
        faraday_list = []
        for k in range(len(self.frequencies)):
            faraday_rotated = lib.cl_rotation(self.spectra.spectra/len(self.frequencies),
                                              self.faraday_angles[k]*u.rad)
            faraday_list.append(faraday_rotated)
            del faraday_rotated
        faraday_list = np.array(faraday_list)
        # print('shape faraday_list = ', np.shape(faraday_list))
        faraday_tot = np.sum(faraday_list, 0)
        self.cl_faraday = faraday_tot

    def get_noise_dict(self):
        noise_dict = {}
        if self.instrument == 'SAT':
            for i in [0, 1, 2]:
                for j in [0, 1, 2]:
                    print('j', j)
                    noise_str = '{}'.format(i)+'{}'.format(j)
                    # print('le noise string = ', len(noise_str))
                    noise_nl = V3.so_V3_SA_noise(i, j, self.obs_time, self.fsky,
                                                 self.l_max_instru)[1]
                    noise_cl = lib.get_cl_noise(noise_nl, instrument=self.instrument)[0, 0]
                    noise_dict[noise_str] = np.append([0, 0], noise_cl)

        if self.instrument == 'LAT':
            for i in [0, 1, 2]:
                noise_str = '{}'.format(i)
                print('d')
                # try:
                noise_nl = V3.so_V3_LA_noise(i,  self.fsky, self.l_max_instru)[2]
                # print('Shape noise_nl = ', np.shape(noise_nl))
                noise_cl = lib.get_cl_noise(noise_nl, instrument=self.instrument)[0, 0]

                noise_dict[noise_str] = np.append([0, 0], noise_cl)

        self.noise_dict = noise_dict

    def get_noise(self, key='00'):

        if self.instrument == 'SAT':
            V3_results = V3.so_V3_SA_noise(int(key[0]), int(key[1]),
                                           self.obs_time, 0.1,
                                           self.l_max_instru)
            noise_nl = V3_results[1]
            ell_noise = V3_results[0]
            noise_cl = lib.get_cl_noise(noise_nl, instrument=self.instrument)[0, 0]
            noise_cl = np.append([0, 0], noise_cl)
            ell_noise = np.append([0, 1], ell_noise)

        if self.instrument == 'LAT':
            V3_results = V3.so_V3_LA_noise(int(key[0]),  self.fsky,
                                           self.l_max_instru)
            noise_nl = V3_results[2]
            ell_noise = V3_results[0]
            noise_cl = lib.get_cl_noise(noise_nl,
                                        instrument=self.instrument)[0, 0]

            noise_cl = np.append([0, 0], noise_cl)

        self.noise_cl = power_spectra_obj(noise_cl, ell=ell_noise,
                                          normalisation=0)
        self.noise_key = key

    def spectra_addition(self, spectra1, spectra2, list_addition=[]):
        if len(spectra1.ell) != len(spectra2.ell):
            print('ERROR: the 2 spectra don\'t have the same size ')
        elif spectra1.normalisation != spectra2.normalisation:
            print('ERROR: the 2 spectra don\'t have the same normalisation')
        else:
            if list_addition == []:
                added_spectra = spectra1.spectra + spectra2.spectra
            else:
                added_spectra = []
                if len(list_addition[0]) != len(list_addition[0]):
                    print('ERROR: list_addition ill defined')
                else:
                    for num_spec in range(len(list_addition[0])):
                        if list_addition[1][num_spec] == -1:
                            added_spectra.append(
                                spectra1.spectra[:, list_addition[0][num_spec]])
                        else:
                            added_spectra.append(
                                spectra1.spectra[:, list_addition[0][num_spec]] +
                                spectra2.spectra[:, list_addition[1][num_spec]])
            return added_spectra

    def get_instrument_spectra(self):
        spectra = copy.deepcopy(self.cl_rot)
        # TODO: check auto which spectra to use, spectra, cl_rot, faraday...
        new_ell = [i for i in range(self.l_min_instru, self.l_max_instru)]
        spectra.ell = new_ell
        if hasattr(self, 'noise_cl'):
            self.noise_cl.ell = new_ell
            if spectra.normalisation != self.noise_cl.normalisation:
                print('ERROR: noise and cmb spectra don\'t have the same normalisation')
            else:
                added_spectra = copy.deepcopy(spectra.spectra).T
                added_spectra[0] += self.noise_cl.spectra/2
                added_spectra[1] += self.noise_cl.spectra
                added_spectra[2] += self.noise_cl.spectra
                # added_spectra[4] += self.noise_cl.spectra

                self.instru_spectra = power_spectra_obj(
                    added_spectra.T, spectra.ell,
                    normalisation=spectra.normalisation)
        else:
            print('Warning, no noise included in intru_spectra. Use get_noise() if needed.')
            self.instru_spectra = power_spectra_obj(
                spectra.spectra, spectra.ell,
                normalisation=spectra.normalisation)


def fisher_pws(cov, deriv, f_sky, ell=None, cov2=None, deriv2=None, return_elements=False):
    if np.any(ell == 0) or np.any(ell == 1):
        print('ERROR: 0th or first multipole can be used in fisher')
        return
    if cov2 is None:
        cov2 = cov
    if deriv2 is None:
        deriv2 = deriv
    if ell is None:
        ell_inter1 = np.intersect1d(cov.ell, deriv.ell)
        ell_inter2 = np.intersect1d(cov2.ell, deriv2.ell)
        ell_inter = np.intersect1d(ell_inter1, ell_inter2)
    else:
        ell_inter = ell
    if sum(ell_inter == 0) or sum(ell_inter == 1):
        print('WARNING: in fisher_pws() ell = 0 or 1 are present which could',
              ' lead to a wrong Fisher estimation')

    cov1_index = [cov.ell.tolist().index(element) for element in ell_inter]
    cov2_index = [cov2.ell.tolist().index(element) for element in ell_inter]
    deriv1_index = [deriv.ell.tolist().index(element) for element in ell_inter]
    deriv2_index = [deriv2.ell.tolist().index(element) for element in ell_inter]

    deriv = deriv.spectra[deriv1_index]
    deriv2 = deriv2.spectra[deriv2_index]
    if len(np.shape(deriv)) == 1:
        print('coucou')
        cov_matrix_inv1 = 1/cov.spectra[cov1_index]
        cov_matrix_inv2 = 1/cov.spectra[cov2_index]
        sq_in_trace1 = cov_matrix_inv1 * deriv
        sq_in_trace2 = cov_matrix_inv2 * deriv2
        in_trace = sq_in_trace1*sq_in_trace2
        trace_fisher = in_trace

    else:
        print('welcome')
        cov_matrix_inv1 = np.linalg.inv(cov.spectra[cov1_index])

        cov_matrix_inv2 = np.linalg.inv(cov2.spectra[cov2_index])

        sq_in_trace1 = np.einsum('kij,kjl->kil', cov_matrix_inv1, deriv)

        sq_in_trace2 = np.einsum('kij,kjl->kil', cov_matrix_inv2, deriv2)

        in_trace = np.array([np.dot(sq_in_trace1[k, :, :], sq_in_trace2[k, :, :])
                             for k in range(len(ell_inter))])

        trace_fisher = np.trace(in_trace, axis1=1, axis2=2)

    fisher_element = []
    fisher = 0
    l_counter = 0
    for l in ell_inter:
        fisher_ell = (2*l + 1) * 0.5 * f_sky * trace_fisher[l_counter]
        fisher += fisher_ell
        fisher_element.append(fisher_ell)

        l_counter += 1

    fisher_element = np.array(fisher_element)
    if return_elements:
        return fisher, fisher_element
    return fisher


def likelihood_pws(model, data, f_sky=0.1, ell=None, return_elements=False):
    if ell is None:
        ell_inter = np.intersect1d(model.ell, data.ell)
    else:
        ell_inter = ell
    if sum(ell_inter == 0) or sum(ell_inter == 1):
        print('WARNING: in likelihood_pws() ell = 0 or 1 are present which',
              ' could lead to a wrong Fisher estimation')
    if model.normalisation or data.normalisation:
        print('WARNING: in likelihood_pws() input matrices should not be',
              ' normalised. e.g. by (2l + 1)/2')

    model_index = [model.ell.tolist().index(element) for element in ell_inter]
    data_index = [data.ell.tolist().index(element) for element in ell_inter]

    model_reduced = model.spectra[model_index]
    data_reduced = data.spectra[data_index]
    if len(np.shape(model_reduced)) == 1:
        # print('1 dim in likelihood_pws()')
        model_inverse = 1/model_reduced
        # if np.isnan(model_inverse).any():
        #     print('nan in model inverse')
        # if np.isinf(model_inverse).any():
        #     print('inf in model inverse')
        if np.isnan(data_reduced).any():
            print('!!!!!! nan in data_reduced')
        if np.isinf(data_reduced).any():
            print('!!!!!! inf in data_reduced')

        Cm1D = model_inverse * data_reduced
        # likelihood_trace = Cm1D + np.log(np.abs(model_reduced))
        # if np.all(model_reduced <= 0):
        #     likelihood_trace = Cm1D - np.log(-model_reduced)
        # else:
        #     likelihood_trace = Cm1D + np.log(model_reduced)

        likelihood_trace = Cm1D + np.log(model_reduced)

        # print('Cm1D = ', Cm1D)

        # if np.isnan(likelihood_trace).any():
        # print('nan in likelihood_trace')
        # print('log model_reduced', np.log(model_reduced))
        # print(likelihood_trace)
        # if np.isinf(likelihood_trace).any():
        # print('inf in likelihood_trace')
        # if np.isinf(np.log(model_reduced)).any():
        # print('inf in log model')

        # valid_points = np.logical_or(np.isnan(likelihood_trace), np.isinf(likelihood_trace))
        # ell_inter = ell_inter[valid_points]
        # if len(ell_inter) == 0:
        #     print('!!!!!!!!!!!!! no valid ell in likelihood!!!!!!!!!!!o')
        #     likelihood = 0  # np.nan
        # else:
        #     print(valid_points)
        #     likelihood_trace = likelihood_trace[valid_points]

        # print('Check identité ', np.sum(model_reduced.dot(model_inverse)))

    else:
        # print('likelihood uses matrices')
        model_inverse = np.linalg.inv(model_reduced)

        Cm1D = np.einsum('kij,kjl->kil', model_inverse, data_reduced)
        log_det = np.log(np.linalg.det(model_reduced))
        if 1:  # np.any(np.abs(log_det) <= 1e-6):
            # IPython.embed()
            # print('log matrix used instead of det as it was too small')
            log_C = [logm(model_reduced[k]) for k in range(len(ell_inter))]
            likelihood_trace = np.trace(Cm1D + log_C,
                                        axis1=1, axis2=2)
        else:
            likelihood_trace = np.trace(Cm1D,
                                        axis1=1, axis2=2) + log_det
        # print('Check identité ', np.sum(np.einsum('kij,kjl->kil', model_reduced, model_inverse), 0))
    likelihood = 0
    ell_counter = 0
    likelihood_element = []
    for l in ell_inter:
        likelihood_ell = f_sky*(2*l + 1)*0.5 * likelihood_trace[ell_counter]
        if np.isnan(likelihood_ell):
            print('!!!!!!! Nan in likelihood_ell at ell = {}'.format(l))
            print(len(np.shape(model_reduced)))
            print('model = ', model.spectra)
            print('data = ', data.spectra)
            print('Cm1D = ', Cm1D)
            print('np.log(model_reduced) = ', np.log(model_reduced))

        likelihood += likelihood_ell
        likelihood_element.append(likelihood_ell)
        ell_counter += 1
    if return_elements:
        return likelihood, likelihood_element  # , Cm1D
    return likelihood  # , Cm1D


def likelihood_for_hessian_a(angle, model_, data_matrix, bin, nside, f_sky, spectra_used='all'):
    angle = angle * u.rad
    print('angle likelihood = ', angle)
    # r = param_array[1]
    # IPython.embed()
    # model.get_spectra(r=r)
    model = copy.deepcopy(model_)
    model.spectra_rotation(angle)

    # print("WARNING: lmin and lmax instrument have been changed in power_spectra object in likelihood_for_hessian_a()")
    model.l_min_instru = 0
    model.l_max_instru = 3*nside
    model.get_noise()
    model.get_instrument_spectra()
    # model_spectra = model.instru_spectra.spectra
    model_spectra = bin.bin_cell(model.instru_spectra.spectra[:3*nside].T).T
    # print('model_spectra =', model_spectra)
    # model_spectra = bin.bin_cell(copy.deepcopy(model.cl_rot.spectra[:3*nside].T)).T
    if spectra_used == 'all':
        model_matrix = power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, bin.get_effective_ells())
        # model_matrix = power_spectra_obj(np.array(
        #     [[model_spectra[:, 1], model_spectra[:, 4]],
        #      [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
    elif spectra_used == 'EE':
        # print('EE !')
        model_matrix = power_spectra_obj(np.array(
            model_spectra[:, 1]).T, bin.get_effective_ells())
        # model_matrix = power_spectra_obj(np.array(
        #     model_spectra[:, 1]).T,  model.instru_spectra.ell)
        # print(len(np.shape(model_matrix.spectra)))
    elif spectra_used == 'BB':
        # print('BB !')
        model_matrix = power_spectra_obj(np.array(
            model_spectra[:, 2]).T, bin.get_effective_ells())
    elif spectra_used == 'EB':
        # print('EB !')
        model_matrix = power_spectra_obj(np.array(
            model_spectra[:, 4]).T, bin.get_effective_ells())
    likelihood_value = likelihood_pws(
        model_matrix, data_matrix, f_sky, return_elements=False)
    return likelihood_value  # , model_matrix, data_matrix, Cm1D, likelihood_element


def main():

    r_data = 0.0
    rotation = 0.00 * u.rad
    noise_model = '00'

    """====================FISHER COMPUTATION================================="""

    test = power_spectra_operation(r=r_data, rotation_angle=rotation, l_max=700)
    # Bfield_strength=45 * 10**-13 * u.T)
    test.get_spectra()
    test.spectra_rotation()
    test.get_noise(noise_model)
    test.get_instrument_spectra()
    noisy_spectra = test.instru_spectra.spectra
    # noisy_spectra = test.cl_rot.spectra

    if np.shape(noisy_spectra)[1] == 4:
        # TODO: use diagonal instead of hardcoded ell
        print('olaa')
        cov_matrix = power_spectra_obj(np.array(
            [[noisy_spectra[:, 1], np.zeros(270)],
             [np.zeros(270), noisy_spectra[:, 2]]]).T, test.instru_spectra.ell)
    else:
        print('halloo')
        cov_matrix = power_spectra_obj(np.array(
            [[noisy_spectra[:, 1], noisy_spectra[:, 4]],
             [noisy_spectra[:, 4], noisy_spectra[:, 2]]]).T, test.instru_spectra.ell)
    deriv1 = power_spectra_obj(lib.cl_rotation_derivative(
        test.spectra.spectra, rotation), test.spectra.ell)
    deriv_matrix1 = power_spectra_obj(np.array(
        [[deriv1.spectra[:, 1], deriv1.spectra[:, 4]],
         [deriv1.spectra[:, 4], deriv1.spectra[:, 2]]]).T, deriv1.ell)

    pw_r1 = power_spectra_operation(r=1, rotation_angle=rotation,
                                    l_max=700, powers_name='unlensed_total')
    pw_r1.get_spectra()

    deriv_matrix2 = power_spectra_obj(lib.get_dr_cov_bir_EB(
        pw_r1.spectra.spectra, rotation).T, pw_r1.spectra.ell)

    fishaa = fisher_pws(cov_matrix, deriv_matrix1, 0.1)
    fishrr = fisher_pws(cov_matrix, deriv_matrix2, 0.1)
    fishar = fisher_pws(cov_matrix, deriv_matrix1, 0.1, deriv2=deriv_matrix2)
    fishra = fisher_pws(cov_matrix, deriv_matrix2, 0.1, deriv2=deriv_matrix1)
    # IPython.embed()
    # cov_matrixBB = power_spectra_obj(cov_matrix.spectra[:, 1, 1], cov_matrix.ell)
    # deriv_matrixBB = power_spectra_obj(deriv_matrix2.spectra[:, 1, 1], deriv_matrix2.ell)
    # fishrrBB = fisher_pws(cov_matrixBB, deriv_matrixBB, 0.1)

    lensed_scalar = power_spectra_operation(r=1, l_max=700, rotation_angle=rotation,
                                            powers_name='lensed_scalar')
    lensed_scalar.get_spectra()
    data2 = power_spectra_operation(r=r_data, l_max=700, rotation_angle=rotation)
    data2.get_spectra()
    data2.spectra.spectra[:, 2] = r_data*pw_r1.spectra.spectra[:, 2] + \
        lensed_scalar.spectra.spectra[:, 2]
    data2.get_noise(noise_model)
    data2.spectra_rotation()
    data2.get_instrument_spectra()
    cov_matrixBB = power_spectra_obj(data2.instru_spectra.spectra[:, 2], data2.instru_spectra.ell)
    deriv_matrixBB = power_spectra_obj(deriv_matrix2.spectra[:, 1, 1], deriv_matrix2.ell)
    fishrrBB = fisher_pws(cov_matrixBB, deriv_matrixBB, 0.1)

    fisher_matrix = np.array([[fishaa, fishar], [fishar, fishrr]])
    sigma_sq_matrix = np.linalg.inv(fisher_matrix)

    # IPython.embed()

    visu.corner_norm([rotation.value, r_data], sigma_sq_matrix, labels=[r'$\alpha$', r'$r$'])
    plt.show()

    IPython.embed()

    """========================= LIKELIHOOD ON ALPHA ========================="""

    data = power_spectra_operation(r=r_data, l_max=700, rotation_angle=rotation)
    data.get_spectra()
    data.get_noise(noise_model)
    data.spectra_rotation()
    data.get_instrument_spectra()
    data_spectra = data.instru_spectra.spectra
    data_matrix = power_spectra_obj(
        np.array([[data_spectra[:, 1], data_spectra[:, 4]],
                  [data_spectra[:, 4], data_spectra[:, 2]]]).T,
        data.instru_spectra.ell)

    model = power_spectra_operation(r=r_data, l_max=700, rotation_angle=0.0*u.rad)
    model.get_spectra()
    model.get_noise(noise_model)
    model.spectra_rotation()
    model.get_instrument_spectra()

    min_angle = rotation.value - 5*(1/np.sqrt(fishaa))
    max_angle = rotation.value + 5*(1/np.sqrt(fishaa))
    nstep_angle = 100
    angle_grid = np.arange(min_angle, max_angle,
                           (max_angle - min_angle)/nstep_angle)*u.radian
    idx1 = (np.abs(angle_grid.value - rotation.value)).argmin()
    print('angle_grid check ', angle_grid[idx1])
    likelihood_values = []

    for angle in angle_grid:
        model.spectra_rotation(angle)
        model.get_instrument_spectra()
        model_spectra = model.instru_spectra.spectra

        model_matrix = power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)

        likelihood_val = likelihood_pws(model_matrix, data_matrix, 0.1)
        likelihood_values.append(likelihood_val)

    """======================= PLOT LIKELIHOOD ON ALPHA ======================="""

    likelihood_norm = np.array(likelihood_values) - min(likelihood_values)
    plt.plot(angle_grid, np.exp(-likelihood_norm),
             label='Likelihood -min(likelihood)')

    fisher_gaussian = np.exp(-(angle_grid.value - rotation.value)**2 / (2*sigma_sq_matrix[0, 0]))
    # fisher_gaussian_norm = fisher_gaussian - min(fisher_gaussian)
    plt.plot(angle_grid, fisher_gaussian, label='fisher gaussian')

    plt.xlabel(r'$\alpha$ in radian')
    plt.ylabel('Likelihood')
    plt.title(r'Likelihood on $\alpha$ with true $\alpha$={}'.format(rotation))
    plt.legend()
    plt.show()

    print('gradient fisher = ', np.gradient(np.gradient(fisher_gaussian))[idx1])
    print('gradient likelihood = ', np.gradient(np.gradient(np.exp(-likelihood_norm)))[idx1])

    """================= GRIDDING LIKELIHOOD ON ALPHA AND R ================="""

    likelihood_values_ar = []
    likelihood_values_ar_2D = []

    model = power_spectra_operation(r=0, l_max=700, rotation_angle=rotation)
    model.get_spectra()
    model.get_noise(noise_model)
    model.spectra_rotation()
    model.get_instrument_spectra()
    min_r = r_data-5*(1/np.sqrt(fishrr))
    max_r = r_data+5*(1/np.sqrt(fishrr))
    nstep_r = 100
    r_grid = np.arange(min_r, max_r, (max_r - min_r)/nstep_r)

    idx2 = (np.abs(r_grid - r_data)).argmin()
    print('r_grid check', r_grid[idx2])

    # angle_grid = np.array([(0.33*u.deg).to(u.rad).value])*u.rad
    # nstep_angle = 1
    # idx1 = 0
    # lensed_scalar = power_spectra_operation(r=1, l_max=700, rotation_angle=rotation,
    #                                         powers_name='lensed_scalar')
    # lensed_scalar.get_spectra()

    # lensed_scalar = lensed_scalar_['lensed_scalar'][]
    # data2 = power_spectra_operation(r=r_data, l_max=700, rotation_angle=rotation)
    # data2.get_spectra()
    # data2.spectra.spectra[:, 2] = r_data*pw_r1.spectra.spectra[:, 2] + \
    #     lensed_scalar.spectra.spectra[:, 2]
    #
    # data2.get_noise(noise_model)
    # data2.spectra_rotation()
    # data2.get_instrument_spectra()
    # data_spectra = data2.instru_spectra.spectra
    # data_matrix = power_spectra_obj(np.array(
    #     [[data_spectra[:, 2]]]).T, model.instru_spectra.ell)
    # data_matrix = power_spectra_obj(
    #     np.array([r_data*pw_r1.spectra.spectra[:, 2] + lensed_scalar.spectra.spectra[:, 2]
    #               ]).T,
    #     pw_r1.spectra.ell)

    # model_r07 = power_spectra_operation(r=0.07, l_max=700, rotation_angle=rotation)
    # model_r07.get_spectra()
    # model_r07.get_noise(noise_model)
    # model_r07.spectra_rotation()
    # model_r07.get_instrument_spectra()

    start = time.time()
    model = power_spectra_operation(l_max=700, rotation_angle=rotation)
    model.get_spectra(r=r_data)
    model.get_noise(noise_model)
    for r in r_grid:
        model.get_spectra(r=r)

        # modelBB = r * pw_r1.spectra.spectra[:, 2] + \
        #     lensed_scalar.spectra.spectra[:, 2]
        # model.spectra.spectra[:, 2] = modelBB

        # likelihood_values_ar_2D.append([])
        for angle in angle_grid:
            # model_r07.spectra_rotation(angle)
            # model_r07.get_instrument_spectra()

            model.spectra_rotation(angle)
            model.get_instrument_spectra()
            # model.spectra.spectra[:, 1] = model_r07.spectra.spectra[:, 1]

            model_spectra = model.instru_spectra.spectra
            model_matrix = power_spectra_obj(np.array(
                [[model_spectra[:, 1], model_spectra[:, 4]],
                 [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
            # model_matrix = power_spectra_obj(np.array(
            #     [[model_r07.instru_spectra.spectra[:, 1], model_spectra[:, 4]],
            #      [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)

            # model_matrix = power_spectra_obj(np.array(
            #     [[model_spectra[:, 2]]]).T, model.instru_spectra.ell)

            likelihood_value = likelihood_pws(model_matrix, data_matrix, 0.1)
            likelihood_values_ar.append(likelihood_value)
            # likelihood_values_ar_2D[-1].append(likelihood_value)

    print('time gridding r =', time.time() - start)

    angle_array, r_array = np.meshgrid(angle_grid, r_grid, indexing='xy')
    likelihood_mesh = np.reshape(likelihood_values_ar, (-1, nstep_angle))
    # plt.plot(r_grid, likelihood_mesh)

    """==================== PLOT LIKELIHOOD ON R AND ALPHA ===================="""

    # HEEERE FOR LIKELIHOOD
    likelihood_norm = likelihood_mesh[:, idx1] - min(likelihood_mesh[:, idx1])
    plt.plot(r_grid, np.exp(-likelihood_norm), label='likelihood-min(likelihood)')

    fisher_gaussian_r = np.exp(-((r_grid - r_data)**2) / (2*sigma_sq_matrix[1, 1]))
    # fisher_gaussian_norm_r = fisher_gaussian_r - min(fisher_gaussian_r)
    plt.plot(r_grid, fisher_gaussian_r, label='fisher gaussian')

    # fisher_gaussian_rbb = np.exp(-((r_grid - r_data)**2) / (2/fishrrBB))
    # fisher_gaussian_norm_r = fisher_gaussian_r - min(fisher_gaussian_r)
    # plt.plot(r_grid, fisher_gaussian_rbb, label='fisher gaussian BB')

    plt.xlabel('r')
    plt.ylabel('likelihood')
    plt.legend()
    plt.title('likelihood on r with true r={}'.format(r_data))
    plt.show()
    print('gradient fisher = ', np.gradient(np.gradient(fisher_gaussian_r))[idx2])
    print('gradient likelihood = ', np.gradient(np.gradient(np.exp(-likelihood_norm)))[idx2])
    IPython.embed()
    # plt.contour(r_array, angle_array, likelihood_mesh)

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    mu = np.array([rotation.value, r_data])
    pos = np.empty(angle_array.shape + (2,))
    pos[:, :, 0] = angle_array
    pos[:, :, 1] = r_array

    fisher_2D_gaussian = multivariate_gaussian(pos, mu, sigma_sq_matrix)

    # plt.contourf(angle_array, r_array, np.exp(-(likelihood_mesh-np.min(likelihood_mesh))))
    # plt.colorbar()
    # plt.contour(angle_array, r_array, fisher_2D_gaussian, colors='r', linestyles='--')
    #
    # plt.ylabel(r'$r$')
    # plt.xlabel(r'miscalibration angle $\alpha$ in radian')
    # plt.title(r'Joint likelihood for estimation of miscalibration angle $\alpha$ and $r$. True value are $r=${}, $\alpha={}$'.format(
    #     r_data, rotation))
    # # IPython.embed()
    # plt.show()

    # plt.pcolormesh(angle_array.value, r_array,
    #                np.exp(-(likelihood_mesh-np.min(likelihood_mesh))), vmin=0, vmax=1)
    #
    # plt.colorbar()
    # plt.clim(0,1)
    # plt.contour(angle_array, r_array, fisher_2D_gaussian, colors='r',linestyles = '--')

    # plt.ylabel(r'$r$')
    # plt.xlabel(r'miscalibration angle $\alpha$ in radian')
    # plt.title(r'Joint likelihood for estimation of miscalibration angle $\alpha$ and $r$. True value are $r=${}, $\alpha={}$'.format(
    #     r_data, rotation))
    # # IPython.embed()
    # plt.show()
    plt.rc('font', size=22)
    fig, ax = plt.subplots()
    levels = np.arange(0, 1+1/8, 1/8)
    cs = ax.contourf(angle_array, r_array,
                     np.exp(-(likelihood_mesh-np.min(likelihood_mesh))), levels=levels)
    cs2 = ax.contour(angle_array, r_array, fisher_2D_gaussian /
                     np.max(fisher_2D_gaussian), levels=cs.levels, colors='r', linestyles='--')
    cbar = fig.colorbar(cs)
    cbar.add_lines(cs2)
    cbar.ax.set_xlabel(r'$\mathcal{L}$')
    plt.ylabel(r'$r$')
    plt.xlabel(r'miscalibration angle $\alpha$ in radian')
    plt.title(r'Joint likelihood on $r$ and $\alpha$ with $r_{input} =$'+'{},'.format(r_data)+r' $\alpha_{input}=$'+'{}'.format(
        rotation))
    h1, _ = cs2.legend_elements()
    ax.legend([h1[0]], ["Fisher prediction"])

    plt.show()

    # def likelihood_for_hessian_a(angle, model, data_matrix):
    #     angle = angle * u.rad
    #     # r = param_array[1]
    #
    #     # model.get_spectra(r=r)
    #     model.spectra_rotation(angle)
    #     model.get_instrument_spectra()
    #     model_spectra = model.instru_spectra.spectra
    #     model_matrix = power_spectra_obj(np.array(
    #         [[model_spectra[:, 1], model_spectra[:, 4]],
    #          [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
    #     likelihood_value = likelihood_pws(
    #         model_matrix, data_matrix, 0.1)
    #     return likelihood_value

    def likelihood_for_hessian_r(r, model, data_matrix):
        # angle = angle * u.rad

        model.get_spectra(r=r)
        model.spectra_rotation()
        model.get_instrument_spectra()
        model_spectra = model.instru_spectra.spectra
        model_matrix = power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
        likelihood_value = likelihood_pws(
            model_matrix, data_matrix, 0.1)
        return likelihood_value

    def likelihood_for_hessian(param_array, model, data_matrix):
        angle = param_array[0] * u.rad
        r = param_array[1]

        model.get_spectra(r=r)
        model.spectra_rotation(angle)
        model.get_instrument_spectra()
        model_spectra = model.instru_spectra.spectra
        model_matrix = power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
        likelihood_value = likelihood_pws(
            model_matrix, data_matrix, 0.1)
        return likelihood_value

    IPython.embed()

    """=================================purgatory=============================="""
    # test.get_frequencies()
    # test.get_faraday_angles()
    # # test.split_spectra()
    # test.get_faraday_spectra()
    # freq1 = test.faraday_angles
    # cl_farad_norm1 = lib.get_normalised_cl(test.cl_faraday)
    # dico = {'faraday': cl_farad_norm1}
    #
    # test.spectra_rotation(min(test.faraday_angles)*u.rad)
    # test.cl_rot.normalisation = 1
    # cl_min = test.cl_rot.spectra
    #
    # test.spectra_rotation(max(test.faraday_angles)*u.rad)
    # test.cl_rot.normalisation = 1
    # cl_max = test.cl_rot.spectra
    #
    # dico['min'] = cl_min
    # dico['max'] = cl_max
    #
    #
    # dico['normal'] = lib.get_normalised_cl(test.spectra.spectra)

    # hessian = nd.Hessian(likelihood_for_hessian)
    # h = hessian([rotation, r_data], model, data_matrix)[0, 0]
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
