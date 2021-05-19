# from mpl_toolkits import mplot3d
# from matplotlib import cm
import matplotlib.pyplot as plt
import time
# from pandas import DataFrame as df
import IPython
from astropy import units as u
import numpy as np

import bjlib.lib_project as lib
from scipy.optimize import minimize

import healpy as hp
from fgbuster import MixingMatrix
from fgbuster.component_model import CMB, Dust, Synchrotron
from scipy.linalg import block_diag
from fgbuster.observation_helpers import get_sky
import pysm
import bjlib.V3calc as V3
# from fgbuster import visualization as visu


class sky_map:

    cmb_freq = 200
    dust_freq = 150
    synchrotron_freq = 20

    def __init__(self, nside=128, instrument='SAT', sky_model='c1s0d0',
                 bir_angle=0.0*u.rad,
                 frequencies_by_instrument=[2, 2, 2],
                 miscal_angles=[0.2, 0.0, 0.0]*u.rad):
        # [0.00575959, -0.00575959,  0.00287979]*u.rad):
        # [0.33, -0.33, 0.33/2] * u.deg):

        self._nside = nside
        self._instrument = instrument
        self._sky_model = sky_model
        self._bir_angle = bir_angle
        # self._cal_angle = cal_angle
        self.frequencies_by_instrument = frequencies_by_instrument
        self.miscal_angles = miscal_angles

    def _get_nside(self):
        return self._nside

    def _get_instrument(self):
        return self._instrument

    def _get_sky_model(self):
        return self._sky_model

    def _get_bir_angle(self):
        return self._bir_angle

    def _set_bir_angle(self, new_angle):
        # print('WARNING: birefringence angle changed, other attribute might',
        #       ' need updating')
        if type(new_angle) == u.quantity.Quantity:
            new_angle_rad = new_angle.to(u.rad)
        else:
            # print('no unit has be given for new angle, radian is assumed')
            new_angle_rad = new_angle * u.rad
        self._bir_angle = new_angle_rad

    nside = property(_get_nside)
    instrument = property(_get_instrument)
    sky_model = property(_get_sky_model)
    bir_angle = property(_get_bir_angle, _set_bir_angle)
    # cal_angle = property(_get_cal_angle, _set_cal_angle)

    def get_pysm_sky(self):
        sky = pysm.Sky(get_sky(self.nside, self.sky_model))
        self.sky = sky

    def get_frequency(self):
        if self.instrument == 'SAT':
            print(self.instrument)
            self.frequencies = V3.so_V3_SA_bands()
        if self.instrument == 'LAT':
            print(self.instrument)
            self.frequencies = V3.so_V3_LA_bands()
        if self.instrument == 'LiteBIRD':
            print(self.instrument)
            self.frequencies = np.array([40.0, 50.0, 60.0, 68.4, 78.0, 88.5,
                                         100.0, 118.9, 140.0, 166.0, 195.0,
                                         234.9, 280.0, 337.4, 402.1])

    def get_freq_maps(self, output=0):
        cmb_freq_maps = self.sky.cmb(sky_map.cmb_freq) * \
            pysm.convert_units('K_RJ', 'K_CMB', sky_map.cmb_freq)
        dust_freq_maps = self.sky.dust(sky_map.dust_freq) * \
            pysm.convert_units('K_RJ', 'K_CMB', sky_map.dust_freq)
        sync_freq_maps = self.sky.synchrotron(sky_map.synchrotron_freq) *\
            pysm.convert_units('K_RJ', 'K_CMB', sky_map.synchrotron_freq)
        self.cmb_freq_maps = cmb_freq_maps
        self.dust_freq_maps = dust_freq_maps
        self.sync_freq_maps = sync_freq_maps
        if output:
            return cmb_freq_maps, dust_freq_maps, sync_freq_maps

    def cmb_rotation(self, output=0):
        cmb_freq_rot = lib.map_rotation(self.cmb_freq_maps, self.bir_angle)
        self.cmb_freq_rot = cmb_freq_rot
        # Updating birefringence matrix
        self.get_bir_matrix()
        # rotation_block = np.array(
        #     [[np.cos(2*self.bir_angle),  np.sin(2*self.bir_angle)],
        #      [-np.sin(2*self.bir_angle), np.cos(2*self.bir_angle)]
        #      ])
        # bir_matrix = block_diag(rotation_block, np.eye(4))
        # self.bir_matrix = bir_matrix
        if output:
            return cmb_freq_rot

    def get_bir_matrix(self):
        rotation_block = np.array(
            [[np.cos(2*self.bir_angle),  np.sin(2*self.bir_angle)],
             [-np.sin(2*self.bir_angle), np.cos(2*self.bir_angle)]
             ])
        bir_matrix = block_diag(rotation_block, np.eye(4))
        self.bir_matrix = bir_matrix

    def faraday_rotation(self, output=0):

        cmb_map_array = np.array([self.cmb_freq_maps/len(self.frequencies) for
                                  i in range(len(self.frequencies))])
        try:
            cmb_faraday = np.sum(lib.map_rotation(cmb_map_array,
                                                  self.bir_angle), 0)
            self.cmb_faraday = cmb_faraday
            if output:
                return cmb_faraday
        except TypeError:
            print('In faraday rotation, only one angle given. \n \
                   Faraday rotation impossible, try self.cmb_rotation()')

    def get_signal(self, output=0):
        if self.instrument == 'LAT':
            start_spectra = 0
        else:
            start_spectra = 1

        if hasattr(self, 'cmb_freq_rot'):
            signal_ = np.append(self.cmb_freq_rot[start_spectra:],
                                self.dust_freq_maps[start_spectra:], 0)
            signal = np.append(signal_, self.sync_freq_maps[start_spectra:], 0)
            del signal_

            print('Signal with cmb rotation')

        elif hasattr(self, 'cmb_faraday'):
            signal_ = np.append(self.cmb_faraday[start_spectra:],
                                self.dust_freq_maps[start_spectra:], 0)
            signal = np.append(signal_, self.sync_freq_maps[start_spectra:], 0)
            del signal_
            print('Signal with Faraday rotation')

        else:
            signal_ = np.append(self.cmb_freq_maps[start_spectra:],
                                self.dust_freq_maps[start_spectra:], 0)
            signal = np.append(signal_, self.sync_freq_maps[start_spectra:], 0)
            print('Signal with no rotation')

        self.signal = signal
        if output:
            return signal

    def get_A_ev(self, fix_temp=False):
        self.fix_temp = fix_temp
        if not self.fix_temp:
            components = [CMB(), Dust(sky_map.dust_freq),
                          Synchrotron(sky_map.synchrotron_freq)]
        else:
            print('fix temp in get_A_ev')
            components = [CMB(), Dust(sky_map.dust_freq, temp=20),
                          Synchrotron(sky_map.synchrotron_freq)]
        A = MixingMatrix(*components)
        self.A = A
        A_ev = A.evaluator(self.frequencies)
        self.A_ev = A_ev

    def evaluate_mixing_matrix(self, spectral_indices=[1.59, 20, -3]):
        if not self.fix_temp:
            self.spectral_indices = spectral_indices
        else:
            print('fix temp in evaluate_mixing_matrix')
            self.spectral_indices = spectral_indices[::2]
        A_ = self.A_ev(self.spectral_indices)
        self.A_ = A_
        if self.instrument == 'SAT':
            mixing_matrix = np.repeat(A_, 2, 0)
        else:
            print('Only SAT supported for mxing matrix for now')

        mixing_matrix = np.repeat(mixing_matrix, 2, 1)

        for i in range(np.shape(mixing_matrix)[0]):
            for j in range(np.shape(mixing_matrix)[1]):
                mixing_matrix[i, j] = mixing_matrix[i, j] *\
                    (((i % 2)-(1-j % 2)) % 2)

        self.mixing_matrix = mixing_matrix

    def get_mixing_matrix(self):
        self.get_A_ev()
        self.evaluate_mixing_matrix()
        '''
        components = [CMB(), Dust(sky_map.dust_freq),
                      Synchrotron(sky_map.synchrotron_freq)]
        A = MixingMatrix(*components)
        A_ev = A.evaluator(self.frequencies)
        res = [1.59, 20, -3]
        A_ = A_ev(res)
        self.A_ = A_

        if self.instrument == 'SAT':
            mixing_matrix = np.repeat(A_, 2, 0)
        if self.instrument == 'LAT':
            print('Error: LAT in mixing matrix')
            # TODO: get LAT mixing matrix, because of T the second repetetion
            # should be changed ...
            # mixing_matrix = np.repeat(A_, 3, 0)
            # print(' WARNING: LAT has same T mixing matrix as Q & U')

        mixing_matrix = np.repeat(mixing_matrix, 2, 1)

        for i in range(np.shape(mixing_matrix)[0]):
            for j in range(np.shape(mixing_matrix)[1]):
                mixing_matrix[i, j] = mixing_matrix[i, j] *\
                    (((i % 2)-(1-j % 2)) % 2)

        self.mixing_matrix = mixing_matrix
        '''

    def get_miscalibration_angle_matrix(self):
        miscal_matrix = 1

        miscal_angles = self.miscal_angles
        frequencies_by_instrument = self.frequencies_by_instrument
        # TODO:  freq by instrument should be SAT or LAT property and class
        # attribute
        try:
            if len(miscal_angles) != len(frequencies_by_instrument) or \
                    sum(frequencies_by_instrument) != len(self.frequencies):
                print('WARNING: miscalibration angles doesnt match the number',
                      'of instrument\n ',
                      'or the number of frequencies by instrument doesnt match'
                      ' the number of frequencies')

            else:
                miscal_matrix = 1
                instrument_nb = 0
                for angle in miscal_angles:

                    rotation_block = np.array(
                        [[np.cos(2*angle),  np.sin(2*angle)],
                         [-np.sin(2*angle), np.cos(2*angle)]
                         ])

                    # print(rotation_block)
                    if type(miscal_matrix) == int:
                        miscal_matrix = rotation_block
                        for i in range(
                                frequencies_by_instrument[instrument_nb]-1):
                            miscal_matrix = block_diag(miscal_matrix,
                                                       rotation_block)
                    else:
                        for i in range(
                                frequencies_by_instrument[instrument_nb]):
                            miscal_matrix = block_diag(miscal_matrix,
                                                       rotation_block)
                    instrument_nb += 1

                self.miscal_matrix = miscal_matrix

        except AttributeError:
            print('No instrument frequencies !')

    def get_data(self):
        if hasattr(self, 'miscal_matrix'):
            A_s = np.dot(self.mixing_matrix, self.signal)
            M_A_s = np.dot(self.miscal_matrix, A_s)
            self.data = M_A_s
        else:
            A_s = np.dot(self.mixing_matrix, self.signal)
            self.data = A_s

    def from_pysm2data(self):
        self.get_pysm_sky()
        self.get_frequency()
        self.get_freq_maps()
        self.cmb_rotation()
        self.get_signal()
        self.get_mixing_matrix()
        self.get_miscalibration_angle_matrix()
        self.get_data()
        return self.data

    def data2alm(self):
        # TODO: DOUBLE CHECK THIS mix between data and signal, signal is probably the right one but must be checked
        ones = np.ones(np.shape(self.signal[-1]))
        TQU_cmb = [ones, self.signal[0], self.signal[1]]
        TQU_dust = [ones, self.signal[2], self.signal[3]]
        TQU_synch = [ones, self.signal[4], self.signal[5]]
        alm_cmb = hp.map2alm(TQU_cmb)
        alm_dust = hp.map2alm(TQU_dust)
        alm_synch = hp.map2alm(TQU_synch)
        alm_data = np.array([alm_cmb[1], alm_cmb[2],
                             alm_dust[1], alm_dust[2],
                             alm_synch[1], alm_synch[2]])

        self.alm_data = alm_data

    def get_primordial_spectra(self):
        pars, results, powers = lib.get_basics(l_max=self.nside*4, raw_cl=True,
                                               lens_potential=False, ratio=0)
        self.prim = powers['total'][:self.nside*3]

    def get_noise(self):
        # TODO: put SAT and LAT caracteristics in class atributes
        if self.instrument == 'SAT':
            print('SAT white noise')
            # noise_covariance = np.eye(12)
            # inv_noise = np.eye(12)
            white_noise = np.repeat(V3.so_V3_SA_noise(2, 2, 1, 0.1,
                                                      self.nside*3)[2],
                                    2, 0)
            # self.white_noise = white_noise
            #
            noise_covariance = np.diag(
                (white_noise / hp.nside2resol(self.nside, arcmin=True))**2)
            inv_noise = np.diag(1 / ((white_noise / hp.nside2resol(
                self.nside, arcmin=True))**2))

            noise_N_ell = np.repeat(
                V3.so_V3_SA_noise(2, 2, 1, 0.1, self.nside*3)[1],
                2, 0)
            ells = np.shape(noise_N_ell)[-1]
            noise_cov_ell = [np.diag(noise_N_ell[:, k]) for k in range(ells)]
            inv_noise_cov_ell = [np.diag(1/noise_N_ell[:, k])
                                 for k in range(ells)]

        if self.instrument == 'LAT':
            white_noise = np.repeat(V3.so_V3_LA_noise(0, 0.4, 5000)[2], 2, 0)
            noise_covariance = np.diag(
                (white_noise / hp.nside2resol(self.nside, arcmin=True))**2)
            inv_noise = np.diag(1 / (white_noise / hp.nside2resol(
                self.nside, arcmin=True))**2)

        if self.instrument == 'LiteBIRD':
            sensitivity = np.array([37.5, 24.0, 19.9, 16.2, 13.5, 11.7, 9.2,
                                    7.6, 5.9, 6.5, 5.8, 7.7, 13.2, 19.5, 37.5])

        self.noise_covariance = noise_covariance
        self.inv_noise = inv_noise
        self.noise_cov_ell = noise_cov_ell
        self.inv_noise_ell = inv_noise_cov_ell

    def prim_rotation(self):
        cl_rot = lib.cl_rotation(self.prim, self.bir_angle)
        self.cl_rot = cl_rot

    def get_bir_prior(self):
        # cl_cmb = hp.alm2cl(self.alm_data[:2])
        cl_cmb = self.cl_rot
        ells = np.shape(cl_cmb)[0]
        self.ell_max = ells
        cmb_block = np.array(
            [[cl_cmb[:, 1], cl_cmb[:, 4]],
             [cl_cmb[:, 4], cl_cmb[:, 2]]
             ])
        # print('shape cmb_block', np.shape(cmb_block))
        cmb_block_inv = [(1/(cl_cmb[k, 2]*cl_cmb[k, 1] - cl_cmb[k, 4]*cl_cmb[k, 4]))
                         * np.array(
            [[cl_cmb[k, 2], -cl_cmb[k, 4]],
             [-cl_cmb[k, 4], cl_cmb[k, 1]]
             ]) for k in range(ells)]
        # print('shape cmb_block_inv', np.shape(cmb_block_inv))

        prior_cov = np.array(
            [block_diag(cmb_block[:, :, k], np.zeros([4, 4]))
             for k in range(ells)])
        prior_cov_inv = np.array(
            [block_diag(cmb_block_inv[k], np.zeros([4, 4]))
             for k in range(ells)])

        self.prior_cov = prior_cov
        self.prior_cov_inv = prior_cov_inv

        # self.prior_cov_inv = np.array([np.zeros([6, 6]) for k in range(ells)])
        # self.prior_cov = np.array([np.zeros([6, 6]) for k in range(ells)])
    # def get_noise_covariance

    def get_projection_op(self, prior=False):
        # TODO: put SAT and LAT caracteristics in class atributes

        mix_effectiv = np.dot(self.miscal_matrix,
                              np.dot(self.mixing_matrix, self.bir_matrix))
        mix_T = mix_effectiv.T

        if prior:
            print('Prior option in get_projection_op() no longer supported')
            # ATNA = np.einsum('ij,kjl,lm->kim', mix_T, self.inv_noise_ell,
            #                  mix_effectiv)
            # inverse = np.linalg.inv(ATNA + self.prior_cov_inv[2:])
            # self.inverse = inverse
            # # inverse = np.array([np.zeros(6, 6), np.zeros(6, 6)].append(inverse_))
            #
            # ATN = np.einsum('ij,kjm->kim', mix_T, self.inv_noise_ell)
            # self.ATN = ATN
            # # print('SHAPE ATN = ', np.shape(ATN))
            # IATN = np.einsum('kij,kjl->kil', inverse, ATN)
            # # print('SHAPE IATN = ', np.shape(IATN))
            # NA = np.dot(self.inv_noise_ell, mix_effectiv)
            # # print('SHAPE NA = ', np.shape(NA))
            #
            # proj_nonoise = np.einsum('kij,kjl->kil', NA, IATN)
            # # print('SHAPE proj_nonoise = ', np.shape(proj_nonoise))
            #
            # projection = self.inv_noise_ell - proj_nonoise
            # # print('SHAPE projection = ', np.shape(projection))
            #
            # # np.dot(inverse, ATN)

        else:
            inverse = np.linalg.inv(
                np.dot(np.dot(mix_T, self.inv_noise), mix_effectiv)
            )
            projection = - np.dot(
                np.dot(self.inv_noise, mix_effectiv),
                np.dot(
                    inverse,
                    np.dot(mix_T, self.inv_noise))
            ) + self.inv_noise

        self.mix_effectiv = mix_effectiv

        self.projection = projection

    def get_mask(self, path='/global/homes/j/jost/BBPipe'):
        if self.instrument == 'SAT':
            # BBPipe_path = '/global/homes/j/jost/BBPipe'
            # BBPipe_path = '/home/baptiste/BBPipe'
            # mask_ = hp.read_map(path +
            #                     "/test_mapbased_param/norm_nHits_SA_35FOV_G_nside512_binary.fits")
            mask_ = hp.read_map(path +
                                "/test_mapbased_param/mask_04000.fits")
            mask = hp.ud_grade(mask_, self.nside)
            del mask_
            self.mask = mask
        else:
            print('Only SAT mask supported for now')


def get_chi_squared(angle_array, data_skm, model_skm, prior=False):

    # print('SHAPE ddt=', np.shape(ddt))
    # model_skm.miscal_angles = [angle_array[0], angle_array[1],
    #                            angle_array[2], angle_array[3], angle_array[4], angle_array[5]]  # , angle_array[1], 0]
    # model_skm.frequencies_by_instrument = [6, 0, 0]
    model_skm.miscal_angles = [1, 2, angle_array[0]]

    model_skm.bir_angle = 0  # angle_array[1]

    model_skm.get_miscalibration_angle_matrix()
    model_skm.cmb_rotation()

    if prior:
        # model_skm.from_pysm2data()

        # model_skm.get_signal()
        # model_skm.get_data()
        # model_skm.data2alm()
        model_skm.prim_rotation()
        model_skm.get_bir_prior()
        model_skm.get_projection_op(prior=prior)

        d = np.dot(data_skm.mix_effectiv, data_skm.alm_data)
        ddt = np.absolute(np.einsum('ik...,...kj->ijk', d, np.matrix.getH(d)))
        # print(np.shape(ddt))
        first_term = model_skm.inv_noise_ell - model_skm.projection
        # test = [[data.noise_cov_ell[i]]*(i+3)
        #         for i in range(np.shape(data.noise_cov_ell)[0])]

        noise_elltot_ = np.append([np.zeros([12, 12])],
                                  model_skm.noise_cov_ell, 0)
        noise_elltot = np.append([np.zeros([12, 12])], noise_elltot_, 0)

        ell = hp.Alm.getlm(np.shape(noise_elltot)[0]-1)[0]
        noise_cov_lm = [noise_elltot[i] for i in ell]
        # noise_cov_lm = np.concatenate([[noise_elltot[i]]*(i+1) for i in ell])

        first_term_tot_ = np.append([np.zeros([12, 12])], first_term, 0)
        first_term_tot = np.append([np.zeros([12, 12])], first_term_tot_, 0)

        ell = hp.Alm.getlm(np.shape(first_term_tot)[0]-1)[0]
        first_term_lm = [first_term_tot[i] for i in ell]
        # first_term_lm = np.concatenate([[first_term_tot[i]]*(i+1) for i in ell])

        # print('shape test = ', np.shape(test))
        # print('shape test1 = ', np.shape(test1))

        # print('SHAPE first_term', np.shape(first_term))
        # print('SHAPE first_term_lm', np.shape(first_term_lm))

        second_term_lm = np.array([noise_cov_lm[i] + ddt[:, :, i]
                                   for i in range(len(ddt[0, 0, :]))]).T
        # print('SHAPE second_term', np.shape(second_term_lm))

        in_sum = np.einsum('l...ij,jk...l->ikl',
                           first_term_lm, second_term_lm)
        # in_sum = -np.absolute(in_sum_)
        # in_sum = in_sum_
    else:
        # d = np.dot(data_skm.mix_effectiv, data_skm.signal)
        d = data_skm.data

        ddt = np.einsum('ik...,...kj->ijk', d, d.T)
    # model_skm.get_mixing_matrix()
        model_skm.get_projection_op()
    # print(df(model_skm.projection))
        #
        # first_term = data_skm.inv_noise - model_skm.projection
        # second_term = np.array([data_skm.noise_covariance + ddt[:, :, i]
        #                         for i in range(len(ddt[0, 0, :]))]).T

        first_term = model_skm.inv_noise - model_skm.projection
        second_term = np.array([model_skm.noise_covariance + ddt[:, :, i]
                                for i in range(len(ddt[0, 0, :]))]).T

        in_sum = np.einsum('ij,jk...l->ikl', first_term, second_term)

    sum = np.sum(in_sum, -1)

    chi_squared = - np.trace(sum)

    return chi_squared


def main():

    data = sky_map()
    model = sky_map()

    data.from_pysm2data()
    model.from_pysm2data()

    data.get_noise()
    model.get_noise()
    # IPython.embed()
    data.get_projection_op()
    model.get_projection_op()
    # IPython.embed()

    data.data2alm()
    model.data2alm()

    data.get_primordial_spectra()
    model.get_primordial_spectra()

    start = time.time()
    grid = np.arange(-1*np.pi, 1*np.pi, 2*np.pi/100)
    # for i in grid:
    #     get_chi_squared([i], data, model)

    # print('time chi2 in s = ', time.time() - start)
    # IPython.embed()

    start = time.time()
    prior_ = False
    results = minimize(get_chi_squared, [0, 0, 0], (data, model, prior_),
                       bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])
    print('time minimize in s = ', time.time() - start)

    IPython.embed()
    # print('results = ', results.x)
    # IPython.embed()
    # print('hessian = ', results.hess_inv)

    # visu.corner_norm(results.x, results.hess_inv)
    plt.show()
    # bir_grid, misc_grid = np.meshgrid(grid, grid,
    # indexing='ij')
    start = time.time()
    get_chi_squared([0, 0, 0], data, model, prior=True)
    print('time chi2 prior = ', time.time() - start)
    # slice_chi2 = np.array([[get_chi_squared([i, j, 0, 0], data, model) for i in grid]
    #                        for j in grid])
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(grid, misc_grid, slice_chi2, cmap=cm.viridis)
    # plt.show()

    plt.plot(grid, [-get_chi_squared([i], data, model) for i in grid])
    print('time grid in s = ', time.time() - start)

    # plt.yscale('log')
    plt.show()

    # IPython.embed()
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
