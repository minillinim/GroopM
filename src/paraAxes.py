#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    paraAxes.py                                                              #
#                                                                             #
#    GroopM - managing parallel axes (+ clustering)                           #
#                                                                             #
#    Copyright (C) Michael Imelfort                                           #
#                                                                             #
###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################


__author__ = "Michael Imelfort"
__copyright__ = "Copyright 2012-2020"
__credits__ = ["Michael Imelfort"]
__license__ = "GPL3"
__maintainer__ = "Michael Imelfort"
__email__ = "michael.imelfort@gmail.com"

###############################################################################

import pandas as pd
import numpy as np
np.seterr(all='raise')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD

###############################################################################

class ParaAxes(object):

    def __init__(self, profile_manager, resolution=1000):
        self.axes = None
        self.pm = profile_manager
        self.resolution = resolution
        self.step = 1. / self.resolution

    def make_axes(self, timer, cut_off):
        self.pm.loadData(
            timer,
            "(length >= %d) " % cut_off,
            loadRawKmers=False,
            makeColors=False,
            loadContigNames=False,
            loadContigGCs=False,
            verbose=False,
            silent=True)

        num_svds = np.shape(self.pm.transformedCP)[1]

        self.norm_cov = np.linalg.norm(self.pm.transformedCP, axis=1)

        self.axes = pd.DataFrame(
            self.pm.transformedCP,
            columns=['SVD_%s' % i for i in range(num_svds)]).apply(np.sqrt)

        self.axes['kmer_1'] = self.pm.kmerSVDs[:,0]
        self.axes['kmer_2'] = self.pm.kmerSVDs[:,1]

        min_max_scaler = preprocessing.MinMaxScaler()
        self.axes = pd.DataFrame(
            min_max_scaler.fit_transform(self.axes),
            columns=self.axes.columns)

        self.plot_lines()

    def to_density_index(self, value):
        return int(value/self.step)

    def make_density(self,
        lower_coverage_bound=10,
        plot=False):
        self.density = np.zeros((self.resolution, len(self.axes.columns)))

        step_between = 200
        back_mult = np.array(list(range(step_between))[::-1])/(step_between - 1.)
        forward_mult = np.array(list(range(step_between)))/(step_between - 1.)

        self.density_graph = np.zeros(
            (self.resolution, step_between * (len(self.axes.columns) - 1)))

        self.make_linear_kernels()

        for ridx, row in self.axes.iterrows():
            if self.norm_cov[ridx] < lower_coverage_bound: continue

            last_val = 0

            for idx, value in enumerate(row):
                self.density[:, idx] += self.get_kernel(value)

                if idx > 0:
                    start_step_idx = step_between * (idx-1)
                    for mult_idx, dg_idx in enumerate(range(start_step_idx, start_step_idx + step_between)):
                        self.density_graph[:,dg_idx] += self.get_kernel(value * forward_mult[mult_idx] + last_val * back_mult[mult_idx])

                last_val = value

        plt.imshow(
            np.sqrt(self.density_graph),
            interpolation='nearest',
            extent=(-0.5, 4000-0.5, -0.5, 2000-0.5))
        plt.show()

    def make_linear_kernels(self, span_ratio=.075):
        '''span_ratio determines what percentage of windows are covered by the kernel'''

        span = int(self.resolution * span_ratio)
        if span % 2 == 0:
            span -= 1
        assert(span > 2)
        half_span = int((span - 1) / 2)
        step = 1. / (half_span + 1)

        self.kernel_stamp = np.pad(
            np.pad(
                [1.],
                (half_span, half_span),
                'linear_ramp',
                end_values=(step, step)),
            (self.resolution-half_span-1, self.resolution-half_span-1),
            'constant',
            constant_values=(0.))

    def get_kernel(self, value):
        ki = max(0, self.resolution - int((1. - value) / self.step) -1)
        return self.kernel_stamp[ki:ki+self.resolution]

    def plot_lines(self, lower_coverage_bound=10):
        if self.axes is None: return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        Xs = range(len(self.axes.columns))
        for idx, row in self.axes.iterrows():
            if self.norm_cov[idx] < lower_coverage_bound: continue
            ax.plot(Xs, row, 'k')
        plt.show()
        plt.close(fig)
        del fig
