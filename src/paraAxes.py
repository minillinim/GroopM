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
        self.kernel_stamp = None
        self.kernel_type = None
        self.pm = profile_manager
        self.resolution = resolution
        self.step = 1. / self.resolution
        self.norm_coverages = np.linalg.norm(self.pm.transformedCP, axis=1)
        self.num_contigs = len(self.norm_coverages)
        self.num_columns = 6

    def _make_axes(self, force=False):

        if self.axes is not None:
            if not force:
                return

        num_svds = np.shape(self.pm.transformedCP)[1]

        self.axes = pd.DataFrame(
            self.pm.transformedCP,
            columns=['SVD_%s' % i for i in range(num_svds)]).apply(np.sqrt)

        self.axes['kmer_1'] = self.pm.kmerSVDs[:,0]
        self.axes['kmer_2'] = self.pm.kmerSVDs[:,1]

        min_max_scaler = preprocessing.MinMaxScaler()
        self.axes = pd.DataFrame(
            min_max_scaler.fit_transform(self.axes),
            columns=self.axes.columns)

    def init_density_storage(self, kernel_type='linear', plot=False):
        self._make_axes()
        self._make_kernels(kernel_type)
        self.ranked_idxs = self._rank_contigs()
        self.density = np.zeros((self.resolution, len(self.axes.columns)))

    def init_density_grap_storage(self, kernel_type='linear', step_between=200):
        self._make_axes()
        self._make_kernels(kernel_type)
        self.ranked_idxs = self._rank_contigs()
        self.density_graph = np.zeros(
            (self.resolution, step_between * (len(self.axes.columns) - 1)))
        back_mult = np.array(list(range(step_between))[::-1])/(step_between - 1.)
        forward_mult = np.array(list(range(step_between)))/(step_between - 1.)
        self.d_graph_data = [step_between, back_mult, forward_mult]

    def _make_kernels(self, kernel_type, span_ratio=.075):
        '''span_ratio determines what percentage of windows are covered by the kernel'''

        self.kernel_type = kernel_type

        if self.kernel_type == 'linear':

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

    def _rank_contigs(self, force=False):
        rankings = dict(zip(np.argsort(self.norm_coverages), range(len(self.norm_coverages))))
        for rank_idx, pm_idx in enumerate(np.argsort(self.pm.contigLengths)):
            rankings[pm_idx] += rank_idx
        return [k for k, v in sorted(rankings.items(), key=lambda item: item[1])][::-1]

    def density_cluster(self, include=1.0):
        self.init_density_storage()
        for _, row in self.axes[:int(include*self.num_contigs)].iterrows():
            _update_density(row)

    def _update_density(self, row):
        for idx, value in enumerate(row):
            self.density[:, idx] += self._get_kernel(value)

    def _update_density_graph(self, row):
        last_val = 0
        step_between, back_mult, forward_mult = self.d_graph_data
        for idx, value in enumerate(row):
            if idx > 0:
                start_step_idx = step_between * (idx-1)
                for mult_idx, dg_idx in enumerate(
                    range(
                        start_step_idx,
                        start_step_idx + step_between)):

                    self.density_graph[:,dg_idx] += self._get_kernel(
                        value * forward_mult[mult_idx] +
                        last_val * back_mult[mult_idx])

            last_val = value

    def _get_kernel(self, value):
        ki = max(0, self.resolution - int((1. - value) / self.step) -1)
        return self.kernel_stamp[ki:ki+self.resolution]

    def plot(self, num_blocks=20):

        self.init_density_grap_storage()

        block_step = int(self.num_contigs / num_blocks)
        block_boundaries = [i * block_step for i in range(num_blocks)]
        block_boundaries.append(self.num_contigs)
        Xs = range(len(self.axes.columns))
        for block_idx in range(num_blocks):

            fig = plt.figure()
            dens_ax = fig.add_subplot(2, 1, 1)
            line_ax = fig.add_subplot(2, 1, 2)

            for _, row in self.axes[:block_boundaries[block_idx]].iterrows():
                line_ax.plot(Xs, row, 'k')

            line_ax.set_xticklabels(['C1', 'C2', 'C3', 'C4', 'K1', 'K2'])
            line_ax.set_yticklabels([])
            line_ax.set_yticks([])

            for _, row in self.axes[block_boundaries[block_idx]:block_boundaries[block_idx+1]].iterrows():
                line_ax.plot(Xs, row, 'r')
                self._update_density_graph(row)

            dens_ax.imshow(
                np.sqrt(self.density_graph),
                interpolation='nearest',
                extent=(-0.5, (self.num_columns*1000)-0.5, -0.5, 3000-0.5))

            plt.axis('off')
            plt.savefig('/tmp/%s.png' % block_idx, dpi=300, format='png')
            plt.close(fig)
            del fig

    def plot_lines(self, ax, lower_coverage_bound=10):
        if self.axes is None: return

        Xs = range(len(self.axes.columns))
        for idx, row in self.axes.iterrows():
            if self.norm_coverages[idx] < lower_coverage_bound: continue
            ax.plot(Xs, row, 'k')
