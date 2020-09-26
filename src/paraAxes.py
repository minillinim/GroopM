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

import logging
L = logging.getLogger('groopm')

from .densityTools import DensityKernel, DensityStore, DensityGraph
from . import binManager

###############################################################################

class TmpBin(object):

    def __init__(self, num_columns):
        self.lower_bounds = [0. for _ in range(num_columns)]
        self.upper_bounds = [0. for _ in range(num_columns)]

    def add_bound(self, column, lower, upper):
        self.lower_bounds[column] = lower
        self.upper_bounds[column] = upper

    def init_idxs(self, idxs):
        self.idxs = list(idxs)

    def add_idx(self, idx, row):
        for column, value in enumerate(row):
            if value < self.lower_bounds[column] or value > self.upper_bounds[column]:
                return False
        self.idxs.append(idx)
        return True

class ParaAxes(object):

    def __init__(self, profile_manager, resolution=1000):
        self.pm = profile_manager
        self.resolution = resolution
        self.kernel = DensityKernel(self.resolution)
        self.axes = None
        self.axes_sorter = []
        self.num_contigs = len(self.pm.normCoverages)
        self.num_columns = 6
        self.ranked_idxs = np.argsort(self.pm.normCoverages)[::-1]
        self.tmp_bins = []

    def L(self, indent_level, message):
        L.info('%s%s' % (
            ''.join([' ']*(indent_level*2)),
            message))

    def make_axes(self, force=False):
        self.L(0, "Making parallel axes")

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

    def density_cluster(self,
        bin_manager,
        save_bins=True,
        window=2000,
        tolerance=0.025,
        plot_bins=False,
        plot_journey=False,
        limit=0):

        self.make_axes()

        # BM = binManager.BinManager(pm=self.pm)
        # BM.setColorMap('HSV')
        # all_bin = BM.makeNewBin(rowIndices=self.ranked_idxs)
        # BM.plotBins(bids=[all_bin.id], axes=self.axes.values, FNPrefix='ALLBIN')
        # return

        rows_assigned = 0

        density_store = DensityStore(
            self.num_columns,
            self.kernel,
            plot_journey=plot_journey)

        self.L(0, 'Start PAX density clustering')
        left_boundary = window
        used_idxs = set()
        target_idxs = self.ranked_idxs[:window]
        self.L(1, 'Making first density block using a window of %s ranked rows (Total: %s)' %  (
            window,
            self.num_contigs))

        for row in self.axes.values[self.ranked_idxs[:window], :]:
            density_store.add(row)

        count = 0
        if limit == 0:
            from sys import maxsize
            limit = maxsize
        keep_going = True
        while(keep_going):
            count += 1

            self.L(2, 'Pass: %s - %s rows assigned (%0.2f pct)' % (
                count,
                rows_assigned,
                100. * (rows_assigned / self.num_contigs)))

            if count > limit:
                keep_going = False

            if plot_journey:
                fig = plt.figure()
                density_store.plot(fig.add_subplot(self.num_columns + 1, 2, 1))
                current_plot = 2

            _density_store = density_store
            _target_idxs = target_idxs
            used_columns = []
            tmp_bin = TmpBin(self.num_columns)

            for round in range(self.num_columns):
                self.L(3, 'Round: %s - working with: %s rows' % (round, len(_target_idxs)))
                if len(_target_idxs) == 0:
                    keep_going = False
                    break

                # locate the brightest part of the density graph
                (brightest_column, brightest_value, target_value) = _density_store.find_brightest(used_columns)
                used_columns.append(brightest_column)

                lower_bound = np.max([0, target_value-tolerance])
                upper_bound = np.min([1, target_value+tolerance])
                tmp_bin.add_bound(brightest_column, lower_bound, upper_bound)

                self.L(
                    4,
                    'col: %s | val: %s | lb: %s | tgt: %s | ub: %s' % (
                        brightest_column,
                        '%0.3f' % brightest_value,
                        '%0.3f' % lower_bound,
                        '%0.3f' % target_value,
                        '%0.3f' % upper_bound))

                sorted_idxs = np.argsort(self.axes.values[_target_idxs, brightest_column])
                included_idx = np.searchsorted(
                    self.axes.values[_target_idxs, brightest_column],
                    lower_bound,
                    'left',
                    sorted_idxs)

                __target_idxs = []
                _density_store = DensityStore(
                    self.num_columns,
                    self.kernel,
                    plot_journey=plot_journey)

                self.L(4, 'len sorted: %s' % len(sorted_idxs))
                self.L(4, 'Start from: %s' % (included_idx))

                if plot_journey:
                    line_ax = fig.add_subplot(self.num_columns + 1, 2, current_plot)
                    current_plot += 1
                    Xs = range(self.num_columns)

                for idx in _target_idxs[sorted_idxs[included_idx:]]:
                    row = self.axes.values[idx, :]
                    if row[brightest_column] >= upper_bound: break
                    __target_idxs.append(idx)
                    _density_store.add(row)
                    if plot_journey:
                        line_ax.plot(Xs, row, 'k')

                _target_idxs = np.array(__target_idxs)

                self.L(4, 'Added %s | %s rows to ds' % (len(_target_idxs), _density_store.num_rows))

                if plot_journey:
                    _density_store.plot(fig.add_subplot(self.num_columns + 1, 2, current_plot))
                    current_plot += 1

            tmp_bin.init_idxs(_target_idxs)
            self.tmp_bins.append(tmp_bin)

            if plot_journey:
                self.L(3, 'Plotting journey #%s' % count)
                plt.savefig('journey-%s-%s.png' % (count, len(_target_idxs)), dpi=300, format='png')
                plt.close(fig)
                del fig

            self.L(3, 'Update targets')

            if len(_target_idxs) == 0:
                break

            rows_assigned += len(_target_idxs)

            for row in self.axes.values[_target_idxs, :]:
                density_store.remove(row)

            target_idxs = list(set(target_idxs) - set(_target_idxs))

            if left_boundary < self.num_contigs:
                while(len(target_idxs) < window):
                    try:
                        _idx = self.ranked_idxs[left_boundary]
                        left_boundary += 1
                        row = self.axes.values[_idx]
                        add = True
                        # Test if the new row can be added to existing...
                        for tmp_bin in self.tmp_bins:
                            if tmp_bin.add_idx(_idx, row):
                                add = False
                                rows_assigned += 1
                                break
                        if add:
                            target_idxs.append(_idx)
                            density_store.add(row)

                    except IndexError: break

            target_idxs = np.array(target_idxs)

        self.L(2, 'DONE. %s rows of %s assigned (%0.2f pct)' % (
            rows_assigned,
            self.num_contigs,
            100. * (rows_assigned / self.num_contigs)))

        for tmp_bin in self.tmp_bins:
            bin = bin_manager.makeNewBin(rowIndices=np.array(tmp_bin.idxs))
            with open('BINDATA_%s' % (bin.id), 'w') as fh:
                for ridx in tmp_bin.idxs:
                    fh.write('%s\n' % (str(self.axes.values[ridx])))

        if plot_bins:
            bin_manager.plotBins(axes=self.axes.values)

        if save_bins:
            bin_manager.saveBins(nuke=True)

        # import code
        # code.interact(local=locals())

    def plot(self, num_blocks=1, include=1.0):

        self.make_axes()

        self.L(0, 'Start PAX density plotting')
        density_graph = DensityGraph(self.num_columns, self.kernel)
        include_up_to = int(include*self.num_contigs)
        target_idxs = self.ranked_idxs[:include_up_to]

        self.L(0, 'Making density plots using %s percent of the rows (%s)' %  (
            int(include*100),
            include_up_to))

        block_step = int(include_up_to / num_blocks)
        block_boundaries = [i * block_step for i in range(num_blocks)]
        block_boundaries.append(include_up_to)

        for block_idx in range(num_blocks):
            self.plot_single(
                target_idxs[block_boundaries[block_idx]:block_boundaries[block_idx+1]],
                block_idx+1,
                black_line_idxs=target_idxs[:block_boundaries[block_idx]],
                density_graph=density_graph)

    def plot_single(self,
        target_idxs,
        block_id,
        black_line_idxs=[],
        density_graph=None):

        if density_graph is None:
            density_graph = DensityGraph(self.num_columns, self.kernel)

        fig = plt.figure()
        line_ax = fig.add_subplot(2, 1, 2)
        Xs = range(self.num_columns)

        if len(black_line_idxs) > 0:
            self.L(1, 'plotting black lines for block: %s' % (block_id))
            for row in self.axes.values[black_line_idxs]:
                line_ax.plot(Xs, row, 'k')

        self.L(1, 'plotting red + dens for block: %s' % (block_id))
        count = 0
        for idx, row in enumerate(self.axes.values[target_idxs]):
            line_ax.plot(Xs, row, 'r')
            density_graph.add(row)
            count += 1
            if count >= 1000:
                self.L(2, 'PPLOT Processed %s rows' % (idx + 1))
                count = 0

        density_graph.plot_max_line(density_graph.plot(fig.add_subplot(2, 1, 1)))

        plt.axis('off')
        plt.savefig('/tmp/%s.png' % block_id, dpi=300, format='png')
        plt.close(fig)
        del fig

    def plot_lines(self, ax, data=None):
        if data is None:
            if self.axes is None: return
            data = self.axes.values

        Xs = range(self.num_columns)
        for row in data:
            ax.plot(Xs, row, 'k')
