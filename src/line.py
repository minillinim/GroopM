#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    line.py                                                                  #
#                                                                             #
#    Sorting ParaAxes rows                                                    #
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

import sys
import numpy as np
np.seterr(all='raise')
import itertools
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import logging
L = logging.getLogger('groopm')

from groopm.rainbow import Rainbow

###############################################################################
###############################################################################
###############################################################################
###############################################################################
# from groopm.line import LineSorter
# means = np.array(means)
# LS = LineSorter()
# LS.sort_sample(means, len(means), plot=True)

class LineConstants(object):
    __instance = None
    def __new__(cls, num_columns):
        if LineConstants.__instance is None:
            LineConstants.__instance = object.__new__(cls)
        LineConstants.__instance.num_columns = num_columns
        LineConstants.__instance.neg_ones = np.array([-1.]*num_columns).reshape((num_columns,1))
        LineConstants.__instance.pos_ones = np.array([1.]*num_columns).reshape((num_columns,1))
        LineConstants.__instance.value_weights = np.array([1. - (0.1 * i) for i in range(num_columns)])
        LineConstants.__instance.shape = (num_columns, 1)
        return LineConstants.__instance

LC = LineConstants(6)

class LineSorter(object):

    def __init__(self): pass

    def sort_sample(self, axes, sample_size, ax=None):
        # sort a set of lines based on smoothness when adding
        L.debug('Start sort')
        num_columns = np.shape(axes)[1]
        random_axes = np.random.choice(axes.shape[0], sample_size, replace=False)
        start = Line(np.array([0.] * num_columns), is_limit=True)
        end = Line(np.array([1.] * num_columns), is_limit=True)
        Line(axes[random_axes[0]], id=random_axes[0]).insert_between(start, end)
        for idx in random_axes[1:]:
            self.insert(start, Line(axes[idx, :], id=idx))

        if ax is not None:
            self.meshgrid(start, ax)

        return start, end

    def resort(self, start, end, sample_size, num_lines, ax=None):
        if end is None:
            lines_to_resort = [start.after]
        else:
            lines_to_resort = [start.after, end.before]

        random_lines = set(np.random.choice(num_lines, sample_size, replace=False))
        current = start.next()
        idx = 0
        while(current is not None):
            if idx in random_lines:
                lines_to_resort.append(current)
                L.debug('Will resort: %s' % current.id)
            idx += 1
            current = current.next()

        for line in lines_to_resort:
            L.debug('Start resort: %s' % line)
            self.remove(line)
            L.debug(str(line))
            self.insert(start, line)
            L.debug(str(line))

        if ax is not None:
            self.meshgrid(start, ax)

        return start, end

    def insert(self, start, line):
        # find the best place to insert a new line
        insert_after = start
        best_deltas = line.get_deltas_between(start, start.after)
        best_delta = np.sum(best_deltas)
        B = start.after
        A = B.after
        while(A is not None):
            deltas = line.get_deltas_between(B, A)
            delta = np.sum(deltas)
            if delta < best_delta:
                best_deltas = deltas
                best_delta = delta
                insert_after = B
            B = A
            A = B.after

        line.insert_between(insert_after, insert_after.after, best_deltas[2:])
        return line

    def remove(self, line):
        if line.is_limit: return
        before = line.before
        after = line.after
        line.before.after = line.after
        line.after.before = line.before
        line.before.set_triangle()
        line.after.set_triangle()
        line.before = line.after = None
        line.triangle = 0.
        return line

    def meshgrid(self, start, ax):

        joiners = []
        last_values = None
        current = start.next()
        num_columns = len(current.values)
        Xs = np.array(list(range(num_columns)))
        while(current is not None):
            if last_values is not None:
                joiners.append(np.sum(LC.value_weights * np.abs(current.values - last_values)))
            last_values = current.values
            current = current.next()

        joiners = np.array(joiners)
        print(joiners)
        ss = np.argsort(joiners)[::-1]
        print(ss)
        ssl = ss[:20]
        print(ssl)
        print(joiners[ssl])

        joiners = np.array(joiners)
        joiners -= np.min(joiners)
        joiners /= np.max(joiners)

        idx = 0
        current = start.next()
        while(current is not None):
            # plot main value lines
            ax.plot(
                Xs,
                np.array([idx] *num_columns),
                current.values,
                'grey')

            last_values = current.values
            idx += 1
            current = current.next()

        R = Rainbow(0, 1, 200, type='br')
        colors = [R.getHex(j) for j in joiners]

        idx = 0
        last_values = None
        current = start.next()
        while(current is not None):
            # plot joiner lines
            if last_values is not None:
                for _idx, v in enumerate(current.values):
                    ax.plot(
                        [Xs[_idx], Xs[_idx]],
                        [idx - 1, idx],
                        [last_values[_idx], v],
                        c=colors[idx-1])

            last_values = current.values
            idx += 1
            current = current.next()

        ax.azim = -59
        ax.elev = 86

class Line(object):

    _ID = 0
    def __init__(self, values, is_limit=False, id=None):
        self.values = values
        self.is_limit = is_limit
        self.triangle = 0.
        self.before = None
        self.after = None
        if id is None:
            self.id = self._ID; self.__class__._ID += 1
        else:
            self.id = id

    def next(self):
        if self.after.is_limit:
            return None
        return self.after

    def get_deltas_between(self, before, after):
        # returns a list with 5 elements.
        # The first two are triangles that would be removed and the last three
        # are triangles that would need to be created for before, self, after
        return [
            -1 * before.triangle,
            -1 * after.triangle,
            before.get_triangle(before.before, self),
            self.get_triangle(before, after),
            after.get_triangle(self, after.after)]

    def get_triangle(self, before, after):
        if self.is_limit: return 0.
        elif ((not self.before is None) and self.before.is_limit):
            return np.sum(LC.value_weights * np.abs(after.values-self.values))
        elif ((not self.after is None) and self.after.is_limit):
            return np.sum(LC.value_weights * np.abs(before.values-self.values))
        return (
            np.sum(LC.value_weights * np.abs(before.values-self.values)) +
            np.sum(LC.value_weights * np.abs(after.values-self.values))) / 2.

    def set_triangle(self, triangle=None):
        if triangle is None:
            if self.before is not None and self.after is not None:
                self.triangle = self.get_triangle(self.before, self.after)
            else:
                self.triangle = 0.
        else:
            self.triangle = triangle

    def init_ordering(self, start, end):
        # ensure lowest, middle, highest at start
        triangles = [
            start.get_triangle(self, end),
            self.get_triangle(start, end),
            end.get_triangle(start, self)]
        lowest = np.argmin(triangles)
        triangles = [0., triangles[lowest], 0.]

        if lowest == 0:
            if np.linalg.norm(self.values[0]) < np.linalg.norm(end.values[0]):
                start.insert_between(self, end, triangles)
                return self, end
            start.insert_between(end, self, triangles)
            return end, self

        elif lowest == 2:
            if np.linalg.norm(start.values[0]) < np.linalg.norm(self.values[0]):
                end.insert_between(start, self, triangles)
                return start, self
            end.insert_between(self, start, triangles)
            return self, start

        if np.linalg.norm(start.values[0]) < np.linalg.norm(end.values[0]):
            self.insert_between(start, end, triangles)
            return start, end
        self.insert_between(end, start, triangles)
        return end, start

    def insert_between(self, before, after, triangles=[None, None, None]):
        self.before = before
        before.after = self
        self.after = after
        after.before = self
        before.set_triangle(triangles[0])
        self.set_triangle(triangles[1])
        after.set_triangle(triangles[2])

    def print_chain(self):
        current = self
        chain = []
        while(True):
            chain.append(str(current))
            if current.after is None:
                break
            current = current.after

        return ' -> '.join(chain)

    def __str__(self):
        if self.before is None:
            before_id = 'NONE'
        else:
            before_id = self.before.id

        if self.after is None:
            after_id = 'NONE'
        else:
            after_id = self.after.id

        return 'ID: %s - %s - TR: %s - B4: %s - AF: %s' % tuple([str(i) for i in [
            self.id,
            self.values,
            self.triangle,
            before_id,
            after_id]])


###############################################################################
###############################################################################
###############################################################################
###############################################################################
