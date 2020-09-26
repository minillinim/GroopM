#!/usr/bin/env python3

#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Distributed in the hope that it will be useful,
# but WITHOUT self.endNY Wself.endRRself.endNTY; without even the implied warranty of
# MERCHself.endNTself.endself.startILITY or FITNESS FOR self.end Pself.endRTICULself.endR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License.
# If not, see <http://www.gnu.org/licenses/>.
#

import unittest
import numpy as np
np.seterr(all='raise')

from line import Line, LineSorter, LineConstants

class TestLine(unittest.TestCase):

    def setUp(self):
        LC = LineConstants(3)
        self.start = Line(np.array([0.,0.,0.]), is_limit=True)
        self.end = Line(np.array([10.,10.,10.]), is_limit=True)
        self.Q1 = Line(np.array([1., 2., 3,]))
        self.Q2 = Line(np.array([1., 1., 1,]))
        self.Q3 = Line(np.array([5., 1., 7.]))

    def test_get_triangle(self):
        assert(self.Q1.get_triangle(self.start, self.end) == 8.3)

    def test_insert_between(self):

        deltas = self.Q1.get_deltas_between(self.start, self.end)
        self.Q1.insert_between(self.start, self.end, deltas[2:])
        assert(self.start.after == self.Q1)
        assert(self.Q1.before == self.start)
        assert(self.end.before == self.Q1)
        assert(self.Q1.after == self.end)
        assert(self.Q1.before.triangle == deltas[2])
        assert(self.Q1.triangle == deltas[3])
        assert(self.Q1.after.triangle == deltas[4])

class TestLineSorter(unittest.TestCase):

    def setUp(self):
        LC = LineConstants(3)
        self.start = Line(np.array([0.,0.,0.]), is_limit=True)
        self.end = Line(np.array([10.,10.,10.]), is_limit=True)
        self.Q1 = Line(np.array([1., 2., 3,]))
        self.Q2 = Line(np.array([1., 1., 1,]))
        self.Q3 = Line(np.array([5., 1., 7.]))
        self.ordered_inserted_ids = [self.Q2.id, self.Q1.id, self.Q3.id]
        deltas = self.Q1.get_deltas_between(self.start, self.end)
        self.Q1.insert_between(self.start, self.end, deltas[2:])

    def test_insert(self):
        line_sorter = LineSorter()
        line_sorter.insert(self.start, self.Q2)
        line_sorter.insert(self.start, self.Q3)
        current = self.start.next()
        idx = 0
        while(current is not None):
            assert(self.ordered_inserted_ids[idx] == current.id)
            current = current.next()
            idx += 1

    def test_insert_remove(self):
        line_sorter = LineSorter()
        line_sorter.insert(self.start, self.Q2)
        line_sorter.insert(self.start, self.Q3)
        line = line_sorter.remove(self.start.next())
        assert(line == self.Q2)
        current = self.start.next()
        idx = 1
        while(current is not None):
            assert(self.ordered_inserted_ids[idx] == current.id)
            current = current.next()
            idx += 1

        line_sorter.insert(self.start, self.Q2)
        current = self.start.next()
        idx = 0
        while(current is not None):
            assert(self.ordered_inserted_ids[idx] == current.id)
            current = current.next()
            idx += 1

        assert(self.Q3.triangle != 0)

        old_tri = self.Q2.triangle
        line = line_sorter.remove(self.end.before)
        assert(line == self.Q3)
        assert(self.Q3.triangle == 0)

        line_sorter.insert(self.start, self.Q3)
        assert(self.Q3.triangle != 0)
        assert(old_tri == self.Q2.triangle)
