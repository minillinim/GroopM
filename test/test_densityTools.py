#!/usr/bin/env python3

#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License.
# If not, see <http://www.gnu.org/licenses/>.
#

import unittest
import numpy as np

from densityTools import DensityKernel, DensityStore, DensityGraph

class Tests(unittest.TestCase):

    def test_to_kidx(self):
        kernel = DensityKernel(1000)
        assert(kernel.to_kidx(0) == 0)
        assert(kernel.to_kidx(0.001) == 0)
        assert(kernel.to_kidx(0.5) == 499)
        assert(kernel.to_kidx(1) == 999)

    def test_add_remove_cancels(self):
        num_columns = 6
        kernel = DensityKernel(1000)
        row = np.array([np.random.randint(100)/100. for _ in range(num_columns)])
        density_store = DensityStore(num_columns, kernel)

        assert(np.sum(density_store.store) == 0)
        density_store.add(row)
        assert(np.sum(density_store.store) != 0)
        density_store.remove(row)
        assert(np.sum(density_store.store) == 0)
