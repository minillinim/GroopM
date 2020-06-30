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

from ksig import KmerSigEngine

class Tests(unittest.TestCase):

    def test_init(self):
        KSE = KmerSigEngine()
        assert(KSE.num_mers == 136)
        assert(KSE.mer_2_idx['AAAA'] == 0)
        assert(KSE.mer_2_idx['TTTT'] == 0)
        assert(KSE.mer_2_idx['CGCT'] == KSE.mer_2_idx['AGCG'])

    def test_rev_comp_gc(self):
        KSE = KmerSigEngine()
        seq = 'ACNCTGGTTGCCT'
        assert(KSE.rev_comp(seq) == 'AGGCAACCAGNGT')
        assert(KSE.get_gc(seq) == 7/12)
        assert(KSE.get_gc('NNN') == 0)

    def test_shift_low_lexi(self):
        KSE = KmerSigEngine()
        assert(KSE.shift_low_lexi('ATTC') == 'ATTC')
        assert(KSE.shift_low_lexi('TTTC') == 'GAAA')
