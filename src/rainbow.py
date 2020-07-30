#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    rainbow.py                                                               #
#                                                                             #
#    Simple dimple heatmap                                                    #
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
__license__ = "GPL3"
__version__ = "0.2.1"
__maintainer__ = "Michael Imelfort"
__email__ = "michael.imelfort@gmail.com"

import math

class Rainbow:
    def __init__(self, lb, ub, res, type='rb'):
        """
        Specify the upper and lower bounds for your data.
        resolution refers to the number of bins which are available in this space
        Supports four heatmap types: red-blue, blue-red, red-green-blue and blue-green-red
        """

        self.lower_bound = lb
        self.upper_bound = ub
        self.resolution = res
        self.tick_size = \
            (self.upper_bound - self.lower_bound) / (self.resolution - 1)

        self.RB_lower_offset = 0.5
        self.RB_divisor = (2./3.)
        self.rotation_third = (2. * math.pi) / 3.
        self.RB_ERROR_COLOUR = (0, 0, 0)

        self.ignore_red = self.ignore_green = self.ignore_blue = False
        self.red_offset = self.green_offset = self.blue_offset = 0.

        if (type == 'rb'):
            self.blue_offset = self.rotation_third
            self.ignore_green = True

        elif(type == "rbg"): # red-blue-green
            self.green_offset = self.rotation_third * 2.
            self.blue_offset = self.rotation_third

        elif(type == "gbr"): # green-blue-red
            self.red_offset = self.rotation_third * 2.
            self.blue_offset = self.rotation_third

        elif(type == "br"): # blue-red
            self.red_offset = self.rotation_third
            self.ignore_green = True

        self.upperScale = max([self.red_offset, self.green_offset, self.blue_offset])
        self.scaleMultiplier = self.upperScale / (self.upper_bound - self.lower_bound)

    def getHex(self, val):
        return '#%s' % ''.join([format(_val, '02X') for _val in self.getColor(val)])

    def getColor(self, val):
        """Return a color for the given value.

        If nothing makes sense. return black
        """
        if(val > self.upper_bound or val < self.lower_bound):
            return self.RB_ERROR_COLOUR

        # normalise the value to suit the ticks
        normalised_value = round(val / self.tick_size) * self.tick_size

        # map the normalised value onto the horizontal scale
        scaled_value = (normalised_value - self.lower_bound) * self.scaleMultiplier

        red = 0
        green = 0
        blue = 0

        def scaled_2_rgb(scaled_value, offset):
            val = int(round(self.getValue(scaled_value - offset) * 255))
            if val < 0: return 0
            return val

        if not self.ignore_red:
            red = scaled_2_rgb(scaled_value, self.red_offset)
        if not self.ignore_green:
            green = scaled_2_rgb(scaled_value, self.green_offset)
        if not self.ignore_blue:
            blue = scaled_2_rgb(scaled_value, self.blue_offset)

        return (red, green, blue)

    def getValue(self, val):
        """Get a raw value, not a color"""
        return (math.cos(val) + self.RB_lower_offset) * self.RB_divisor
