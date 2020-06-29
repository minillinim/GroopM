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
__credits__ = ["Michael Imelfort"]
__license__ = "GPL3"
__version__ = "0.2.1"
__maintainer__ = "Michael Imelfort"
__email__ = "michael.imelfort@gmail.com"
__status__ = "Released"

###############################################################################


import math


###############################################################################
###############################################################################
###############################################################################
###############################################################################

class Rainbow:
    def __init__(self, lb, ub, res, type="rb"):
        """
        Specify the upper and lower bounds for your data.
        resolution refers to the number of bins which are available in this space
        Supports four heatmap types is red-blue, blue-red, red-green-blue and blue-green-red
        """

        # constants
        self.RB_lower_offset = 0.5
        self.RB_divisor = (2.0/3.0)
        self.RB_ERROR_COLOUR = [0,0,0]

        # set the limits
        self.lowerBound = lb
        self.upperBound = ub
        self.resolution = res
        self.tickSize = (self.upperBound - self.lowerBound)/(self.resolution - 1)

        # set the type, red-blue by default
        self.type = type
        self.redOffset = 0.0
        self.greenOffset = self.RB_divisor * math.pi * 2.0
        self.blueOffset = self.RB_divisor * math.pi

        self.ignoreRed = False
        self.ignoreGreen = True
        self.ignoreBlue = False

        self.lowerScale = 0.0
        self.upperScale = self.RB_divisor * math.pi

        if(self.type == "rbg"): # red-blue-green
            self.redOffset = 0.0
            self.greenOffset = self.RB_divisor * math.pi * 2.0
            self.blueOffset = self.RB_divisor * math.pi

            self.ignoreRed = False
            self.ignoreGreen = False
            self.ignoreBlue = False

            self.lowerScale = 0.0
            self.upperScale = (self.RB_divisor * math.pi * 2.0)

        elif(self.type == "gbr"): # green-blue-red
            self.redOffset = self.RB_divisor * math.pi * 2.0
            self.greenOffset = 0.0
            self.blueOffset = self.RB_divisor * math.pi

            self.ignoreRed = False
            self.ignoreGreen = False
            self.ignoreBlue = False

            self.lowerScale = 0.0
            self.upperScale = (self.RB_divisor * math.pi * 2.0)

        elif(self.type == "br"): # blue-red
            self.redOffset = self.RB_divisor * math.pi
            self.greenOffset = self.RB_divisor * math.pi * 2.0
            self.blueOffset = 0.0

            self.ignoreRed = False
            self.ignoreGreen = True
            self.ignoreBlue = False

            self.lowerScale = 0.0
            self.upperScale = (self.RB_divisor * math.pi)

        self.scaleMultiplier = (self.upperScale - self.lowerScale)/(self.upperBound - self.lowerBound)

    def getValue(self, val):
        """Get a raw value, not a color"""
        return (math.cos(val) + self.RB_lower_offset) * self.RB_divisor

    def getColor(self, val):
        """Return a color for the given value.

        If nothing makes sense. return black
        """
        if(val > self.upperBound or val < self.lowerBound):
            return self.RB_ERROR_COLOUR

        # normalise the value to suit the ticks
        normalised_value = round(val/self.tickSize) * self.tickSize

        # map the normalised value onto the horizontal scale
        scaled_value = (normalised_value - self.lowerBound) * self.scaleMultiplier + self.lowerScale

        red = 0
        green = 0
        blue = 0

        if(not self.ignoreRed):
            red = int(round(self.getValue(scaled_value - self.redOffset) * 255))
        if(not self.ignoreGreen):
            green = int(round(self.getValue(scaled_value - self.greenOffset) * 255))
        if(not self.ignoreBlue):
            blue = int(round(self.getValue(scaled_value - self.blueOffset) * 255))

        return (red, green, blue)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
