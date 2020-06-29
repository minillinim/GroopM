#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    groopmExceptions.py                                                      #
#                                                                             #
#    Like it says on the box                                                  #
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

#------------------------------------------------------------------------------
# BIN MANAGER
class GMBinException(BaseException): pass
class BinNotFoundException(GMBinException): pass
class ModeNotAppropriateException(GMBinException): pass

#------------------------------------------------------------------------------
# SOM MANAGER
class GMSOMException(BaseException): pass
class SOMDataNotFoundException(GMSOMException): pass
class SOMFlavourException(GMSOMException): pass
class SOMTypeException(GMSOMException): pass
class RegionsDontExistException(GMSOMException): pass

#------------------------------------------------------------------------------
# ARG PARSER
class GMARGException(BaseException): pass
class ExtractModeNotAppropriateException(GMARGException): pass

###############################################################################
###############################################################################
###############################################################################
###############################################################################

import traceback
class Tracer:
    def __init__(self, oldstream):
        self.oldstream = oldstream
        self.count = 0
        self.lastStack = None

    def write(self, s):
        newStack = traceback.format_stack()
        if newStack != self.lastStack:
            self.oldstream.write("".join(newStack))
            self.lastStack = newStack
        self.oldstream.write(s)

    def flush(self):
        self.oldstream.flush()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
