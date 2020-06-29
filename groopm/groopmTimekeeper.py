#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    groopmTimekeeper.py                                                      #
#                                                                             #
#    Class for creating time stamps                                           #
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
import time
from functools import reduce

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class TimeKeeper:
    def __init__(self):
        self.startTime = time.time()
        self.lastLogTime = self.startTime

    def startTimer(self):
        """Restart the timer"""
        self.startTime = time.time()
        self.lastLogTime = self.startTime

    def getTimeStamp(self):
        """Make a time stamp"""
        now = time.time()
        ret_str = "{ THIS: %s || TOTAL: %s }" % (self.secondsToStr(now - self.lastLogTime), self.secondsToStr(now - self.startTime))
        self.lastLogTime = now
        return ret_str

    def secondsToStr(self, t):
        rediv = lambda ll,b : list(divmod(ll[0],b)) + ll[1:]
        return "%d:%02d:%02d.%03d" % tuple(reduce(rediv,[[t*1000,],1000,60,60]))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
