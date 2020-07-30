#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    profileManager.py                                                        #
#                                                                             #
#    GroopM - High level data management                                      #
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

from sys import exc_info, exit, stdout as sys_stdout
from operator import itemgetter
from colorsys import hsv_to_rgb as htr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
np.seterr(all='raise')
from scipy.spatial.distance import cdist, squareform
from scipy.spatial import KDTree as kdt
from scipy.stats import f_oneway, distributions
from sklearn.decomposition import TruncatedSVD

# GroopM imports
from .PCA import PCA, Center
from .mstore import GMDataManager
from .bin import Bin, mungeCbar
from . import groopmExceptions as ge
from .rainbow import Rainbow

###############################################################################

class ProfileManager:
    """Interacts with the groopm DataManager and local data fields

    Mostly a wrapper around a group of numpy arrays and a pytables quagmire
    """
    def __init__(self, dbFileName, force=False, scaleFactor=1000):
        # data
        self.dataManager = GMDataManager()  # most data is saved to hdf
        self.dbFileName = dbFileName        # db containing all the data we'd like to use
        self.condition = ""                 # condition will be supplied at loading time

        # --> NOTE: ALL of the arrays in this section are in sync
        # --> each one holds information for an individual contig
        self.indices = np.array([])         # indices into the data structure based on condition
        self.covProfiles = np.array([])     # coverage based coordinates
        self.transformedCP = np.array([])   # the munged data points
        self.averageCoverages = np.array([])# average coverage across all stoits
        self.normCoverages = np.array([])   # norm of the raw coverage vectors
        self.kmerSigs = np.array([])        # raw kmer signatures
        self.kmerNormSVD1 = np.array([])    # First SVD of kmer sigs normalized to [0, 1]
        self.kmerSVDs = np.array([])        # PCs of kmer sigs capturing specified variance
        self.stoitColNames = np.array([])
        self.contigNames = np.array([])
        self.contigLengths = np.array([])
        self.contigGCs = np.array([])
        self.colorMapGC = None

        self.binIds = np.array([])          # list of bin IDs
        # --> end section

        # meta
        self.validBinIds = {}               # valid bin ids -> numMembers
        self.isLikelyChimeric = {}          # indicates if a bin is likely to be chimeric
        self.binnedRowIndices = {}          # dictionary of those indices which belong to some bin
        self.restrictedRowIndices = {}      # dictionary of those indices which can not be binned yet
        self.numContigs = 0                 # this depends on the condition given
        self.numStoits = 0                  # this depends on the data which was parsed

        # contig links
        self.links = {}

        # misc
        self.forceWriting = force           # overwrite existng values silently?
        self.scaleFactor = scaleFactor      # scale every thing in the transformed data to this dimension

    def loadData(self,
        timer,
        condition,                 # condition as set by another function
        bids=[],                   # if this is set then only load those contigs with these bin ids
        verbose=True,              # many to some output messages
        silent=False,              # some to no output messages
        loadCovProfiles=True,
        loadRawKmers=True,
        loadKmerSVDs=True,
        makeColors=True,
        loadContigNames=True,
        loadContigLengths=True,
        loadContigGCs=True,
        loadBins=False,
        loadLinks=False):
        """Load pre-parsed data"""

        timer.getTimeStamp()
        if(silent):
            verbose=False
        if verbose:
            print("Loading data from:", self.dbFileName)

        try:
            self.numStoits = self.getNumStoits()
            self.condition = condition
            self.indices = self.dataManager.getConditionalIndices(
                self.dbFileName,
                condition=condition,
                silent=silent)
            if(verbose):
                print("    Loaded indices with condition:", condition)
            self.numContigs = len(self.indices)

            if self.numContigs == 0:
                print("    ERROR: No contigs loaded using condition:", condition)
                return

            if(not silent):
                print("    Working with: %d contigs" % self.numContigs)

            if(loadCovProfiles):
                if(verbose):
                    print("    Loading coverage profiles")
                self.covProfiles = self.dataManager.getCoverageProfiles(self.dbFileName, indices=self.indices)
                self.normCoverages = self.dataManager.getNormalisedCoverageProfiles(self.dbFileName, indices=self.indices)

                # from sklearn.decomposition import TruncatedSVD
                # import pandas as pd
                # import numpy as np
                # num_svds = 3
                # coverages_df = pd.DataFrame(self.covProfiles)
                # self._transformedCP = np.array(TruncatedSVD(
                #     n_components=num_svds,
                #     random_state=42).fit_transform(coverages_df))
                # self._transformedCP[:,0] -= self._transformedCP.min(axis=0)[0]
                #

                # ccf = np.corrcoef(np.transpose(self.covProfiles))
                # print(ccf)
                #
                # import code
                # code.interact(local=locals())

                self._transformedCP = self.covProfiles
                # self._transformedCP -= self._transformedCP.min(axis=0)
                # self._transformedCP /= self._transformedCP.max(axis=0)
                # self._transformedCP *= 1000

                # total_sums = np.argsort(np.sum(self._transformedCP, axis=0))

                self.transformedCP = self._transformedCP[:,[3,2,6]]
                # print(self._transformedCP)
                #
                # # # to spherical polar
                # self.transformedCP = np.zeros(np.shape(self._transformedCP))
                # self.transformedCP[:,2] = np.linalg.norm(self._transformedCP, axis=1)
                # non_zero_xs = np.where(self._transformedCP[:,1] != 0)[0]
                # self.transformedCP[non_zero_xs,0] = np.arctan(
                #     self._transformedCP[non_zero_xs,2] / self._transformedCP[non_zero_xs,1])
                # non_zero_phros = np.where(self.transformedCP[:,0] != 0)[0]
                # self.transformedCP[non_zero_phros,1] = self._transformedCP[non_zero_phros,0] / self.transformedCP[non_zero_phros,2]
                #
                # self.transformedCP[:,2] *= self.transformedCP[:,2]

                # work out average coverages
                self.averageCoverages = np.array([sum(i)/self.numStoits for i in self.covProfiles])

            if loadRawKmers:
                if(verbose):
                    print("    Loading RAW kmer sigs")
                self.kmerSigs = self.dataManager.getKmerSigs(self.dbFileName, indices=self.indices)

            if loadKmerSVDs:
                self.kmerSVDs = self.dataManager.getKmerSVDs(self.dbFileName, indices=self.indices)

                if(verbose):
                    print("    Loading SVD kmer sigs (" + str(len(self.kmerSVDs[0])) + " dimensional space)")

                self.kmerNormSVD1 = np.copy(self.kmerSVDs[:,0])
                self.kmerNormSVD1 -= self.kmerNormSVD1.min()
                self.kmerNormSVD1 /= self.kmerNormSVD1.max()

            if(loadContigNames):
                if(verbose):
                    print("    Loading contig names")
                self.contigNames = self.dataManager.getContigNames(self.dbFileName, indices=self.indices)

            if(loadContigLengths):
                self.contigLengths = self.dataManager.getContigLengths(self.dbFileName, indices=self.indices)
                if(verbose):
                    print("    Loading contig lengths (Total: %d BP)" % ( sum(self.contigLengths) ))

            if(loadContigGCs):
                self.contigGCs = self.dataManager.getContigGCs(self.dbFileName, indices=self.indices)
                if(verbose):
                    print("    Loading contig GC ratios (Average GC: %0.3f)" % ( np.mean(self.contigGCs) ))

            if(makeColors):
                if(verbose):
                    print("    Creating color map")

                # use HSV to RGB to generate colors
                S = 1       # SAT and VAL remain fixed at 1. Reduce to make
                V = 1       # Pastels if that's your preference...
                self.colorMapGC = self.createColorMapHSV()

            if(loadBins):
                if(verbose):
                    print("    Loading bin assignments")

                self.binIds = self.dataManager.getBins(self.dbFileName, indices=self.indices)

                if len(bids) != 0: # need to make sure we're not restricted in terms of bins
                    bin_stats = self.getBinStats()
                    for bid in bids:
                        try:
                            self.validBinIds[bid] = bin_stats[bid][0]
                            self.isLikelyChimeric[bid]= bin_stats[bid][1]
                        except KeyError:
                            self.validBinIds[bid] = 0
                            self.isLikelyChimeric[bid]= False

                else:
                    bin_stats = self.getBinStats()
                    for bid in bin_stats:
                        self.validBinIds[bid] = bin_stats[bid][0]
                        self.isLikelyChimeric[bid] = bin_stats[bid][1]

                # fix the binned indices
                self.binnedRowIndices = {}
                for i in range(len(self.indices)):
                    if(self.binIds[i] != 0):
                        self.binnedRowIndices[i] = True
            else:
                # we need zeros as bin indicies then...
                self.binIds = np.zeros(len(self.indices))

            if(loadLinks):
                self.loadLinks()

            self.stoitColNames = self.getStoitColNames()

        except:
            print("Error loading DB:", self.dbFileName, exc_info()[0])
            raise

    def reduceIndices(self, deadRowIndices):
        """purge indices from the data structures

        Be sure that deadRowIndices are sorted ascending
        """
        # strip out the other values
        self.indices = np.delete(self.indices, deadRowIndices, axis=0)
        self.covProfiles = np.delete(self.covProfiles, deadRowIndices, axis=0)
        self.transformedCP = np.delete(self.transformedCP, deadRowIndices, axis=0)
        self.contigNames = np.delete(self.contigNames, deadRowIndices, axis=0)
        self.contigLengths = np.delete(self.contigLengths, deadRowIndices, axis=0)
        self.contigGCs = np.delete(self.contigGCs, deadRowIndices, axis=0)
        self.kmerSVDs = np.delete(self.kmerSVDs, deadRowIndices, axis=0)
        #self.kmerSigs = np.delete(self.kmerSigs, deadRowIndices, axis=0)
        self.binIds = np.delete(self.binIds, deadRowIndices, axis=0)

#------------------------------------------------------------------------------
# GET / SET

    def getNumStoits(self):
        """return the value of numStoits in the metadata tables"""
        return self.dataManager.getNumStoits(self.dbFileName)

    def getMerColNames(self):
        """return the value of merColNames in the metadata tables"""
        return self.dataManager.getMerColNames(self.dbFileName)

    def getMerSize(self):
        """return the value of merSize in the metadata tables"""
        return self.dataManager.getMerSize(self.dbFileName)

    def getNumMers(self):
        """return the value of numMers in the metadata tables"""
        return self.dataManager.getNumMers(self.dbFileName)

    def getNumBins(self):
        """return the value of numBins in the metadata tables"""
        return self.dataManager.getNumBins(self.dbFileName)

    def setNumBins(self, numBins):
        """set the number of bins"""
        self.dataManager.setNumBins(self.dbFileName, numBins)

    def getStoitColNames(self):
        """return the value of stoitColNames in the metadata tables"""
        return np.array(self.dataManager.getStoitColNames(self.dbFileName).split(","))

    def isClustered(self):
        """Has the data been clustered already"""
        return self.dataManager.isClustered(self.dbFileName)

    def setClustered(self):
        """Save that the db has been clustered"""
        self.dataManager.setClustered(self.dbFileName, True)

    def isComplete(self):
        """Has the data been *completely* clustered already"""
        return self.dataManager.isComplete(self.dbFileName)

    def setComplete(self):
        """Save that the db has been completely clustered"""
        self.dataManager.setComplete(self.dbFileName, True)

    def getBinStats(self):
        """Go through all the "bins" array and make a list of unique bin ids vs number of contigs"""
        return self.dataManager.getBinStats(self.dbFileName)

    def setBinStats(self, binStats):
        """Store the valid bin Ids and number of members

        binStats is a list of tuples which looks like:
        [ (bid, numMembers, isLikelyChimeric) ]
        Note that this call effectively nukes the existing table
        """
        self.dataManager.setBinStats(self.dbFileName, binStats)
        self.setNumBins(len(binStats))

    def setBinAssignments(self, assignments, nuke=False):
        """Save our bins into the DB"""
        self.dataManager.setBinAssignments(
            self.dbFileName,
            assignments,
            nuke=nuke)

    def loadLinks(self):
        """Extra wrapper 'cause I am dumb"""
        self.links = self.getLinks()

    def getLinks(self):
        """Get contig links"""
        # first we get the absolute links
        absolute_links = self.dataManager.restoreLinks(self.dbFileName, self.indices)
        # now convert this into plain old row_indices
        reverse_index_lookup = {}
        for i in range(len(self.indices)):
            reverse_index_lookup[self.indices[i]] = i

        # now convert the absolute links to local ones
        relative_links = {}
        for cid in self.indices:
            local_cid = reverse_index_lookup[cid]
            relative_links[local_cid] = []
            try:
                for link in absolute_links[cid]:
                    relative_links[local_cid].append([reverse_index_lookup[link[0]], link[1], link[2], link[3]])
            except KeyError: # not everyone is linked
                pass

        return relative_links

#------------------------------------------------------------------------------
# DATA TRANSFORMATIONS

    def getAverageCoverage(self, rowIndex):
        """Return the average coverage for this contig across all stoits"""
        return sum(self.transformedCP[rowIndex])/self.numStoits

#------------------------------------------------------------------------------
# DEBUG CRUFT

    def rewriteBins(self):
        """rewrite the bins table in hdf5 based on numbers in meta-contigs"""
        bins = self.dataManager.getBins(self.dbFileName)
        bin_store = {}
        for c in bins:
            if c != 0:
                try:
                    bin_store[c] += 1
                except KeyError:
                    bin_store[c] = 1

        bin_stats = []
        for bid in bin_store:
            # [(bid, size, likelyChimeric)]
            bin_stats.append((bid, bin_store[bid], False))

        self.setBinStats(bin_stats)

#------------------------------------------------------------------------------
# IO and IMAGE RENDERING

    def createColorMapHSV(self):
      S = 1.0
      V = 1.0
      return LinearSegmentedColormap.from_list('GC', [htr((1.0 + np.sin(np.pi * (val/1000.0) - np.pi/2))/2., S, V) for val in range(0, 1000)], N=1000)

    def setColorMap(self, colorMapStr):
        if colorMapStr == 'HSV':
            S = 1
            V = 1
            self.colorMapGC = self.createColorMapHSV()
        elif colorMapStr == 'Accent':
            self.colorMapGC = get_cmap('Accent')
        elif colorMapStr == 'Blues':
            self.colorMapGC = get_cmap('Blues')
        elif colorMapStr == 'Spectral':
            self.colorMapGC = get_cmap('spectral')
        elif colorMapStr == 'Grayscale':
            self.colorMapGC = get_cmap('gist_yarg')
        elif colorMapStr == 'Discrete':
            discrete_map = [(0,0,0)]
            discrete_map.append((0,0,0))
            discrete_map.append((0,0,0))

            discrete_map.append((0,0,0))
            discrete_map.append((141/255.0,211/255.0,199/255.0))
            discrete_map.append((255/255.0,255/255.0,179/255.0))
            discrete_map.append((190/255.0,186/255.0,218/255.0))
            discrete_map.append((251/255.0,128/255.0,114/255.0))
            discrete_map.append((128/255.0,177/255.0,211/255.0))
            discrete_map.append((253/255.0,180/255.0,98/255.0))
            discrete_map.append((179/255.0,222/255.0,105/255.0))
            discrete_map.append((252/255.0,205/255.0,229/255.0))
            discrete_map.append((217/255.0,217/255.0,217/255.0))
            discrete_map.append((188/255.0,128/255.0,189/255.0))
            discrete_map.append((204/255.0,235/255.0,197/255.0))
            discrete_map.append((255/255.0,237/255.0,111/255.0))
            discrete_map.append((1,1,1))

            discrete_map.append((0,0,0))
            discrete_map.append((0,0,0))
            discrete_map.append((0,0,0))
            self.colorMapGC = LinearSegmentedColormap.from_list('GC_DISCRETE', discrete_map, N=20)

        elif colorMapStr == 'DiscretePaired':
            discrete_map = [(0,0,0)]
            discrete_map.append((0,0,0))
            discrete_map.append((0,0,0))

            discrete_map.append((0,0,0))
            discrete_map.append((166/255.0,206/255.0,227/255.0))
            discrete_map.append((31/255.0,120/255.0,180/255.0))
            discrete_map.append((178/255.0,223/255.0,138/255.0))
            discrete_map.append((51/255.0,160/255.0,44/255.0))
            discrete_map.append((251/255.0,154/255.0,153/255.0))
            discrete_map.append((227/255.0,26/255.0,28/255.0))
            discrete_map.append((253/255.0,191/255.0,111/255.0))
            discrete_map.append((255/255.0,127/255.0,0/255.0))
            discrete_map.append((202/255.0,178/255.0,214/255.0))
            discrete_map.append((106/255.0,61/255.0,154/255.0))
            discrete_map.append((255/255.0,255/255.0,179/255.0))
            discrete_map.append((217/255.0,95/255.0,2/255.0))
            discrete_map.append((1,1,1))

            discrete_map.append((0,0,0))
            discrete_map.append((0,0,0))
            discrete_map.append((0,0,0))
            self.colorMapGC = LinearSegmentedColormap.from_list('GC_DISCRETE', discrete_map, N=20)

    def plotUnbinned(self, timer, coreCut, ignoreContigLengths=False):
        """Plot all contigs over a certain length which are unbinned"""
        self.loadData(timer, "((length >= "+str(coreCut)+") & (bid == 0))")
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        if ignoreContigLengths:
            sc = ax1.scatter(self.transformedCP[:,0], self.transformedCP[:,1], self.transformedCP[:,2], edgecolors='none', c=self.contigGCs, cmap=self.colorMapGC, vmin=0.0, vmax=1.0, s=10, marker='.')
        else:
            sc = ax1.scatter(self.transformedCP[:,0], self.transformedCP[:,1], self.transformedCP[:,2], edgecolors='k', c=self.contigGCs, cmap=self.colorMapGC, vmin=0.0, vmax=1.0, s=np.sqrt(self.contigLengths), marker='.')
        sc.set_edgecolors = sc.set_facecolors = lambda *args:None  # disable depth transparency effect

        try:
            plt.show()
            plt.close(fig)
        except:
            print("Error showing image", exc_info()[0])
            raise
        del fig

    def plotAll(self, timer, coreCut, ignoreContigLengths=False):
        """Plot all contigs over a certain length which are unbinned"""
        self.loadData(
            timer,
            "((length >= "+str(coreCut)+"))",
            loadRawKmers=False,
            loadContigNames=False,
            loadContigLengths=False,
            loadContigGCs=False,
            makeColors=False)

        import numpy as np

        coloring = [int(i) for i in self.kmerSVDs[:,0] * 100]

        print(np.min(coloring), np.max(coloring))
        rainbow = Rainbow(np.min(coloring), np.max(coloring), 100, type='rgb')

        # import code
        # code.interact(local=locals())

        ax = plt.subplot(111)
        Xs = range(len(self.stoitColNames)-2)
        for idx, row in enumerate(self.covProfiles):
            ax.plot(Xs, np.sqrt(row[2:]), rainbow.getHex(coloring[idx]))

        plt.show()
        return


        sorted_idxs = np.argsort(np.sum(self.covProfiles, axis=0))
        self.covProfiles = self.covProfiles[:,sorted_idxs]
        num_points = len(self.covProfiles)
        num_cols = len(self.stoitColNames) - 1
        Xs = range(num_cols-2)
        diffs = np.zeros((num_points, num_cols))
        midpoints = np.zeros((num_points, num_cols))
        for i in range(1, num_cols):
            diffs[:,i-1] = self.covProfiles[:,i-1] - self.covProfiles[:,i]
            midpoints[:,i-1] = (self.covProfiles[:,i-1] + self.covProfiles[:,i]) / 2.

        ax = plt.subplot(211)

        for idx, diff in enumerate(diffs):
            ax.plot(Xs, diff, rainbow.getHex(self.contigGCs[idx]))

        ax = plt.subplot(212)

        for idx, midpoint in enumerate(midpoints):
            ax.plot(Xs, midpoint, rainbow.getHex(self.contigGCs[idx]))

        plt.show()
        return

        import numpy as np
        non_zero_zs = list(range(len(self.transformedCP[:,0])))#np.where(self.transformedCP[:,2] > 500)[0]

    def _plotAll(self, timer, coreCut, ignoreContigLengths=False):
        """Plot all contigs over a certain length which are unbinned"""
        self.loadData(
            timer,
            "((length >= "+str(coreCut)+"))",
            loadRawKmers=False,
            loadContigNames=False)

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        if ignoreContigLengths:
            sc = ax1.scatter(
                self.transformedCP[:,0][non_zero_zs],
                self.transformedCP[:,1][non_zero_zs],
                self.transformedCP[:,2][non_zero_zs],
                edgecolors='none',
                c=self.contigGCs[non_zero_zs],
                cmap=self.colorMapGC,
                vmin=0.0,
                vmax=1.0,
                marker='.',
                s=10.)
        else:
            sc = ax1.scatter(
                self.transformedCP[:,0][non_zero_zs],
                self.transformedCP[:,1][non_zero_zs],
                self.transformedCP[:,2][non_zero_zs],
                edgecolors='k',
                c=self.contigGCs[non_zero_zs],
                cmap=self.colorMapGC,
                vmin=0.0,
                vmax=1.0,
                marker='.',
                s=np.sqrt(self.contigLengths[non_zero_zs]))

        sc.set_edgecolors = sc.set_facecolors = lambda *args:None  # disable depth transparency effect

        cbar = plt.colorbar(sc, shrink=0.5)
        cbar.ax.tick_params()
        cbar.ax.set_title("% GC", size=10)
        cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        cbar.ax.set_ylim([0.15, 0.85])
        mungeCbar(cbar)


        ax1.azim = 0
        ax1.elev = 0

        ax1.set_xlabel('ax1')
        ax1.set_ylabel('ax2')
        ax1.set_zlabel('ax3')

        try:
            plt.show()
            plt.close(fig)
        except:
            print("Error showing image", exc_info()[0])
            raise
        del fig

    def plotTransViews(self, tag="fordens"):
        """Plot top, side and front views of the transformed data"""
        self.renderTransData(tag+"_top.png",azim = 0, elev = 90)
        self.renderTransData(tag+"_front.png",azim = 0, elev = 0)
        self.renderTransData(tag+"_side.png",azim = 90, elev = 0)

    def renderTransCPData(self,
                          fileName="",
                          show=True,
                          elev=45,
                          azim=45,
                          all=False,
                          showAxis=False,
                          primaryWidth=12,
                          primarySpace=3,
                          dpi=300,
                          format='png',
                          fig=None,
                          highlight=None,
                          restrictedBids=[],
                          alpha=1,
                          ignoreContigLengths=False):
        """Plot transformed data in 3D"""
        del_fig = False
        if(fig is None):
            fig = plt.figure()
            del_fig = True
        else:
            plt.clf()
        if(all):
            myAXINFO = {
                'x': {'i': 0, 'tickdir': 1, 'juggled': (1, 0, 2),
                'color': (0, 0, 0, 0, 0)},
                'y': {'i': 1, 'tickdir': 0, 'juggled': (0, 1, 2),
                'color': (0, 0, 0, 0, 0)},
                'z': {'i': 2, 'tickdir': 0, 'juggled': (0, 2, 1),
                'color': (0, 0, 0, 0, 0)},
            }

            ax = fig.add_subplot(131, projection='3d')
            sc = ax.scatter(self.transformedCP[:,0], self.transformedCP[:,1], self.transformedCP[:,2], edgecolors='k', c=self.contigGCs, cmap=self.colorMapGC, vmin=0.0, vmax=1.0, marker='.')
            sc.set_edgecolors = sc.set_facecolors = lambda *args:None  # disable depth transparency effect
            ax.azim = 0
            ax.elev = 0
            ax.set_xlim3d(0,self.scaleFactor)
            ax.set_ylim3d(0,self.scaleFactor)
            ax.set_zlim3d(0,self.scaleFactor)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            for axis in ax.w_xaxis, ax.w_yaxis, ax.w_zaxis:
                for elt in axis.get_ticklines() + axis.get_ticklabels():
                    elt.set_visible(False)
            ax.w_xaxis._AXINFO = myAXINFO
            ax.w_yaxis._AXINFO = myAXINFO
            ax.w_zaxis._AXINFO = myAXINFO

            ax = fig.add_subplot(132, projection='3d')
            sc = ax.scatter(self.transformedCP[:,0], self.transformedCP[:,1], self.transformedCP[:,2], edgecolors='k', c=self.contigGCs, cmap=self.colorMapGC, vmin=0.0, vmax=1.0, marker='.')
            sc.set_edgecolors = sc.set_facecolors = lambda *args:None  # disable depth transparency effect
            ax.azim = 90
            ax.elev = 0
            ax.set_xlim3d(0,self.scaleFactor)
            ax.set_ylim3d(0,self.scaleFactor)
            ax.set_zlim3d(0,self.scaleFactor)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            for axis in ax.w_xaxis, ax.w_yaxis, ax.w_zaxis:
                for elt in axis.get_ticklines() + axis.get_ticklabels():
                    elt.set_visible(False)
            ax.w_xaxis._AXINFO = myAXINFO
            ax.w_yaxis._AXINFO = myAXINFO
            ax.w_zaxis._AXINFO = myAXINFO

            ax = fig.add_subplot(133, projection='3d')
            sc = ax.scatter(self.transformedCP[:,0], self.transformedCP[:,1], self.transformedCP[:,2], edgecolors='k', c=self.contigGCs, cmap=self.colorMapGC, vmin=0.0, vmax=1.0, marker='.')
            sc.set_edgecolors = sc.set_facecolors = lambda *args:None  # disable depth transparency effect
            ax.azim = 0
            ax.elev = 90
            ax.set_xlim3d(0,self.scaleFactor)
            ax.set_ylim3d(0,self.scaleFactor)
            ax.set_zlim3d(0,self.scaleFactor)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            for axis in ax.w_xaxis, ax.w_yaxis, ax.w_zaxis:
                for elt in axis.get_ticklines() + axis.get_ticklabels():
                    elt.set_visible(False)
            ax.w_xaxis._AXINFO = myAXINFO
            ax.w_yaxis._AXINFO = myAXINFO
            ax.w_zaxis._AXINFO = myAXINFO
        else:
            ax = fig.add_subplot(111, projection='3d')
            if len(restrictedBids) == 0:
                if highlight is None:
                    print("BF:", np.shape(self.transformedCP))
                    if ignoreContigLengths:
                        sc = ax.scatter(self.transformedCP[:,0],
                                   self.transformedCP[:,1],
                                   self.transformedCP[:,2],
                                   edgecolors='none',
                                   c=self.contigGCs,
                                   cmap=self.colorMapGC,
                                   s=10.,
                                   vmin=0.0,
                                   vmax=1.0,
                                   marker='.')
                    else:
                        sc = ax.scatter(self.transformedCP[:,0],
                                   self.transformedCP[:,1],
                                   self.transformedCP[:,2],
                                   edgecolors='none',
                                   c=self.contigGCs,
                                   cmap=self.colorMapGC,
                                   vmin=0.0,
                                   vmax=1.0,
                                   s=np.sqrt(self.contigLengths),
                                   marker='.')
                    sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect
                else:
                    #draw the opaque guys first
                    """
                    sc = ax.scatter(self.transformedCP[:,0],
                                    self.transformedCP[:,1],
                                    self.transformedCP[:,2],
                                    edgecolors='none',
                                    c=self.contigGCs,
                                    cmap=self.colorMapGC,
                                    vmin=0.0,
                                    vmax=1.0,
                                    s=100.,
                                    marker='s',
                                    alpha=alpha)
                    sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect
                    """
                    # now replot the highlighted guys
                    disp_vals = np.array([])
                    disp_GCs = np.array([])

                    thrower = {}
                    hide_vals = np.array([])
                    hide_GCs = np.array([])

                    num_points = 0
                    for bin in highlight:
                        for row_index in bin.rowIndices:
                            num_points += 1
                            disp_vals = np.append(disp_vals, self.transformedCP[row_index])
                            disp_GCs = np.append(disp_GCs, self.contigGCs[row_index])
                            thrower[row_index] = False
                    # reshape
                    disp_vals = np.reshape(disp_vals, (num_points, 3))

                    num_points = 0
                    for i in range(len(self.indices)):
                        try:
                            thrower[i]
                        except KeyError:
                            num_points += 1
                            hide_vals = np.append(hide_vals, self.transformedCP[i])
                            hide_GCs = np.append(hide_GCs, self.contigGCs[i])
                    # reshape
                    hide_vals = np.reshape(hide_vals, (num_points, 3))

                    sc = ax.scatter(hide_vals[:,0],
                                    hide_vals[:,1],
                                    hide_vals[:,2],
                                    edgecolors='none',
                                    c=hide_GCs,
                                    cmap=self.colorMapGC,
                                    vmin=0.0,
                                    vmax=1.0,
                                    s=100.,
                                    marker='s',
                                    alpha=alpha)
                    sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

                    sc = ax.scatter(disp_vals[:,0],
                                    disp_vals[:,1],
                                    disp_vals[:,2],
                                    edgecolors='none',
                                    c=disp_GCs,
                                    cmap=self.colorMapGC,
                                    vmin=0.0,
                                    vmax=1.0,
                                    s=10.,
                                    marker='.')
                    sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

                    print(np.shape(disp_vals), np.shape(hide_vals), np.shape(self.transformedCP))

                # render color bar
                cbar = plt.colorbar(sc, shrink=0.5)
                cbar.ax.tick_params()
                cbar.ax.set_title("% GC", size=10)
                cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                cbar.ax.set_ylim([0.15, 0.85])
                mungeCbar(cbar)
            else:
                r_trans = np.array([])
                r_cols=np.array([])
                num_added = 0
                for i in range(len(self.indices)):
                    if self.binIds[i] not in restrictedBids:
                        r_trans = np.append(r_trans, self.transformedCP[i])
                        r_cols = np.append(r_cols, self.contigGCs[i])
                        num_added += 1
                r_trans = np.reshape(r_trans, (num_added,3))
                print(np.shape(r_trans))
                #r_cols = np.reshape(r_cols, (num_added,3))
                sc = ax.scatter(r_trans[:,0],
                                r_trans[:,1],
                                r_trans[:,2],
                                edgecolors='none',
                                c=r_cols,
                                cmap=self.colorMapGC,
                                s=10.,
                                vmin=0.0,
                                vmax=1.0,
                                marker='.')
                sc.set_edgecolors = sc.set_facecolors = lambda *args:None  # disable depth transparency effect

                # render color bar
                cbar = plt.colorbar(sc, shrink=0.5)
                cbar.ax.tick_params()
                cbar.ax.set_title("% GC", size=10)
                cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                cbar.ax.set_ylim([0.15, 0.85])
                mungeCbar(cbar)

            ax.azim = azim
            ax.elev = elev
            ax.set_xlim3d(0,self.scaleFactor)
            ax.set_ylim3d(0,self.scaleFactor)
            ax.set_zlim3d(0,self.scaleFactor)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            if(not showAxis):
                ax.set_axis_off()

        if(fileName != ""):
            try:
                if(all):
                    fig.set_size_inches(3*primaryWidth+2*primarySpace,primaryWidth)
                else:
                    fig.set_size_inches(primaryWidth,primaryWidth)
                plt.savefig(fileName,dpi=dpi,format=format)
            except:
                print("Error saving image",fileName, exc_info()[0])
                raise
        elif(show):
            try:
                plt.show()
            except:
                print("Error showing image", exc_info()[0])
                raise
        if del_fig:
            plt.close(fig)
            del fig

###############################################################################
###############################################################################
###############################################################################
###############################################################################

    def r2nderTransCPData(self,
                          fig,
                          alphaIndices=[],
                          visibleIndices=[],
                          alpha=1,
                          ignoreContigLengths=False,
                          elev=45,
                          azim=45,
                          fileName="",
                          dpi=300,
                          format='png',
                          primaryWidth=6,
                          title="",
                          showAxis=False,
                          showColorbar=True,):
        """Plot transformed data in 3D"""
        # clear any existing plot
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')

        # work out the coords an colours based on indices
        alpha_coords = self.transformedCP[alphaIndices]
        alpha_GCs = self.contigGCs[alphaIndices]
        visible_coords = self.transformedCP[visibleIndices]
        visible_GCs = self.contigGCs[visibleIndices]

        # lengths if needed
        if not ignoreContigLengths:
            alpha_lengths = self.contigLengths[alphaIndices]
            visible_lengths = self.contigLengths[visibleIndices]
        else:
            alpha_lengths = 10.
            visible_lengths = 10.

        # first plot alpha points
        if len(alpha_GCs) > 0:
            sc = ax.scatter(alpha_coords[:,0],
                            alpha_coords[:,1],
                            alpha_coords[:,2],
                            edgecolors='none',
                            c=alpha_GCs,
                            cmap=self.colorMapGC,
                            vmin=0.0,
                            vmax=1.0,
                            s=alpha_lengths,
                            marker='.',
                            alpha=alpha)
            sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

        # then plot full visible points
        if len(visible_GCs) > 0:
            sc = ax.scatter(visible_coords[:,0],
                            visible_coords[:,1],
                            visible_coords[:,2],
                            edgecolors='none',
                            c=visible_GCs,
                            cmap=self.colorMapGC,
                            s=visible_lengths,
                            vmin=0.0,
                            vmax=1.0,
                            marker='.')
            sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

        # render color bar
        if showColorbar:
            cbar = plt.colorbar(sc, shrink=0.5)
            cbar.ax.tick_params()
            cbar.ax.set_title("% GC", size=10)
            cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            cbar.ax.set_ylim([0.15, 0.85])
            mungeCbar(cbar)

        # set aspect
        ax.azim = azim
        ax.elev = elev

        # make it purdy
        ax.set_xlim3d(0,self.scaleFactor)
        ax.set_ylim3d(0,self.scaleFactor)
        ax.set_zlim3d(0,self.scaleFactor)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        plt.tight_layout()

        if title != "":
            plt.title(title)

        if(not showAxis):
            ax.set_axis_off()

        if(fileName != ""):
            try:
                fig.set_size_inches(primaryWidth,primaryWidth)
                plt.savefig(fileName,dpi=dpi,format=format)
            except:
                print("Error saving image",fileName, exc_info()[0])
                raise
        else:
            try:
                plt.show()
            except:
                print("Error showing image", exc_info()[0])
                raise

###############################################################################
###############################################################################
###############################################################################
###############################################################################
