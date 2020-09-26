#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    binManager.py                                                            #
#                                                                             #
#    GroopM - High level bin data management                                  #
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


__author__ = 'Michael Imelfort'
__copyright__ = 'Copyright 2012-2020'
__credits__ = ['Michael Imelfort']
__license__ = 'GPL3'
__maintainer__ = 'Michael Imelfort'
__email__ = 'michael.imelfort@gmail.com'

###############################################################################
from os.path import join as osp_join
from sys import exc_info, exit, stdout as sys_stdout
from operator import itemgetter

import numpy as np
np.seterr(all='raise')
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, distributions
from scipy.cluster.vq import kmeans, vq

# GroopM imports
from .profileManager import ProfileManager
from .bin import Bin, mungeCbar
from . import groopmExceptions as ge
from .groopmUtils import makeSurePathExists
from .ellipsoid import EllipsoidTool

import logging
L = logging.getLogger('groopm')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class BinManager:
    '''Class used for manipulating bins'''
    def __init__(self,
                 dbFileName='',
                 pm=None,
                 minSize=10,
                 minVol=1000000):
        # data storage
        if(dbFileName != ''):
            self.PM = ProfileManager(dbFileName)
        elif(pm is not None):
            self.PM = pm

        # all about bins
        self.nextFreeBinId = 0                      # increment before use!
        self.bins = {}                              # bid -> Bin

        # misc
        self.minSize=minSize           # Min number of contigs for a bin to be considered legit
        self.minVol=minVol             # Override on the min size, if we have this many BP

    def setColorMap(self, colorMapStr):
        self.PM.setColorMap(colorMapStr)

#------------------------------------------------------------------------------
# LOADING / SAVING

    def loadBins(self,
         timer,
         getUnbinned=False,
         bids=[],
         makeBins=False,
         loadKmerSVDs=False,
         loadRawKmers=False,
         loadCovProfiles=True,
         loadContigLengths=True,
         loadLinks=False,
         loadContigNames=True,
         cutOff=0):
        '''Load data and make bin objects'''
        # build the condition

        if getUnbinned:
            # get everything
            condition = '(length >= %d) ' % cutOff
        else:
            # make sense of bin information
            if bids == []:
                condition = '((length >= %d) & (bid != 0))' % cutOff
            else:
                condition = '((length >= %d) & %s)' % (
                    cutOff,
                    ' | '.join(['(bid == %d)' % bid for bid in bids]))

        # if we're going to make bins then we'll need kmer sigs
        if(makeBins):
            loadKmerSVDs=True
            loadCovProfiles=True

        self.PM.loadData(
            timer,
            condition,
            bids=bids,
            loadCovProfiles=loadCovProfiles,
            loadKmerSVDs=loadKmerSVDs,
            loadRawKmers=loadRawKmers,
            makeColors=True,
            loadContigNames=loadContigNames,
            loadContigLengths=loadContigLengths,
            loadBins=True,
            loadLinks=loadLinks)

        # exit if no bins loaded
        if self.PM.numContigs == 0:
            return

        if(makeBins):
            L.info('Making bin objects')
            self.makeBins(self.getBinMembers())
            L.info('Loaded %d bins from database' % len(self.bins))

        L.info(str(timer.getTimeStamp()))
        sys_stdout.flush()

    def getBinMembers(self):
        '''Munge the raw data into something more usable

        self.PM.binIds is an array, contains 0 for unassigned rows
        By default this creates an array for the '0' bin. You need
        to ignore it later if you want to
        '''
        # fill them up
        bin_members = {0:[]}
        for row_index in range(np.size(self.PM.indices)):
            try:
                bin_members[self.PM.binIds[row_index]].append(row_index)
            except KeyError:
                bin_members[self.PM.binIds[row_index]] = [row_index]

        # we need to get the largest BinId in use
        if len(bin_members) > 0:
            self.nextFreeBinId = np.max(list(bin_members.keys()))
        return bin_members

    def makeBins(self, binMembers, zeroIsBin=False):
        '''Make bin objects from loaded data'''
        invalid_bids = []
        for bid in binMembers:
            if bid != 0 or zeroIsBin:
                if len(binMembers[bid]) == 0:
                    invalid_bids.append(bid)
                else:
                    self.bins[bid] = Bin(
                        np.array(binMembers[bid]), bid, self.PM.scaleFactor-1)
                    self.bins[bid].makeBinDist(
                        self.PM.transformedCP,
                        self.PM.averageCoverages,
                        self.PM.kmerNormSVD1,
                        self.PM.kmerSVDs,
                        self.PM.contigGCs,
                        self.PM.contigLengths)
        if len(invalid_bids) != 0:
            L.error('MT bins!')
            L.error(str(invalid_bids))
            exit(-1)

    def saveBins(self, binAssignments={}, nuke=False):
        '''Save binning results

        binAssignments is a hash of LOCAL row indices Vs bin ids
        { row_index : bid }
        PM.setBinAssignments needs GLOBAL row indices

        We always overwrite the bins table (It is smallish)
        '''
        # save the bin assignments
        L.info('saving bins')
        self.PM.setBinAssignments(
            self.getGlobalBinAssignments(binAssignments), # convert to global indices
            nuke=nuke)

        # overwrite the bins table
        self.setBinStats()

    def getGlobalBinAssignments(self, binAssignments={}):
        '''Merge the bids, raw DB indexes and core information so we can save to disk

        returns a hash of type:

        { global_index : bid }
        '''
        # we need a mapping from cid (or local index) to to global index to binID
        bin_assignment_update = {}

        if binAssignments != {}:
            # we have been told to do only these guys
            for row_index in binAssignments:
                bin_assignment_update[self.PM.indices[row_index]] = binAssignments[row_index]

        else:
            # this are all our regularly binned guys
            L.info(str(self.getBids()))
            for bid in self.getBids():
                for row_index in self.bins[bid].rowIndices:
                    bin_assignment_update[self.PM.indices[row_index]] = bid

        print(bin_assignment_update)
        return bin_assignment_update

    def setBinStats(self):
        '''Update / overwrite the table holding the bin stats

        Note that this call effectively nukes the existing table
        '''

        # create and array of tuples:
        # [(bid, size, likelyChimeric)]
        bin_stats = []
        for bid in self.getBids():
            # no point in saving empty bins
            if np.size(self.bins[bid].rowIndices) > 0:
                bin_stats.append((bid, np.size(self.bins[bid].rowIndices), self.PM.isLikelyChimeric[bid]))
        self.PM.setBinStats(bin_stats)

#------------------------------------------------------------------------------
# REMOVE ALREADY LOADED BINS

    def removeBinAndIndices(self, bid):
        '''Remove indices from the PM based on bin identity

        'unload' some data
        '''
        # get some info
        rem_bin = self.getBin(bid)
        original_length = len(self.PM.indices)
        rem_list = np.sort(rem_bin.rowIndices)

        # affect the raw data in the PM
        self.PM.reduceIndices(rem_list)
        del self.PM.validBinIds[bid]

        # remove the bin here
        del self.bins[bid]

        # now fix all the rowIndices in all the other bins
        for bid in self.getBids():
            self.bins[bid].rowIndices = self.fixRowIndexLists(original_length, np.sort(self.bins[bid].rowIndices), rem_list)

    def fixRowIndexLists(self, originalLength, oldList, remList):
        '''Fix up row index lists which reference into the
        data structure after a call to reduceIndices

        originalLength is the length of all possible row indices
        before the removal (ie self.indices)
        oldList is the old list of row indices
        remList is the list of indices to be removed

        BOTH OLD AND REM LIST MUST BE SORTED ASCENDING!
        '''
        shift_down = 0;
        old_list_index = 0
        new_list = np.array([])
        for i in range(originalLength):
            if(i in remList):
                shift_down+=1
            elif(i in oldList):
                new_list = np.append(new_list, oldList[old_list_index]-shift_down)
                old_list_index += 1
        return new_list

#------------------------------------------------------------------------------
# LINKS

    def getLinkingContigs(self, bid):
        '''Get all contigs and their bin IDs which link to contigs in this bin'''
        bin2count = {}
        for row_index in bin.rowIndices:
            try:
                for link in self.PM.links[row_index]:
                    try:
                        link_bid = self.PM.binIds[link[0]]
                        if link_bid != bid and link_bid != 0:
                            try:
                                bin2count[link_bid] += 1.0
                            except KeyError:
                                bin2count[link_bid] = 1.0
                    except KeyError:
                        pass
            except KeyError:
                pass
        return bin2count

    def getConnectedBins(self, rowIndex):
        '''Get a  list of bins connected to this contig'''
        ret_links = []
        for link in self.PM.links[rowIndex]:
            cid = link[0]
            try:
                bid = self.PM.binIds[cid]
            except KeyError:
                bid = 0
            ret_links.append((cid, bid, link[1]))
        return ret_links

    def getAllLinks(self):
        '''Return a sorted array of all links between all bins'''
        bids = self.getBids()
        # first, work out who links with whom...
        all_links = {}
        for bid in bids:
            links = self.getLinkingContigs(bid)
            # links is a hash of type bid : num_links
            for link in links:
                key = self.makeBidKey(bid, link)
                if key not in all_links:
                    all_links[key] = links[link]

        # sort and return
        return sorted(iter(all_links.items()), key=itemgetter(1), reverse=True)

    def getWithinLinkProfiles(self):
        '''Determine the average number of links between contigs for all bins'''
        bids = self.getBids()
        link_profiles = {}
        for bid in bids:
            link_profiles[bid] = self.getWithinBinLinkProfile(bid)
        return link_profiles

    def getWithinBinLinkProfile(self, bid):
        '''Determine the average number of links between contigs in a bin'''
        links = []
        min_links = 1000000000
        for row_index in bin.rowIndices:
            try:
                for link in self.PM.links[row_index]:
                    link_bid = self.PM.binIds[link[0]]
                    if link_bid == bid:
                        links.append(link[1])
                        if link[1] < min_links:
                            min_links = link[1]
            except KeyError:
                pass
        return (np.mean(links), np.std(links), min_links)

#------------------------------------------------------------------------------
# BIN UTILITIES

    def isGoodBin(self, totalBP, binSize, ms=0, mv=0):
        '''Does this bin meet my exacting requirements?'''
        if(ms == 0):
            ms = self.minSize               # let the user choose
        if(mv == 0):
            mv = self.minVol                # let the user choose

        if(totalBP < mv):                   # less than the good volume
            if(binSize > ms):               # but has enough contigs
                return True
        else:                               # contains enough bp to pass regardless of number of contigs
            return True
        return False

    def getBids(self):
        '''Return a sorted list of bin ids'''
        return sorted(self.bins.keys())

    def getCentroidProfiles(self, mode='mer'):
        '''Return an array containing the centroid stats for each bin'''
        if(mode == 'mer'):
            ret_vecs = np.zeros((len(self.bins)))
            outer_index = 0
            for bid in self.getBids():
                ret_vecs[outer_index] = self.bins[bid].kValMeanNormPC1
                outer_index += 1
            return ret_vecs
        elif(mode == 'cov'):
            ret_vecs = np.zeros((len(self.bins), len(self.PM.transformedCP[0])))
            outer_index = 0
            for bid in self.getBids():
                ret_vecs[outer_index] = self.bins[bid].covMedians
                outer_index += 1
            return ret_vecs
        else:
            raise ge.ModeNotAppropriateException(
                'Mode', mode, 'unknown')

    def split(self,
        bid,
        n,
        mode='kmer',
        auto=False,
        saveBins=False,
        printInstructions=True,
        use_elipses=True):
        '''split a bin into n parts

        if auto == True, then just railroad the split
        if test == True, then test via merging
        if savebins == True, save the split (if you will do it)
        if MCut != 0, carry the split through only if both daughter bins have an M
          less than MCut
        '''
        # we need to work out which profile to cluster on
        if(printInstructions and not auto):
            self.printSplitInstructions()

        # make some split bins
        # bids[0] holds the original bin id
        (bin_assignment_update, bids) = self.getSplitties(bid, n, mode)

        if(auto and saveBins):
            # charge on through
            self.deleteBins([bids[0]], force=True)  # delete the combined bin
            # save new bins
            self.saveBins(binAssignments=bin_assignment_update)
            return

        # we will need to confer with the user
        # plot some stuff
        # sort the bins by kmer val
        bid_tuples = [(tbid, self.bins[tbid].kValMeanNormPC1) for tbid in bids[1:]]
        bid_tuples.sort(key=itemgetter(1))
        index = 1
        for pair in bid_tuples:
            bids[index] = pair[0]
            index += 1

        self.plotSideBySide(bids, use_elipses=use_elipses)

        user_option = self.promptOnSplit(n, mode)
        if(user_option == 'Y'):
            if(saveBins):
                # save the temp bins
                self.deleteBins([bids[0]], force=True)  # delete the combined bin
                # save new bins
                self.saveBins(binAssignments=bin_assignment_update)
            return

        # If we're here than we don't need the temp bins
        # remove this query from the list so we don't delete him
        del bids[0]
        self.deleteBins(bids, force=True)

        # see what the user wants to do
        if(user_option == 'N'):
            return
        elif(user_option == 'C'):
            self.split(
                bid,
                n,
                mode='cov',
                auto=auto,
                saveBins=saveBins,
                printInstructions=False,
                use_elipses=use_elipses)
        elif(user_option == 'K'):
            self.split(
                bid,
                n,
                mode='kmer',
                auto=auto,
                saveBins=saveBins,
                printInstructions=False,
                use_elipses=use_elipses)
        elif(user_option == 'L'):
            self.split(
                bid,
                n,
                mode='len',
                auto=auto,
                saveBins=saveBins,
                printInstructions=False,
                use_elipses=use_elipses)
        elif(user_option == 'P'):
            not_got_parts = True
            parts = 0
            while(not_got_parts):
                try:
                    parts = int(input('Enter new number of parts:'))
                except ValueError:
                    print('You need to enter an integer value!')
                    parts = 0
                if(1 == parts):
                    print('Don\'t be a silly sausage!')
                elif(0 != parts):
                    not_got_parts = False
                    self.split(
                        bid,
                        parts,
                        mode=mode,
                        auto=auto,
                        saveBins=saveBins,
                        printInstructions=False,
                        use_elipses=use_elipses)

    def getSplitties(self, bid, n, mode):
        '''Return a set of split bins'''
        obs = np.array([])
        if(mode=='kmer'):
            obs = np.array([self.PM.kmerNormSVD1[i] for i in self.getBin(bid).rowIndices])
        elif(mode=='cov'):
            obs = np.array([self.PM.covProfiles[i] for i in self.getBin(bid).rowIndices])
        elif(mode=='len'):
            obs = np.array([self.PM.contigLengths[i] for i in self.getBin(bid).rowIndices])

        # do the clustering
        try:
            centroids,_ = kmeans(obs,n)
        except ValueError:
            return False
        idx,_ = vq(obs,centroids)

        # build some temp bins
        # this way we can show the user what the split will look like
        idx_sorted = np.argsort(np.array(idx))
        current_group = 0
        bids = [bid]
        bin_assignment_update = {} # row index to bin id
        holding_array = np.array([])
        split_bin = None
        for i in idx_sorted:
            if(idx[i] != current_group):
                # bin is full!
                holding_array = holding_array.astype(int)
                split_bin = self.makeNewBin(holding_array)

                for row_index in holding_array:
                    bin_assignment_update[row_index] = split_bin.id
                split_bin.makeBinDist(self.PM.transformedCP, self.PM.averageCoverages, self.PM.kmerNormSVD1, self.PM.kmerSVDs, self.PM.contigGCs, self.PM.contigLengths)
                bids.append(split_bin.id)
                holding_array = np.array([])
                current_group = idx[i]
            holding_array = np.append(holding_array, int(self.getBin(bid).rowIndices[i]))

        # do the last one
        if(np.size(holding_array) != 0):
            holding_array = holding_array.astype(int)
            split_bin = self.makeNewBin(holding_array)
            for row_index in holding_array:
                bin_assignment_update[int(row_index)] = split_bin.id
            split_bin.makeBinDist(self.PM.transformedCP, self.PM.averageCoverages, self.PM.kmerNormSVD1, self.PM.kmerSVDs, self.PM.contigGCs, self.PM.contigLengths)
            bids.append(split_bin.id)

        return (bin_assignment_update, bids)

    def shouldMerge(self, bin1, bin2, ignoreCov=False, ignoreMer=False, merTol=0, confidence=0.95):
        '''Determine whether its wise to merge two bins

        Perfoms a one-way anova to determine if the larger bin would be
        significantly changed if it consumed the smaller

        OR does a tolerance test on kmerNormSVD1. We assume that bin1 is larger than bin2
        '''
        if(bin1.id != bin2.id):
            if not ignoreCov: # work out coverage distributions
                b1_c_dist = bin1.getAverageCoverageDist(self.PM.averageCoverages)
                b2_c_dist = bin2.getAverageCoverageDist(self.PM.averageCoverages)
                c_dist_1 = b1_c_dist
                if(bin1.binSize < bin2.binSize):
                    c_dist_1 = b2_c_dist
                c_dist_2 = np.append(b2_c_dist, b1_c_dist)
                cov_match = self.isSameVariance(
                    c_dist_1,
                    c_dist_2,
                    confidence=confidence,
                    tag='COV:')
            else:
                cov_match = True

            if not ignoreMer: # work out kmer distributions
                if not cov_match:
                    return False
                if merTol != 0:
                    # Tolerance based testing
                    upper_k_val_cut = bin1.kValMeanNormPC1 + merTol * bin1.kValStdevNormPC1
                    lower_k_val_cut = bin1.kValMeanNormPC1 - merTol * bin1.kValStdevNormPC1

                    if bin2.kValMeanNormPC1 >= lower_k_val_cut and bin2.kValMeanNormPC1 <= upper_k_val_cut:
                        mer_match = True
                    else:
                        mer_match = False
                else:
                    b1_k_dist = bin1.getkmerValDist(self.PM.kmerNormSVD1)
                    b2_k_dist = bin2.getkmerValDist(self.PM.kmerNormSVD1)
                    k_dist_1 = b1_k_dist
                    if(bin1.binSize < bin2.binSize):
                        k_dist_1 = b2_k_dist
                    k_dist_2 = np.append(b2_k_dist, b1_k_dist)
                    mer_match = self.isSameVariance(
                        k_dist_1,
                        k_dist_2,
                        confidence=confidence,
                        tag='MER: %0.4f %0.4f' % (np.mean(k_dist_2), np.std(k_dist_2)))
            else:
                mer_match = True

            return cov_match and mer_match

        return False

    def isSameVariance(self, dist1, dist2, confidence=0.95, tag=''):
        '''Test to see if the kmerValues for two bins are the same'''
        F_cutoff =  distributions.f.ppf(confidence, 2, len(dist1)+len(dist2)-2)
        F_value = f_oneway(dist1,dist2)[0]
        if tag != '':
            L.info('%s [V: %f, C: %f]' % (tag, F_value, F_cutoff))
        return F_value < F_cutoff

    def merge(self,
        bids,
        auto=False,
        manual=False,
        newBid=False,
        saveBins=False,
        printInstructions=True,
        use_elipses=True):
        '''Merge two or more bins

        It's a bit strange to have both manual and auto params
        NOTE: manual ALWAYS overrides auto. In the condensing code, auto is
        set programmaticaly, manual is always set by the user. So we listen
        to manual first
        '''
        parent_bin = None

        if(printInstructions and not auto):
            self.printMergeInstructions()

        if(newBid):
            # we need to make this into a new bin
            parent_bin = self.makeNewBin()

            # now merge it with the first in the new list
            dead_bin = self.getBin(bids[0])
            for row_index in dead_bin.rowIndices:
                self.PM.binIds[row_index] = parent_bin.id
            parent_bin.consume(
                self.PM.transformedCP,
                self.PM.averageCoverages,
                self.PM.kmerNormSVD1,
                self.PM.kmerSVDs,
                self.PM.contigGCs,
                self.PM.contigLengths,
                dead_bin)
            self.deleteBins([bids[0]], force=True)
        else:
            # just use the first given as the parent
            parent_bin = self.getBin(bids[0])

        # a merged bin specified by the user should not be considered chimeric since
        # they are indicating both original bins were reasonable
        if manual:
            self.PM.isLikelyChimeric[parent_bin.id] = False

        # let this guy consume all the other guys
        ret_val = 0
        some_merged = False
        for i in range(1,len(bids)):
            continue_merge = False
            dead_bin = self.getBin(bids[i])
            if(auto and not manual):
                ret_val = 2
                continue_merge = True
            else:
                tmp_bin = self.makeNewBin(np.concatenate([parent_bin.rowIndices,dead_bin.rowIndices]))
                tmp_bin.makeBinDist(
                    self.PM.transformedCP,
                    self.PM.averageCoverages,
                    self.PM.kmerNormSVD1,
                    self.PM.kmerSVDs,
                    self.PM.contigGCs,
                    self.PM.contigLengths)

                self.plotSideBySide(
                    [parent_bin.id, dead_bin.id, tmp_bin.id],
                    use_elipses=use_elipses)

                self.deleteBins([tmp_bin.id], force=True)
                user_option = self.promptOnMerge(bids=[parent_bin.id,dead_bin.id])
                if(user_option == 'N'):
                    print('Merge skipped')
                    ret_val = 1
                    continue_merge=False
                elif(user_option == 'Q'):
                    print('All mergers skipped')
                    return 0
                else:
                    ret_val = 2
                    continue_merge=True
            if(continue_merge):

                for row_index in dead_bin.rowIndices:
                    self.PM.binIds[row_index] = parent_bin.id

                parent_bin.consume(
                    self.PM.transformedCP,
                    self.PM.averageCoverages,
                    self.PM.kmerNormSVD1,
                    self.PM.kmerSVDs,
                    self.PM.contigGCs,
                    self.PM.contigLengths,
                    dead_bin)
                self.deleteBins([bids[i]], force=True)
                some_merged = True

        if some_merged:
            # Fix up the r2b indices and bin updates
            parent_bid = parent_bin.id
            bin_assignment_update = {}
            for row_index in parent_bin.rowIndices:
                bin_assignment_update[row_index] = parent_bid
                try:
                    self.PM.binIds[row_index] = parent_bid
                except KeyError:
                    pass

            if saveBins:
                self.saveBins(binAssignments=bin_assignment_update)

        return ret_val

    def makeBidKey(self, bid1, bid2):
        '''Make a unique key from two bids'''
        if(bid1 < bid2):
            return (bid1, bid2)
        return (bid2, bid1)

    def getBin(self, bid):
        '''get a bin or raise an error'''
        if bid in self.bins:
            return self.bins[bid]
        else:
            raise ge.BinNotFoundException('Cannot find: in bins dicts' % (bid))

    def getChimericBinIds(self):
        bids = []
        for bid in self.bins:
            if self.PM.isLikelyChimeric[bid]:
                bids.append(bid)

        return bids

    def getNonChimericBinIds(self):
        bids = []
        for bid in self.bins:
            if not self.PM.isLikelyChimeric[bid]:
                bids.append(bid)

        return bids

    def deleteBins(self, bids, force=False, freeBinnedRowIndices=False, saveBins=False):
        '''Purge a bin from our lists'''
        if(not force):
            user_option = self.promptOnDelete(bids)
            if(user_option != 'Y'):
                return False
        bin_assignment_update = {}
        for bid in bids:
            if bid in self.bins:
                if(freeBinnedRowIndices):
                    for row_index in self.bins[bid].rowIndices:
                        try:
                            del self.PM.binnedRowIndices[row_index]
                        except KeyError:
                            L.error(bid, row_index, 'FUNG')
                        self.PM.binIds[row_index] = 0

                        bin_assignment_update[row_index] = 0
                del self.bins[bid]
                del self.PM.isLikelyChimeric[bid]
            else:
                raise ge.BinNotFoundException('Cannot find: %s in bins dicts' %(bid))

        if(saveBins):
            self.saveBins(binAssignments=bin_assignment_update)
        return True

    def makeNewBin(self, rowIndices=np.array([]), bid=None):
        '''Make a new bin and add to the list of existing bins'''
        if bid is None:
            self.nextFreeBinId +=1
            bid = self.nextFreeBinId

        self.PM.isLikelyChimeric[bid] = False
        self.bins[bid] = Bin(rowIndices, bid, self.PM.scaleFactor-1)
        return self.bins[bid]

#------------------------------------------------------------------------------
# UI

    def printMergeInstructions(self):
        input('\n'.join([
            '****************************************************************',
            ' MERGING INSTRUCTIONS - PLEASE READ CAREFULLY',
            '****************************************************************',
            ' The computer cannot always be trusted to perform bin mergers',
            ' automatically, so during merging you may be shown a 3D plot',
            ' which should help YOU determine whether or not the bins should',
            ' be merged. Look carefully at each plot and then close the plot',
            ' to continue with the merging operation.',
            ' The image on the far right shows the bins after merging',
            ' Press any key to produce plots...']))
        print('****************************************************************')

    def printSplitInstructions(self):
        input('\n'.join([
            '****************************************************************',
            ' SPLITTING INSTRUCTIONS - PLEASE READ CAREFULLY',
            '****************************************************************',
            ' The computer cannot always be trusted to perform bin splits',
            ' automatically, so during splitting you may be shown a 3D plot',
            ' which should help YOU determine whether or not the bin should',
            ' be split. Look carefully at each plot and then close the plot',
            ' to continue with the splitting operation.',
            ' Press any key to produce plots...']))
        print('****************************************************************')

    def getPlotterMergeIds(self):
        '''Prompt the user for ids to be merged and check that it's all good'''
        input_not_ok = True
        ret_bids = []
        while(input_not_ok):
            ret_bids = []
            option = input('Please enter "space" separated bin Ids or "q" to quit: ')
            if(option.upper() == 'Q'):
                return []
            bids = option.split(' ')
            for bid in bids:
                try:
                    # check that it's an int
                    i_bid = int(bid)
                    # check that it's in the bins list
                    if(i_bid not in self.bins):
                        print('**Error: bin %s not found' % bid)
                        input_not_ok = True
                        break
                    input_not_ok = False
                    ret_bids.append(i_bid)
                except ValueError:
                    print('**Error: invalid value: %s' % bid)
                    input_not_ok = True
                    break
        return ret_bids

    def promptOnMerge(self, bids=[], minimal=False):
        '''Check that the user is ok with this merge'''
        input_not_ok = True
        valid_responses = ['Y','N','Q']
        vrs = ','.join([str.lower(str(x)) for x in valid_responses])
        bin_str = ''
        if(len(bids) != 0):
            bin_str = ': %s' % str(bids[0])
            for i in range(1, len(bids)):
                bin_str += ' and %s' % str(bids[i])
        while(input_not_ok):
            if(minimal):
                option = input(' Merge? (%s) : ' % vrs).upper()
            else:
                option = input('\n'.join([
                    ' ****WARNING**** About to merge bins %s' % (bin_str),
                    ' If you continue you *WILL* overwrite existing bins!',
                    ' You have been shown a 3d plot of the bins to be merged.',
                    ' Continue only if you\'re sure this is what you want to do!',
                    ' y = yes, n = no, q = no and quit merging',
                    ' Merge? (%s) : ' % (vrs)])).upper()
            if(option in valid_responses):
                print('****************************************************************')
                return option
            else:
                print('Error, unrecognised choice "%s"' % option)
                minimal = True

    def promptOnSplit(self, parts, mode, minimal=False):
        '''Check that the user is ok with this split'''
        input_not_ok = True
        mode=mode.upper()
        valid_responses = ['Y','N','C','K','L','P']
        vrs = ','.join([str.lower(str(x)) for x in valid_responses])
        while(input_not_ok):
            if(minimal):
                option = input(' Split? (%s) : ' % (vrs)).upper()
            else:
                option = input('\n'.join([
                    ' ****WARNING**** About to split bin into %s parts' % (parts),
                    ' If you continue you *WILL* overwrite existing bins!',
                    ' You have been shown a 3d plot of the bin after splitting.',
                    ' Continue only if you\'re sure this is what you want to do!',
                    ' y = yes, n = no, c = redo but use coverage profile,',
                    ' k = redo but use kmer profile, l = redo but use length profile,',
                    ' p = choose new number of parts',
                    ' Split? (%s) : ' % (vrs)])).upper()
            if(option.upper() in valid_responses):
                if(
                    (option == 'K' and mode == 'KMER') or
                    (option == 'C' and mode == 'COV') or
                    (option == 'L' and mode == 'LEN')):
                    print('Error, you are already using that profile to split!')
                    minimal=True
                else:
                    print('****************************************************************')
                    return option.upper()
            else:
                print('Error, unrecognised choice "%s"' % option)
                minimal = True

    def promptOnDelete(self, bids, minimal=False):
        '''Check that the user is ok with this split'''
        input_not_ok = True
        valid_responses = ['Y','N']
        vrs = ','.join([str.lower(str(x)) for x in valid_responses])
        bids_str = ','.join([str.lower(str(x)) for x in bids])
        while(input_not_ok):
            if(minimal):
                option = input(' Delete? (%s) : ' % (vrs)).upper()
            else:
                option = input('\n'.join([
                    ' ****WARNING**** About to delete bin(s):',
                    ' %s' % (bids_str),
                    ' If you continue you *WILL* overwrite existing bins!',
                    ' Continue only if you\'re sure this is what you want to do!',
                    ' y = yes, n = no',
                    ' Delete? (%s) : ' % (vrs)])).upper()
            if(option.upper() in valid_responses):
                print('****************************************************************')
                return option.upper()
            else:
                print('Error, unrecognised choice "%s"' % option)
                minimal = True

#------------------------------------------------------------------------------
# BIN STATS
    def findCoreCentres(self, gc_range=None, getKVals=False, processChimeric=False):
        '''Find the point representing the centre of each core'''
        bin_centroid_points = np.array([])
        bin_centroid_colors = np.array([])
        bin_centroid_kvals = np.array([])
        bin_centroid_gc = np.array([])
        bids = np.array([])

        if gc_range is not None:
            # we only want to plot a subset of these guys
            gc_low = gc_range[0]
            gc_high = gc_range[1]
        num_added = 0
        for bid in self.getBids():
            if not processChimeric and self.PM.isLikelyChimeric[bid]:
                continue

            add_bin = True
            if gc_range is not None:
                avg_gc = np.median([self.PM.contigGCs[row_index] for row_index in self.bins[bid].rowIndices])
                if avg_gc < gc_low or avg_gc > gc_high:
                    add_bin = False
            if add_bin:
                bin_centroid_points = np.append(bin_centroid_points,
                                                self.bins[bid].covMedians)

                bin_centroid_colors = np.append(bin_centroid_colors, self.PM.colorMapGC(np.median(self.PM.contigGCs[self.bins[bid].rowIndices])))

                bin_centroid_gc = np.append(bin_centroid_gc, np.median(self.PM.contigGCs[self.bins[bid].rowIndices]))

                if getKVals:
                    bin_centroid_kvals = np.append(bin_centroid_kvals,
                                                   np.median([
                                                            self.PM.kmerNormSVD1[row_index] for row_index in
                                                            self.bins[bid].rowIndices
                                                            ],
                                                           axis=0)
                                                   )

                bids = np.append(bids, bid)
                num_added += 1

        if num_added != 0:
            bin_centroid_points = np.reshape(bin_centroid_points, (num_added, 3))
            bin_centroid_colors = np.reshape(bin_centroid_colors, (num_added, 4))

        if getKVals:
            return (bin_centroid_points, bin_centroid_colors, bin_centroid_gc, bin_centroid_kvals, bids)

        return (bin_centroid_points, bin_centroid_colors, bin_centroid_gc, bids)

    def getAngleBetween(self, rowIndex1, rowIndex2, ):
        '''Find the angle between two contig's coverage vectors'''
        u = self.PM.covProfiles[rowIndex1]
        v = self.PM.covProfiles[rowIndex2]
        try:
            ac = np.arccos(np.dot(u,v)/self.PM.normCoverages[rowIndex1]/self.PM.normCoverages[rowIndex1])
        except FloatingPointError:
            return 0
        return ac

    def scoreContig(self, rowIndex, bid):
        '''Determine how well a particular contig fits with a bin'''
        return self.getBin(bid).scoreProfile(self.PM.kmerNormSVD1[rowIndex], self.PM.transformedCP[rowIndex])

    def measureBinVariance(self, mode='kmer', makeKillList=False, tolerance=1.0):
        '''Get the stats on M's across all bins

        If specified, will return a list of all bins which
        fall outside of the average M profile
        '''
        Ms = {}
        Ss = {}
        Rs = {}
        for bid in self.getBids():
            if(mode == 'kmer'):
                (Ms[bid], Ss[bid], Rs[bid]) = self.bins[bid].getInnerVariance(
                    self.PM.kmerNormSVD1)
            elif(mode == 'cov'):
                (Ms[bid], Ss[bid], Rs[bid]) = self.bins[bid].getInnerVariance(
                    self.PM.transformedCP,
                    mode='cov')

        # find the mean and stdev
        mv = np.array(list(Ms.values()))
        if(not makeKillList):
            sv = np.array(list(Ss.values()))
            return np.mean(mv), np.std(mv), np.median(sv), np.std(sv)

        else:
            cutoff = np.mean(mv) + tolerance * np.std(mv)
            kill_list = []
            for bid in Ms:
                if(Ms[bid] > cutoff):
                    kill_list.append(bid)
            return (kill_list, cutoff)

    def analyseBinKVariance(self, outlierTrim=0.1, plot=False):
        '''Measure within and between bin variance of kmer sigs

        return a list of potentially confounding kmer indices
        '''

        L.info('Measuring kmer type variances')
        means = np.array([])
        stdevs = np.array([])
        bids = np.array([])

        # work out the mean and stdev for the kmer sigs for each bin
        for bid in self.getBids():
            bkworking = np.array([])
            for row_index in self.bins[bid].rowIndices:
                bkworking = np.append(bkworking, self.PM.kmerSigs[row_index])
            bkworking = np.reshape(bkworking, (self.bins[bid].binSize, np.size(self.PM.kmerSigs[0])))
            bids = np.append(bids, [bid])
            means = np.append(means, np.mean(bkworking, axis=0))
            stdevs = np.append(stdevs, np.std(bkworking, axis=0))

        means = np.reshape(means, (len(self.bins), np.size(self.PM.kmerSigs[0])))
        stdevs = np.reshape(stdevs, (len(self.bins), np.size(self.PM.kmerSigs[0])))

        # now work out the between and within core variances
        between = np.std(means, axis=0)
        within = np.median(stdevs, axis=0)

        B = np.arange(0, np.size(self.PM.kmerSigs[0]), 1)
        names = self.PM.getMerColNames().split(',')

        # we'd like to find the indices of the worst 10% for each type so we can ignore them
        # specifically, we'd like to remove the least variable between core kms and the
        # most variable within core kms.
        sort_between_indices = np.argsort(between)
        sort_within_indices = np.argsort(within)[::-1]
        number_to_trim = int(outlierTrim* float(np.size(self.PM.kmerSigs[0])))

        return_indices =[]
        for i in range(0,number_to_trim):
            if(sort_between_indices[i] not in return_indices):
                return_indices.append(sort_between_indices[i])
            if(sort_within_indices[i] not in return_indices):
                return_indices.append(sort_within_indices[i])

        if(plot):
            L.info('BETWEEN')
            for i in range(0,number_to_trim):
                L.info(names[sort_between_indices[i]])
            L.info('WITHIN')
            for i in range(0,number_to_trim):
                L.info(names[sort_within_indices[i]])

            plt.figure(1)
            plt.subplot(211)
            plt.plot(B, between, 'r--', B, within, 'b--')
            plt.xticks(B, names, rotation=90)
            plt.grid()
            plt.subplot(212)
            ratio = between/within
            plt.plot(B, ratio, 'r--')
            plt.xticks(B, names, rotation=90)
            plt.grid()
            plt.show()

        return return_indices

#------------------------------------------------------------------------------
# IO and IMAGE RENDERING

    def printBins(self, outFormat, fileName=''):
        '''Wrapper for print (handles piping to file or stdout)'''
        if('' != fileName):
            try:
                # redirect stdout to a file
                stdout = open(fileName, 'w')
                self.printInner(outFormat, stdout)
            except:
                L.error('Error diverting stout to file: %s %s' % (fileName, exc_info()[0]))
                raise
        else:
            self.printInner(outFormat)

    def printInner(self, outFormat, stream=sys_stdout):
        # handle the headers first
        separator = '\t'
        if(outFormat == 'contigs'):
            stream.write('%s\n' % separator.join([
                '"#bid"',
                '"cid"',
                '"length"',
                '"GC"']))

        elif(outFormat == 'bins'):
            header = [
                '"bin id"',
                '"Likely chimeric"',
                '"length (bp)"',
                '"# seqs"',
                '"GC mean"',
                '"GC std"']

            for i in range(0, len(self.PM.covProfiles[0])):
                header.append('"Coverage %s mean"' % (i+1))
                header.append('"Coverage %s std"' % (i+1))

            stream.write(separator.join(header))

        elif(outFormat == 'full'):
            pass

        else:
            L.error('Error: Unrecognised format: %s' % (outFormat))
            return

        for bid in self.getBids():
            self.bins[bid].makeBinDist(
                self.PM.transformedCP,
                self.PM.averageCoverages,
                self.PM.kmerNormSVD1,
                self.PM.kmerSVDs,
                self.PM.contigGCs,
                self.PM.contigLengths)

            self.bins[bid].printBin(
                self.PM.contigNames,
                self.PM.covProfiles,
                self.PM.contigGCs,
                self.PM.contigLengths,
                self.PM.isLikelyChimeric,
                outFormat=outFormat,
                separator=separator,
                stream=stream)

    def plotProfileDistributions(self):
        '''Plot the coverage and kmer distributions for each bin'''
        for bid in self.getBids():
            self.bins[bid].plotProfileDistributions(
                self.PM.transformedCP,
                self.PM.kmerSigs,
                fileName='PROFILE_%s' % (bid))

    def plotSelectBins(self,
                       bids,
                       plotMers=False,
                       fileName='',
                       plotEllipsoid=False,
                       ignoreContigLengths=False,
                       ET=None):
        '''Plot a selection of bids in a window'''
        if plotEllipsoid and ET == None:
            ET = EllipsoidTool()

        # we need to do some fancy-schmancy stuff at times!
        if plotMers:
            num_cols = 2
        else:
            num_cols = 1

        fig = plt.figure(figsize=(6.5*num_cols, 6.5))
        ax = fig.add_subplot(1,num_cols,1, projection='3d')
        for i, bid in enumerate(bids):
            self.bins[bid].plotOnAx(
                ax,
                self.PM.transformedCP,
                self.PM.contigGCs,
                self.PM.contigLengths,
                self.PM.colorMapGC,
                self.PM.isLikelyChimeric,
                ET=ET,
                printID=True,
                ignoreContigLengths=ignoreContigLengths,
                plotColorbar=(num_cols==1 and i==0))

        if plotMers:
            title_parts = [
                'Bin: %d : %d contigs : %s BP' % (
                    bid,
                    len(self.bins[bid].rowIndices),
                    np.sum(self.PM.contigLengths[self.bins[bid].rowIndices])),
                'Coverage centroid: %d %d %d' % (
                    np.median(self.PM.transformedCP[self.bins[bid].rowIndices,0]),
                    np.median(self.PM.transformedCP[self.bins[bid].rowIndices,1]),
                    np.median(self.PM.transformedCP[self.bins[bid].rowIndices,2])),
                'GC: median: %.4f stdev: %.4f' % (
                    np.median(self.PM.contigGCs[self.bins[bid].rowIndices]),
                    np.std(self.PM.contigGCs[self.bins[bid].rowIndices]))]

            if self.PM.isLikelyChimeric[bid]:
                title_parts.append('Likely Chimeric')

            ax.set_title('\n'.join(title))

            ax = fig.add_subplot(1, 2, 2)
            for i, bid in enumerate(bids):
                self.bins[bid].plotMersOnAx(
                ax,
                self.PM.kmerSVDs[:,0],
                self.PM.kmerSVDs[:,1],
                self.PM.contigGCs,
                self.PM.contigLengths,
                self.PM.colorMapGC,
                ET=ET,
                printID=True,
                plotColorbar=(i==0))

            ax.set_title('PCA of k-mer signature')

        fig.set_size_inches(6*num_cols, 6)
        if(fileName != ''):
            try:
                plt.savefig(fileName,dpi=300)
            except:
                L.error('Error saving image: %s %s' % (fileName, exc_info()[0]))
                raise
        else:
            try:
                plt.show()
            except:
                L.error('Error showing image: %s' % (exc_info()[0]))
                raise

        plt.close(fig)
        del fig

    def plotMultipleBins(self,
        bins,
        untransformed=False,
        semi_untransformed=False,
        ignoreContigLengths=False,
        squash=False,
        file_name='debug.png'):
        '''plot a bunch of bins, used mainly for debugging'''

        ET = EllipsoidTool()
        fig = plt.figure()

        if untransformed or semi_untransformed:
            coords = self.PM.covProfiles
            et = None
            pc = False
        else:
            coords = self.PM.transformedCP
            et = ET
            pc = True

        if squash:
            # mix all the points together
            # we need to work out how to shape the plots
            num_plots = len(bins)
            plot_rows = float(int(np.sqrt(num_plots)))
            plot_cols = np.ceil(float(num_plots)/plot_rows)
            plot_num = 1
            for bids in bins:
                ax = fig.add_subplot(plot_rows, plot_cols, plot_num, projection='3d')
                disp_vals = np.array([])
                disp_lens = np.array([])
                gcs = np.array([])
                num_points = 0
                for bid in bids:
                    for row_index in self.bins[bid].rowIndices:
                        num_points += 1
                        disp_vals = np.append(disp_vals, coords[row_index])
                        disp_lens = np.append(disp_lens, np.sqrt(self.PM.contigLengths[row_index]))
                        gcs = np.append(gcs, self.PM.contigGCs[row_index])

                # reshape
                disp_vals = np.reshape(disp_vals, (num_points, 3))

                if ignoreContigLengths:
                    sc = ax.scatter(disp_vals[:,0], disp_vals[:,1], disp_vals[:,2], edgecolors='none', c=gcs, cmap=self.PM.colorMapGC, vmin=0.0, vmax=1.0, s=10., marker='.')
                else:
                    sc = ax.scatter(disp_vals[:,0], disp_vals[:,1], disp_vals[:,2], edgecolors='k', c=gcs, cmap=self.PM.colorMapGC, vmin=0.0, vmax=1.0, s=disp_lens, marker='.')
                sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

                plot_num += 1

            special = False
            if special:
                # make a plot for a background etc
                ax.azim = -127
                ax.elev = 4

                # strip all background
                ax.set_xlim3d(0,self.PM.scaleFactor)
                ax.set_ylim3d(0,self.PM.scaleFactor)
                ax.set_zlim3d(0,self.PM.scaleFactor)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                plt.tight_layout()
                ax.set_axis_off()

                fig.set_size_inches(10,10)
                plt.savefig(
                    file_name,
                    dpi=300,
                    format='png')

            else:
                cbar = plt.colorbar(sc, shrink=0.5)
                cbar.ax.tick_params()
                cbar.ax.set_title('% GC', size=10)
                cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                cbar.ax.set_ylim([0.15, 0.85])
                mungeCbar(cbar)
        else:
            # plot all separately
            # we need to work out how to shape the plots
            num_plots = len(bins) + 1
            plot_rows = float(int(np.sqrt(num_plots)))
            plot_cols = np.ceil(float(num_plots)/plot_rows)
            plot_num = 1
            ax = fig.add_subplot(plot_rows, plot_cols, plot_num, projection='3d')

            for bids in bins:
                for bid in bids:
                    self.bins[bid].plotOnAx(ax, coords, self.PM.contigGCs, self.PM.contigLengths, self.PM.colorMapGC, self.PM.isLikelyChimeric, ignoreContigLengths=ignoreContigLengths, ET=et, plotCentroid=pc)

            plot_num += 1
            if semi_untransformed:
                coords = self.PM.transformedCP
                et = ET
                pc = True

            for bids in bins:
                ax = fig.add_subplot(plot_rows, plot_cols, plot_num, projection='3d')
                plot_num += 1
                for bid in bids:
                    self.bins[bid].plotOnAx(ax, coords, self.PM.contigGCs, self.PM.contigLengths, self.PM.colorMapGC, self.PM.isLikelyChimeric, ignoreContigLengths=ignoreContigLengths, ET=et, plotCentroid=pc)

        try:
            plt.show()
        except:
            L.error('Error showing image: %s' % exc_info()[0])
            raise

        plt.close(fig)
        del fig

    def plotBins(
        self,
        bids=None,
        FNPrefix='BIN',
        sideBySide=False,
        folder='',
        plotEllipsoid=False,
        ignoreContigLengths=False,
        ET=None,
        axes=None):
        '''Make plots of all the bins'''
        if plotEllipsoid and ET == None:
            ET = EllipsoidTool()

        if folder != '':
            makeSurePathExists(folder)

        if bids is None:
            bids = self.getBids()

        for bid in bids:
            self.bins[bid].makeBinDist(
                self.PM.transformedCP,
                self.PM.averageCoverages,
                self.PM.kmerNormSVD1,
                self.PM.kmerSVDs,
                self.PM.contigGCs,
                self.PM.contigLengths)

        if(sideBySide):
            L.info('Plotting side by side')
            self.plotSideBySide(bids, tag=FNPrefix, ignoreContigLengths=ignoreContigLengths)
        else:
            L.info('Plotting bins')
            for bid in bids:
                file_name = '%s_%s' % (FNPrefix, bid)
                if folder != '':
                    file_name = osp_join(folder, file_name)

                self.bins[bid].plotBin(
                    self.PM.transformedCP,
                    self.PM.contigGCs,
                    self.PM.kmerNormSVD1,
                    self.PM.contigLengths,
                    self.PM.colorMapGC,
                    self.PM.isLikelyChimeric,
                    fileName=file_name,
                    ignoreContigLengths=ignoreContigLengths,
                    extents=[0,1,0,1,0,1],
                    ET=ET,
                    axes=axes)

    def plotBinCoverage(self, plotEllipses=False, plotContigLengs=False, printID=False):
        '''Make plots of all the bins'''

        L.info('Plotting first 3 stoits in untransformed coverage space')

        # plot contigs in coverage space
        fig = plt.figure()

        if plotContigLengs:
            disp_lens = np.sqrt(self.PM.contigLengths)
        else:
            disp_lens = 30

        # plot contigs in kmer space
        ax = fig.add_subplot(121, projection='3d')
        ax.set_xlabel('kmer PC1')
        ax.set_ylabel('kmer PC2')
        ax.set_zlabel('kmer PC3')
        ax.set_title('kmer space')

        sc = ax.scatter(
            self.PM.kmerSVDs[:,0],
            self.PM.kmerSVDs[:,1],
            self.PM.kmerSVDs[:,2],
            edgecolors='k',
            c=self.PM.contigGCs,
            cmap=self.PM.colorMapGC,
            vmin=0.0,
            vmax=1.0,
            s=disp_lens)

        sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

        if plotEllipses:
            ET = EllipsoidTool()
            for bid in self.getBids():
                row_indices = self.bins[bid].rowIndices
                (center, radii, rotation) = self.bins[bid].getBoundingEllipsoid(self.PM.kmerSVDs[:, 0:3], ET=ET)
                centroid_gc = np.mean(self.PM.contigGCs[row_indices])
                centroid_color = self.PM.colorMapGC(centroid_gc)
                if printID:
                    label=self.id
                else:
                    label=None
                ET.plotEllipsoid(
                    center,
                    radii,
                    rotation,
                    ax=ax,
                    plotAxes=False,
                    cageColor=centroid_color,
                    label=label)

        # plot contigs in untransformed coverage space
        ax = fig.add_subplot(122, projection='3d')
        ax.set_xlabel('coverage 1')
        ax.set_ylabel('coverage 2')
        ax.set_zlabel('coverage 3')
        ax.set_title('coverage space')

        sc = ax.scatter(
            self.PM.covProfiles[:,0],
            self.PM.covProfiles[:,1],
            self.PM.covProfiles[:,2],
            edgecolors='k',
            c=self.PM.contigGCs,
            cmap=self.PM.colorMapGC,
            vmin=0.0,
            vmax=1.0,
            s=disp_lens)

        sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

        cbar = plt.colorbar(sc, shrink=0.5)
        cbar.ax.tick_params()
        cbar.ax.set_title('% GC', size=10)
        cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        cbar.ax.set_ylim([0.15, 0.85])
        mungeCbar(cbar)

        if plotEllipses:
            ET = EllipsoidTool()
            for bid in self.getBids():
                row_indices = self.bins[bid].rowIndices
                (center, radii, rotation) = self.bins[bid].getBoundingEllipsoid(self.PM.covProfiles[:, 0:3], ET=ET)
                centroid_gc = np.mean(self.PM.contigGCs[row_indices])
                centroid_color = self.PM.colorMapGC(centroid_gc)
                if printID:
                    label=self.id
                else:
                    label=None
                ET.plotEllipsoid(center, radii, rotation, ax=ax, plotAxes=False, cageColor=centroid_color, label=label)

        try:
            plt.show()
            plt.close(fig)
        except:
            L.error('Error showing image: %s' % (exc_info()[0]))
            raise

        del fig

    def plotSideBySide(self, bids, fileName='', tag='', use_elipses=True, ignoreContigLengths=False):
        '''Plot two bins side by side in 3d'''
        if use_elipses:
            ET = EllipsoidTool()
        else:
            ET = None
        fig = plt.figure()

        # get plot extents
        xMin = 1e6
        xMax = 0
        yMin = 1e6
        yMax = 0
        zMin = 1e6
        zMax = 0

        for bid in bids:
            x = self.PM.transformedCP[self.bins[bid].rowIndices,0]
            y = self.PM.transformedCP[self.bins[bid].rowIndices,1]
            z = self.PM.transformedCP[self.bins[bid].rowIndices,2]

            xMin = min(min(x), xMin)
            xMax = max(max(x), xMax)

            yMin = min(min(y), yMin)
            yMax = max(max(y), yMax)

            zMin = min(min(z), zMin)
            zMax = max(max(z), zMax)

        # we need to work out how to shape the plots
        num_plots = len(bids)
        plot_rows = float(int(np.sqrt(num_plots)))
        plot_cols = np.ceil(float(num_plots)/plot_rows)
        for plot_num, bid in enumerate(bids):
            title = self.bins[bid].plotOnFig(fig, plot_rows, plot_cols, plot_num+1,
                                              self.PM.transformedCP, self.PM.contigGCs, self.PM.contigLengths,
                                              self.PM.colorMapGC, self.PM.isLikelyChimeric, ET=ET, fileName=fileName,
                                              plotColorbar=(plot_num == len(bids)-1), extents=[xMin, xMax, yMin, yMax, zMin, zMax],
                                              ignoreContigLengths=ignoreContigLengths)

            plt.title(title)
        if(fileName != ''):
            try:
                fig.set_size_inches(12,6)
                plt.savefig(fileName,dpi=300)
            except:
                L.error('Error saving image: %s %s' % (fileName, exc_info()[0]))
                raise
        else:
            try:
                plt.show()
            except:
                L.error('Error showing image: %s' % (exc_info()[0]))
                raise
        plt.close(fig)
        del fig

    def plotBinIds(self, gc_range=None, ignoreRanges=False, showChimeric=False):
        '''Render 3d image of core ids'''
        (bin_centroid_points, bin_centroid_colors, _bin_centroid_gc, bids) = self.findCoreCentres(
            gc_range=gc_range,
            processChimeric=showChimeric)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x coverage')
        ax.set_ylabel('y coverage')
        ax.set_zlabel('z coverage')

        outer_index = 0
        for bid in bids:
            ax.text(bin_centroid_points[outer_index,0],
                    bin_centroid_points[outer_index,1],
                    bin_centroid_points[outer_index,2],
                    str(int(bid)),
                    color=bin_centroid_colors[outer_index]
                    )
            outer_index += 1

        if ignoreRanges:
            mm = np.max(bin_centroid_points, axis=0)
            ax.set_xlim3d(0, mm[0])
            ax.set_ylim3d(0, mm[1])
            ax.set_zlim3d(0, mm[2])

        else:
            ax.set_xlim3d(0, 1000)
            ax.set_ylim3d(0, 1000)
            ax.set_zlim3d(0, 1000)
        try:
            plt.show()
            plt.close(fig)
        except:
            L.error('Error showing image: %s' % (exc_info()[0]))
            raise
        del fig

    def plotBinPoints(self, ignoreRanges=False, plotColorbar=True, showChimeric=False):
        '''Render the image for validating cores'''
        (bin_centroid_points, _bin_centroid_colors, bin_centroid_gc, _bids) = self.findCoreCentres(processChimeric=showChimeric)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(bin_centroid_points[:,0], bin_centroid_points[:,1], bin_centroid_points[:,2], edgecolors='k', c=bin_centroid_gc, cmap=self.PM.colorMapGC, vmin=0.0, vmax=1.0)
        sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

        if plotColorbar:
            cbar = plt.colorbar(sc, shrink=0.5)
            cbar.ax.tick_params()
            cbar.ax.set_title('% GC', size=10)
            cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            cbar.ax.set_ylim([0.15, 0.85])
            mungeCbar(cbar)

        ax.set_xlabel('x coverage')
        ax.set_ylabel('y coverage')
        ax.set_zlabel('z coverage')

        if not ignoreRanges:
            ax.set_xlim3d(0, 1000)
            ax.set_ylim3d(0, 1000)
            ax.set_zlim3d(0, 1000)
        try:
            plt.show()
            plt.close(fig)
        except:
            L.error('Error showing image: %s' % (exc_info()[0]))
            raise
        del fig

    def plotTDist(self, scores, testPoint):
        '''DEBUG: Plot the distribution of the NULL T disytribution'''
        B = np.arange(0, len(scores), 1)
        co_90 = [int(float(len(scores))*0.90)] * 100
        co_95 = [int(float(len(scores))*0.95)] * 100
        co_97 = [int(float(len(scores))*0.97)] * 100
        co_99 = [int(float(len(scores))*0.99)] * 100
        smin = np.min(scores)
        smax = np.max(scores)
        step = (smax-smin)/100
        LL = np.arange(smin, smax, step)

        fig = plt.figure()

        plt.plot(co_90, LL, 'g-')
        plt.plot(co_95, LL, 'g-')
        plt.plot(co_97, LL, 'g-')
        plt.plot(co_99, LL, 'g-')

        plt.plot(B, scores, 'r-')
        plt.plot(B, [testPoint]*len(scores), 'b-')
        plt.show()
        plt.close(fig)
        del fig

###############################################################################
###############################################################################
###############################################################################
###############################################################################
