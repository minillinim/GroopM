#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    groopm.py                                                                #
#                                                                             #
#    Wraps coarse workflows                                                   #
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
__version__ = "0.2.11"
__maintainer__ = "Michael Imelfort"
__email__ = "michael.imelfort@gmail.com"
__status__ = "Released"

###############################################################################

import matplotlib as mpl
import sys

# GroopM imports
from . import mstore
from . import cluster
from . import refine
from . import binManager
from . import groopmUtils
from . import groopmTimekeeper as gtime
from .groopmExceptions import ExtractModeNotAppropriateException
from .mstore import GMDataManager
from .profileManager import ProfileManager
from .paraAxes import ParaAxes

###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Track rogue print statements
#from groopmExceptions import Tracer
#import sys
#sys.stdout = Tracer(sys.stdout)
#sys.stderr = Tracer(sys.stderr)
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class GroopMOptionsParser():
    def __init__(self, version):
        # set default value for matplotlib
        mpl.rcParams['lines.linewidth'] = 1

        mpl.rcParams['xtick.labelsize'] = 8
        mpl.rcParams['ytick.labelsize'] = 8
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['axes.titlesize'] = 12
        mpl.rcParams['axes.linewidth'] = 0.25

        mpl.rcParams['axes3d.grid'] = True

        mpl.rcParams['savefig.dpi'] = 300

        mpl.rcParams['figure.figsize'] = [6.5, 6.5]
        mpl.rcParams['figure.facecolor'] = '1.0'

        self.GMVersion = version

    def parseOptions(self, options ):
        timer = gtime.TimeKeeper()
        if(options.subparser_name == 'parse'):
            # parse raw input
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in data parsing mode..." % self.GMVersion)
            print("*******************************************************************************")

            GMdata = mstore.GMDataManager()

            # check this here:
            if not options.covfile:
                print("Sorry, You must supply a pre-parsed coverages file to use GroopM.\n Exiting...")
                return

            success = GMdata.createDB(
                options.covfile,
                options.reference,
                options.dbname,
                options.cutoff,
                timer,
                bins_file=options.bins,
                force=options.force,
                threads=options.threads)

            if not success:
                print(options.dbname,"not updated")


        elif(options.subparser_name == 'core'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in core creation mode..." % self.GMVersion)
            print("*******************************************************************************")

            pm = ProfileManager(options.dbname)
            pm.loadData(
                timer,
                "(length >= 0) ",
                loadRawKmers=False,
                makeColors=True,
                loadContigLengths=True,
                loadContigNames=False,
                loadContigGCs=True)

            print("    %s" % timer.getTimeStamp())
            sys.stdout.flush()

            BM = binManager.BinManager(pm=pm)
            BM.setColorMap('HSV')

            AXC = ParaAxes(pm)
            AXC.density_cluster(
                BM,
                window=options.window_size,
                tolerance=options.tolerance,
                plot_bins=options.plot_bins,
                plot_journey=options.plot_journey,
                limit=options.limit)

        elif(options.subparser_name == 'refine'):
            # refine bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in core refining mode..." % self.GMVersion)
            print("*******************************************************************************")
            bids = []
            #if options.bids is not None:
            #    bids = options.bids
            auto = options.auto
            transform=True^options.no_transform

            RE = refine.RefineEngine(timer,
                                     dbFileName=options.dbname,
                                     transform=transform,
                                     bids=bids,
                                     loadContigNames=True)

            if options.plot:
                pfx="REFINED"
            else:
                pfx=""
            print("Refine bins")

            RE.refineBins(timer,
                          auto=auto,
                          saveBins=True,
                          plotFinal=pfx)

        elif(options.subparser_name == 'recruit'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in bin expansion mode..." % self.GMVersion)
            print("*******************************************************************************")
            RE = refine.RefineEngine(timer,
                                     dbFileName=options.dbname,
                                     getUnbinned=True,
                                     loadContigNames=False,
                                     cutOff=options.cutoff)

            RE.recruitWrapper(timer,
                              inclusivity=options.inclusivity,
                              step=options.step,
                              saveBins=True)

        elif(options.subparser_name == 'extract'):
            # Extract data
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in '%s' extraction mode..." % (self.GMVersion, options.mode))
            print("*******************************************************************************")
            bids = []
            if options.bids is not None:
                bids = options.bids
            BX = groopmUtils.GMExtractor(options.dbname,
                                          bids=bids,
                                          folder=options.out_folder
                                          )
            if(options.mode=='contigs'):
                BX.extractContigs(timer,
                                  fasta=options.data,
                                  prefix=options.prefix,
                                  cutoff=options.cutoff)

            elif(options.mode=='reads'):
                BX.extractReads(timer,
                                bams=options.data,
                                prefix=options.prefix,
                                mixBams=options.mix_bams,
                                mixGroups=options.mix_groups,
                                mixReads=options.mix_reads,
                                interleaved=options.interleave,
                                bigFile=options.no_gzip,
                                headersOnly=options.headers_only,
                                minMapQual=options.mapping_quality,
                                maxMisMatches=options.max_distance,
                                useSuppAlignments=options.use_supplementary,
                                useSecondaryAlignments=options.use_secondary,
                                verbose=options.verbose,
                                threads=options.threads)

            else:
                raise ExtractModeNotAppropriateException("mode: "+ options.mode + " is unknown")
        elif(options.subparser_name == 'merge'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in bin merging mode..." % self.GMVersion)
            print("*******************************************************************************")
            BM = binManager.BinManager(dbFileName=options.dbname)
            BM.loadBins(timer, makeBins=True, silent=False)
            BM.merge(options.bids, options.force, saveBins=True)

        elif(options.subparser_name == 'split'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in bin splitting mode..." % self.GMVersion)
            print("*******************************************************************************")
            BM = binManager.BinManager(dbFileName=options.dbname)
            BM.loadBins(timer, makeBins=True, silent=False)
            BM.split(options.bid, options.parts, mode=options.mode, saveBins=True, auto=options.force)

        elif(options.subparser_name == 'delete'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in bin deleting mode..." % self.GMVersion)
            print("*******************************************************************************")
            BM = binManager.BinManager(dbFileName=options.dbname)
            BM.loadBins(timer, makeBins=True, silent=True)#, bids=options.bids)
            BM.deleteBins(options.bids, force=options.force, saveBins=True, freeBinnedRowIndices=True)

        elif(options.subparser_name == 'plot'):
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in bin plotting mode..." % self.GMVersion)
            print("*******************************************************************************")
            BM = binManager.BinManager(dbFileName=options.dbname)

            if options.bids is None:
                bids = []
            else:
                bids = options.bids
            BM.loadBins(timer, makeBins=True, silent=False, bids=bids, loadContigNames=False)

            BM.setColorMap(options.cm)

            BM.plotBins(
                FNPrefix=options.tag,
                plotEllipsoid=True,
                ignoreContigLengths=options.points,
                folder=options.folder)

        elif(options.subparser_name == 'pplot'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in parallel plotting mode..." % (self.GMVersion))
            print("*******************************************************************************")

            pm = ProfileManager(options.dbname)
            pm.loadData(
                timer,
                "(length >= %d) " % options.cutoff,
                loadRawKmers=False,
                makeColors=False,
                loadContigNames=False,
                loadContigGCs=False)

            print("    %s" % timer.getTimeStamp())
            sys.stdout.flush()

            AXC = ParaAxes(pm)
            AXC.plot(
                num_blocks=options.blocks,
                include=options.include_ratio)

        elif(options.subparser_name == 'explore'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in '%s' explorer mode..." % (self.GMVersion, options.mode))
            print("*******************************************************************************")

            bids = []
            if options.bids is not None:
                bids = options.bids
            BE = groopmUtils.BinExplorer(options.dbname,
                                         bids=bids,
                                         cmstring=options.cm,
                                         ignoreContigLengths=options.points)
            if(options.mode == 'binpoints'):
                BE.plotPoints(timer)
            elif(options.mode == 'binids'):
                BE.plotIds(timer, ignoreRanges=True)
            elif(options.mode == 'allcontigs'):
                BE.plotContigs(timer, coreCut=options.cutoff, all=True)
            elif(options.mode == 'unbinnedcontigs'):
                BE.plotUnbinned(timer, coreCut=options.cutoff)
            elif(options.mode == 'binnedcontigs'):
                BE.plotContigs(timer, coreCut=options.cutoff)
            elif(options.mode == 'binassignments'):
                BE.plotBinAssignents(timer, coreCut=options.cutoff)
            elif(options.mode == 'compare'):
                BE.plotCompare(timer, coreCut=options.cutoff)
            elif (options.mode == 'together'):
                BE.plotTogether(timer, coreCut=options.cutoff, doMers=options.kmers)
            elif (options.mode == 'sidebyside'):
                BE.plotSideBySide(timer, coreCut=options.cutoff)
            else:
                print("**Error: unknown mode:",options.mode)

        elif(options.subparser_name == 'flyover'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Making a flyover..." % self.GMVersion)
            print("*******************************************************************************")
            bids = []
            if options.bids is not None:
                bids = options.bids
            BE = groopmUtils.BinExplorer(options.dbname,
                                         bids=bids,
                                         ignoreContigLengths=options.points)
            BE.plotFlyOver(timer,
                           fps=options.fps,
                           totalTime=options.totalTime,
                           percentFade=options.firstFade,
                           prefix=options.prefix,
                           showColorbar=options.colorbar,
                           title=options.title,
                           coreCut=options.cutoff,
                           format=options.format)

        elif(options.subparser_name == 'highlight'):
            # make bin cores
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in highlighter mode..." % self.GMVersion)
            print("*******************************************************************************")
            bids = []
            if options.bids is not None:
                bids = options.bids
            BE = groopmUtils.BinExplorer(options.dbname,
                                         bids=bids,
                                         binLabelsFile = options.binlabels,
                                         contigColorsFile = options.contigcolors,
                                         ignoreContigLengths=options.points)
            BE.plotHighlights(timer,
                              options.elevation,
                              options.azimuth,
                              options.file,
                              options.filetype,
                              options.dpi,
                              drawRadius=options.radius,
                              show=options.show,
                              coreCut=options.cutoff,
                              testing=options.place
                              )

        elif(options.subparser_name == 'print'):
            BM = binManager.BinManager(dbFileName=options.dbname)
            bids = []
            if options.bids is not None:
                bids = options.bids
            BM.loadBins(timer, getUnbinned=options.unbinned, makeBins=True, silent=True, bids=bids)
            BM.printBins(options.format, fileName=options.outfile)

        elif(options.subparser_name == 'dump'):
            print("*******************************************************************************")
            print(" [[GroopM %s]] Running in data dumping mode..." % self.GMVersion)
            print("*******************************************************************************")

            # prep fields. Do this first cause users are mot likely to
            # mess this part up!
            allowable_fields = ['names', 'mers', 'svds', 'gc', 'coverage', 'tcoverage', 'ncoverage', 'lengths', 'bins', 'all']
            fields = options.fields.split(',')
            for field in fields:
                if field not in allowable_fields:
                    print("ERROR: field '%s' not recognised. Allowable fields are:" % field)
                    print('\t',",".join(allowable_fields))
                    return
            if options.separator == '\\t':
                separator = '\t'
            else:
                separator = options.separator

            DM = GMDataManager()
            DM.dumpData(options.dbname,
                        fields,
                        options.outfile,
                        separator,
                        not options.no_headers)

        return 0


###############################################################################
###############################################################################
###############################################################################
###############################################################################
