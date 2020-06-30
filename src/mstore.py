#!/usr/bin/env python3
###############################################################################
#                                                                             #
#    mstore.py                                                                #
#                                                                             #
#    GroopM - Low level data management and file parsing                      #
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

__current_GMDB_version__ = 6

###############################################################################

from sys import exc_info

import pandas as pd
import tables
import numpy as np
np.seterr(all='raise')
from scipy.spatial.distance import cdist, squareform

# GroopM imports
from .ksig import KmerSigEngine


class GMDataManager:
    """Top level class for manipulating GroopM data

    Use this class for parsing in raw data into a hdf DB and
    for reading from and updating same DB

    NOTE: All tables are kept in the same order indexed by the contig ID
    Tables managed by this class are listed below

    ------------------------
     PROFILES
    group = '/profile'
    ------------------------
    **Kmer Signature**
    table = 'kms'
    'mer1' : tables.FloatCol(pos=0)
    'mer2' : tables.FloatCol(pos=1)
    'mer3' : tables.FloatCol(pos=2)
    ...

    **Coverage profile**
    table = 'coverage'
    'stoit1' : tables.FloatCol(pos=0)
    'stoit2' : tables.FloatCol(pos=1)
    'stoit3' : tables.FloatCol(pos=2)
    ...

    **Transformed coverage profile**
    table = 'transCoverage'
    'x' : tables.FloatCol(pos=0)
    'y' : tables.FloatCol(pos=1)
    'z' : tables.FloatCol(pos=2)

    **Normalised coverage profile**
    table = 'normCoverage'
    'normCov' : tables.FloatCol(pos=0)

    ------------------------
     LINKS
    group = '/links'
    ------------------------
    ** Links **
    table = 'links'
    'contig1'    : tables.Int32Col(pos=0)            # reference to index in meta/contigs
    'contig2'    : tables.Int32Col(pos=1)            # reference to index in meta/contigs
    'numReads'   : tables.Int32Col(pos=2)            # number of reads supporting this link
    'linkType'   : tables.Int32Col(pos=3)            # the type of the link (SS, SE, ES, EE)
    'gap'        : tables.Int32Col(pos=4)            # the estimated gap between the contigs

    ------------------------
     METADATA
    group = '/meta'
    ------------------------
    ** Metadata **
    table = 'meta'
    'stoit_col_names' : tables.StringCol(512, pos=0)
    'numStoits'     : tables.Int32Col(pos=1)
    'merColNames'   : tables.StringCol(4096,pos=2)
    'merSize'       : tables.Int32Col(pos=3)
    'numMers'       : tables.Int32Col(pos=4)
    'numCons'       : tables.Int32Col(pos=5)
    'numBins'       : tables.Int32Col(pos=6)
    'clustered'     : tables.BoolCol(pos=7)           # set to true after clustering is complete
    'complete'      : tables.BoolCol(pos=8)           # set to true after clustering finishing is complete
    'formatVersion' : tables.Int32Col(pos=9)          # groopm file version

    ** Contigs **
    table = 'contigs'
    'cid'    : tables.StringCol(512, pos=0)
    'bid'    : tables.Int32Col(pos=1)
    'length' : tables.Int32Col(pos=2)
    'gc'     : tables.FloatCol(pos=3)

    ** Bins **
    table = 'bins'
    'bid'        : tables.Int32Col(pos=0)
    'numMembers' : tables.Int32Col(pos=1)
    'isLikelyChimeric' : tables.BoolCol(pos=2)

    **Transformed coverage corners**
    table = 'transCoverageCorners'
    'x' : tables.FloatCol(pos=0)
    'y' : tables.FloatCol(pos=1)
    'z' : tables.FloatCol(pos=2)

    """
    def __init__(self): pass

#------------------------------------------------------------------------------
# DB CREATION / INITIALISATION  - PROFILES

    def createDB(self,
        coverage_file,
        contigs,
        db_file_name,
        cutoff,
        timer,
        kmer_size=4,
        force=False,
        threads=1):
        '''Main wrapper for parsing all input files'''
        # load all the passed vars
        db_file_name = db_file_name
        contigs_file = contigs
        stoit_col_names = []

        kse = KmerSigEngine(kmer_size)
        contig_parser = ContigParser()

        # make sure we're only overwriting existing DBs with the users consent
        try:
            with open(db_file_name) as f:
                if(not force):
                    user_option = self.promptOnOverwrite(db_file_name)
                    if(user_option != 'Y'):
                        print('Operation cancelled')
                        return False
                    else:
                        print('Overwriting database',db_file_name)
        except IOError as e:
            print('Creating new database', db_file_name)

        # create the db
        try:
            with tables.open_file(db_file_name, mode='w', title='GroopM') as h5file:
                # Create groups under "/" (root) for storing profile information and metadata
                profile_group = h5file.create_group('/', 'profile', 'Assembly profiles')
                meta_group = h5file.create_group('/', 'meta', 'Associated metadata')
                links_group = h5file.create_group('/', 'links', 'Paired read link information')

                #------------------------
                # parse contigs
                #
                # Contig IDs are key. Any keys existing in other files but not in this file will be
                # ignored. Any missing keys in other files will be given the default profile value
                # (typically 0). Ironically, we don't store the CIDs here, these are saved one time
                # only in the bin table
                #
                # Before writing to the database we need to make sure that none of them have
                # 0 coverage @ all stoits.
                #------------------------

                # load coverages into a dataframe
                coverages = pd.read_csv(
                    coverage_file,
                    compression='gzip',
                    index_col='contig',
                    sep='\t')
                coverages = coverages.drop(['Length'], axis=1)

                cnames = list(coverages.index)

                stoit_col_names = np.array([
                    name.replace('.bam', '').replace('.', '_') for name in coverages.columns])

                coverages_df = pd.DataFrame(
                    columns=stoit_col_names,
                    index=cnames)

                num_stoits = len(stoit_col_names)
                zero_cov_cnames = []
                covered_cnames = []
                for cname in cnames:
                    try:
                        coverages_df.loc[cname] = coverages.loc[cname]
                        covered_cnames.append(cname)
                    except KeyError:
                        zero_cov_cnames.append(cname)

                norm_coverages_df = coverages_df.apply(np.linalg.norm, axis=1).to_frame()

                print(coverages_df.loc['UTIG00000002'])
                exit(0)


                import mimetypes
                GM_open = open
                open_mode = 'r'
                try:
                    # handle gzipped files
                    mime = mimetypes.guess_type(contigs_file)
                    if mime[1] == 'gzip':
                        import gzip
                        GM_open = gzip.open
                        open_mode = 'rt'
                except:
                    print('Error when guessing contig file mimetype')
                    raise

                with GM_open(contigs_file, open_mode) as f:
                    try:
                        cnames, contigs_df, ksigs_df = contig_parser.parse(f, cutoff, kse)
                        num_contigs = len(cnames)
                    except:
                        print('Error parsing contigs')
                        raise

                if len(zero_cov_cnames) > 0:
                    # report the bad contigs to the user
                    # and strip them before writing to the DB
                    print("****************************************************************")
                    print(" IMPORTANT! - there are %d contigs with 0 coverage" % len(zero_cov_cnames))
                    print(" across all stoits. They will be ignored:")
                    print("****************************************************************")
                    for i in range(0, min(5, len(zero_cov_cnames))):
                        print(zero_cov_cnames[i])
                    if len(zero_cov_cnames) > 5:
                      print('(+ %d additional contigs)' % (len(zero_cov_cnames)-5))
                    print("****************************************************************")

                    cnames = covered_cnames
                    contigs_df = contigs_df.drop(zero_cov_cnames, axis=0)
                    ksigs_df = ksigs_df.drop(zero_cov_cnames, axis=0)
                    coverages_df = coverages_df.drop(zero_cov_cnames, axis=0)

                num_contigs = len(cnames)

                # kmer sigs
                ksig_db_desc = [(mer, float) for mer in kse.kmer_cols]
                try:
                    h5file.create_table(
                        profile_group,
                        'kms',
                        np.array(
                            [tuple(i) for i in ksigs_df.to_numpy(dtype=float)],
                            dtype=ksig_db_desc),
                        title='Kmer signatures',
                        expectedrows=num_contigs)
                except:
                    print("Error creating KMERSIG table:", exc_info()[0])
                    raise

                # raw cov profiles
                coverages_db_desc = [(scn, float) for scn in stoit_col_names]
                try:
                    h5file.create_table(
                        profile_group,
                        'coverage',
                        np.array(
                            [tuple(i) for i in coverages_df.to_numpy(dtype=float)],
                            dtype=coverages_db_desc),
                        title="Bam based coverage",
                        expectedrows=num_contigs)
                except:
                    print("Error creating coverage table:", exc_info()[0])
                    raise

                # normalised coverages
                norm_coverage_db_desc = [('normCov', float)]
                try:
                    h5file.create_table(
                        profile_group,
                        'normCoverage',
                        np.array(
                            [tuple(i) for i in norm_coverages_df.to_numpy(dtype=float)],
                            dtype=norm_coverage_db_desc),
                        title="Normalised coverage",
                        expectedrows=num_contigs)
                except:
                    print("Error creating normalised coverage table:", exc_info()[0])
                    raise

                #------------------------
                # Add a table for the contigs
                #------------------------
                contigs_db_desc = [
                    ('cid', '|S512'),
                    ('bid', int),
                    ('length', int),
                    ('gc', float)]

                try:
                    h5file.create_table(
                        meta_group,
                        'contigs',
                        np.array(
                            [tuple(i) for i in contigs_df.to_numpy()],
                            dtype=contigs_db_desc),
                        title="Contig information",
                        expectedrows=num_contigs)
                except:
                    print("Error creating CONTIG table:", exc_info()[0])
                    raise

                #------------------------
                # Add a table for the bins
                #------------------------
                self.initBinStats((h5file, meta_group))

                print("    %s" % timer.getTimeStamp())

                #------------------------
                # Add metadata
                #------------------------
                meta_data = (
                    str.join(',',stoit_col_names),
                    num_stoits,
                    str.join(',', kse.kmer_cols),
                    kmer_size,
                    kse.num_mers,
                    num_contigs,
                    0,
                    False,
                    False,
                    __current_GMDB_version__)
                self.setMeta(h5file, meta_data)

        except:
            print("Error creating database:", db_file_name, exc_info()[0])
            raise

        print("****************************************************************")
        print("Data loaded successfully!")
        print(" ->",num_contigs,"contigs")
        print(" ->",len(stoit_col_names),"BAM files")
        print("Written to: '"+db_file_name+"'")
        print("****************************************************************")
        print("    %s" % timer.getTimeStamp())

        # all good!
        return True

    def promptOnOverwrite(self, db_file_name, minimal=False):
        """Check that the user is ok with overwriting the db"""
        input_not_ok = True
        valid_responses = ['Y','N']
        vrs = ",".join([str.lower(str(x)) for x in valid_responses])
        while(input_not_ok):
            if(minimal):
                option = input(" Overwrite? ("+vrs+") : ")
            else:

                option = input(" ****WARNING**** Database: '"+db_file_name+"' exists.\n" \
                                   " If you continue you *WILL* delete any previous analyses!\n" \
                                   " Overwrite? ("+vrs+") : ")
            if(option.upper() in valid_responses):
                print("****************************************************************")
                return option.upper()
            else:
                print("Error, unrecognised choice '"+option.upper()+"'")
                minimal = True

#------------------------------------------------------------------------------
# GET LINKS

    def restoreLinks(self, db_file_name, indices=[], silent=False):
        """Restore the links hash for a given set of indices"""
        full_record = []
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                full_record = [list(x) for x in h5file.root.links.links.readWhere("contig1 >= 0")]
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

        if indices == []:
            # get all!
            indices = self.getConditionalIndices(db_file_name, silent=silent)

        links_hash = {}
        if full_record != []:
            for record in full_record:
                # make sure we have storage
                if record[0] in indices and record[1] in indices:
                    try:
                        links_hash[record[0]].append(record[1:])
                    except KeyError:
                        links_hash[record[0]] = [record[1:]]
        return links_hash

#------------------------------------------------------------------------------
# GET / SET DATA TABLES - PROFILES

    def getConditionalIndices(self, db_file_name, condition='', silent=False, checkUpgrade=True):
        """return the indices into the db which meet the condition"""
        # check the DB out and see if we need to change anything about it
        if checkUpgrade:
            self.checkAndUpgradeDB(db_file_name, silent=silent)

        if('' == condition):
            condition = "cid != ''" # no condition breaks everything!
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                return np.array([x.nrow for x in h5file.root.meta.contigs.where(condition)])
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getCoverageProfiles(self, db_file_name, condition='', indices=np.array([])):
        """Load coverage profiles"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                if(np.size(indices) != 0):
                    return np.array([list(h5file.root.profile.coverage[x]) for x in indices])
                else:
                    if('' == condition):
                        condition = "cid != ''" # no condition breaks everything!
                    return np.array([list(h5file.root.profile.coverage[x.nrow]) for x in h5file.root.meta.contigs.where(condition)])
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getTransformedCoverageProfiles(self, db_file_name, condition='', indices=np.array([])):
        """Load transformed coverage profiles"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                if(np.size(indices) != 0):
                    return np.array([list(h5file.root.profile.transCoverage[x]) for x in indices])
                else:
                    if('' == condition):
                        condition = "cid != ''" # no condition breaks everything!
                    return np.array([list(h5file.root.profile.transCoverage[x.nrow]) for x in h5file.root.meta.contigs.where(condition)])
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getNormalisedCoverageProfiles(self, db_file_name, condition='', indices=np.array([])):
        """Load normalised coverage profiles"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                if(np.size(indices) != 0):
                    return np.array([list(h5file.root.profile.normCoverage[x]) for x in indices])
                else:
                    if('' == condition):
                        condition = "cid != ''" # no condition breaks everything!
                    return np.array([list(h5file.root.profile.normCoverage[x.nrow]) for x in h5file.root.meta.contigs.where(condition)])
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def nukeBins(self, db_file_name):
        """Reset all bin information, completely"""
        print("    Clearing all old bin information from",db_file_name)
        self.setBinStats(db_file_name, [])
        self.setNumBins(db_file_name, 0)
        self.setBinAssignments(db_file_name, updates={}, nuke=True)

    def initBinStats(self, storage):
        '''Initialise the bins table

        Inputs:
         storage - (hdf5 file handle, hdf5 node), open db and node to write to

        Outputs:
         None
        '''
        db_desc = [('bid', int),
                   ('numMembers', int),
                   ('isLikelyChimeric', bool)]
        bd = np.array([], dtype=db_desc)

        h5file = storage[0]
        meta_group = storage[1]

        h5file.create_table(meta_group,
                           'bins',
                           bd,
                           title="Bin information",
                           expectedrows=1)

    def setBinStats(self, db_file_name, updates):
        """Set bins table

        updates is a list of tuples which looks like:
        [ (bid, numMembers, isLikelyChimeric) ]
        """

        db_desc = [('bid', int),
                   ('numMembers', int),
                   ('isLikelyChimeric', bool)]
        bd = np.array(updates, dtype=db_desc)

        try:
            with tables.open_file(db_file_name, mode='a', rootUEP="/") as h5file:
                mg = h5file.get_node('/', name='meta')
                # nuke any previous failed attempts
                try:
                    h5file.remove_node(mg, 'tmp_bins')
                except:
                    pass

                try:
                    h5file.create_table(mg,
                                       'tmp_bins',
                                       bd,
                                       title="Bin information",
                                       expectedrows=1)
                except:
                    print("Error creating META table:", exc_info()[0])
                    raise

                # rename the tmp table to overwrite
                h5file.rename_node(mg, 'bins', 'tmp_bins', overwrite=True)
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getBinStats(self, db_file_name):
        """Load data from bins table

        Returns a dict of type:
        { bid : [numMembers, isLikelyChimeric] }
        """
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                ret_dict = {}
                all_rows = h5file.root.meta.bins.read()
                for row in all_rows:
                    ret_dict[row[0]] = [row[1], row[2]]

                return ret_dict
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise
        return {}

    def getBins(self, db_file_name, condition='', indices=np.array([])):
        """Load per-contig bins"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                if(np.size(indices) != 0):
                    return np.array([h5file.root.meta.contigs[x][1] for x in indices]).ravel()
                else:
                    if('' == condition):
                        condition = "cid != ''" # no condition breaks everything!
                    return np.array([list(x)[1] for x in h5file.root.meta.contigs.readWhere(condition)]).ravel()
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def setBinAssignments(self, storage, updates=None, image=None, nuke=False):
        """Set per-contig bins

        updates is a dictionary which looks like:
        { tableRow : binValue }
        if updates is set then storage is the
        path to the hdf file

        image is a list of tuples which look like:
        [(cid, bid, len, gc)]
        if image is set then storage is a tuple of type:
        (h5file, group)
        """
        db_desc = [('cid', '|S512'),
                   ('bid', int),
                   ('length', int),
                   ('gc', float)]
        closeh5 = False
        if updates is not None:
            # we need to build the image
            db_file_name = storage
            cnames = self.getContigNames(db_file_name)
            contig_lengths = self.getContigLengths(db_file_name)
            contig_gcs = self.getContigGCs(db_file_name)
            num_contigs = len(contig_lengths)
            if nuke:
                # clear all bin assignments
                bins = [0]*num_contigs
            else:
                bins = self.getBins(db_file_name)

            # now apply the updates
            for tr in list(updates.keys()):
                bins[tr] = updates[tr]

            # and build the image
            image = np.array(list(zip(cnames, bins, contig_lengths, contig_gcs)),
                             dtype=db_desc)

            try:
                h5file = tables.open_file(db_file_name, mode='a')
            except:
                print("Error opening DB:",db_file_name, exc_info()[0])
                raise
            meta_group = h5file.get_node('/', name='meta')
            closeh5 = True

        elif image is not None:
            h5file = storage[0]
            meta_group = storage[1]
            num_contigs = len(image)
            image = np.array(image,
                             dtype=db_desc)
        else:
            print("get with the program dude")
            return

        # now we write the data
        try:
            # get rid of any failed attempts
            h5file.remove_node(meta_group, 'tmp_contigs')
        except:
            pass

        try:
            h5file.create_table(meta_group,
                               'tmp_contigs',
                               image,
                               title="Contig information",
                               expectedrows=num_contigs)
        except:
            print("Error creating CONTIG table:", exc_info()[0])
            raise

        # rename the tmp table to overwrite
        h5file.rename_node(meta_group, 'contigs', 'tmp_contigs', overwrite=True)
        if closeh5:
            h5file.close()

    def getContigNames(self, db_file_name, condition='', indices=np.array([])):
        """Load contig names"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                if(np.size(indices) != 0):
                    return np.array([h5file.root.meta.contigs[x][0] for x in indices]).ravel()
                else:
                    if('' == condition):
                        condition = "cid != ''" # no condition breaks everything!
                    return np.array([list(x)[0] for x in h5file.root.meta.contigs.readWhere(condition)]).ravel()
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getContigLengths(self, db_file_name, condition='', indices=np.array([])):
        """Load contig lengths"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                if(np.size(indices) != 0):
                    return np.array([h5file.root.meta.contigs[x][2] for x in indices]).ravel()
                else:
                    if('' == condition):
                        condition = "cid != ''" # no condition breaks everything!
                    return np.array([list(x)[2] for x in h5file.root.meta.contigs.readWhere(condition)]).ravel()
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getContigGCs(self, db_file_name, condition='', indices=np.array([])):
        """Load contig gcs"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                if(np.size(indices) != 0):
                    return np.array([h5file.root.meta.contigs[x][3] for x in indices]).ravel()
                else:
                    if('' == condition):
                        condition = "cid != ''" # no condition breaks everything!
                    return np.array([list(x)[3] for x in h5file.root.meta.contigs.readWhere(condition)]).ravel()
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getKmerSigs(self, db_file_name, condition='', indices=np.array([])):
        """Load kmer sigs"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                if(np.size(indices) != 0):
                    return np.array([list(h5file.root.profile.kms[x]) for x in indices])
                else:
                    if('' == condition):
                        condition = "cid != ''" # no condition breaks everything!
                    return np.array([list(h5file.root.profile.kms[x.nrow]) for x in h5file.root.meta.contigs.where(condition)])
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

#------------------------------------------------------------------------------
# GET / SET METADATA

    def getTransformedCoverageCorners(self, db_file_name):
        """Load transformed coverage corners"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                return np.array([list(x) for x in h5file.root.meta.transCoverageCorners.read()])
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def setMeta(self, h5file, metaData, overwrite=False):
        """Write metadata into the table

        metaData should be a tuple of values
        """
        db_desc = [('stoit_col_names', '|S512'),
                   ('numStoits', int),
                   ('merColNames', '|S4096'),
                   ('merSize', int),
                   ('numMers', int),
                   ('numCons', int),
                   ('numBins', int),
                   ('clustered', bool),     # set to true after clustering is complete
                   ('complete', bool),      # set to true after clustering finishing is complete
                   ('formatVersion', int)]
        md = np.array([metaData], dtype=db_desc)

        # get hold of the group
        mg = h5file.get_node('/', name='meta')

        if overwrite:
            t_name = 'tmp_meta'
            # nuke any previous failed attempts
            try:
                h5file.remove_node(mg, 'tmp_meta')
            except:
                pass
        else:
            t_name = 'meta'

        try:
            h5file.create_table(mg,
                               t_name,
                               md,
                               "Descriptive data",
                               expectedrows=1)
        except:
            print("Error creating META table:", exc_info()[0])
            raise

        if overwrite:
            # rename the tmp table to overwrite
            h5file.rename_node(mg, 'meta', 'tmp_meta', overwrite=True)

    def getMetaField(self, db_file_name, fieldName):
        """return the value of fieldName in the metadata tables"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                # theres only one value
                return h5file.root.meta.meta.read()[fieldName][0]
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def setGMDBFormat(self, db_file_name, version):
        """Update the GMDB format version"""
        stoit_col_names = self.getStoitColNames(db_file_name)
        meta_data = (stoit_col_names,
                    len(stoit_col_names.split(',')),
                    self.getMerColNames(db_file_name),
                    self.getMerSize(db_file_name),
                    self.getNumMers(db_file_name),
                    self.getNumCons(db_file_name),
                    self.getNumBins(db_file_name),
                    self.isClustered(db_file_name),
                    self.isComplete(db_file_name),
                    version)
        try:
            with tables.open_file(db_file_name, mode='a', rootUEP="/") as h5file:
                self.setMeta(h5file, meta_data, overwrite=True)
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getGMDBFormat(self, db_file_name):
        """return the format version of this GM file"""
        # this guy needs to be a bit different to the other meta methods
        # becuase earlier versions of GM didn't include a format parameter
        with tables.open_file(db_file_name, mode='r') as h5file:
            # theres only one value
            try:
                this_DB_version = h5file.root.meta.meta.read()['formatVersion'][0]
            except ValueError:
                # this happens when an oldskool formatless DB is loaded
                this_DB_version = 0
        return this_DB_version

    def getNumStoits(self, db_file_name):
        """return the value of numStoits in the metadata tables"""
        return self.getMetaField(db_file_name, 'numStoits')

    def getMerColNames(self, db_file_name):
        """return the value of merColNames in the metadata tables"""
        return self.getMetaField(db_file_name, 'merColNames')

    def getMerSize(self, db_file_name):
        """return the value of merSize in the metadata tables"""
        return self.getMetaField(db_file_name, 'merSize')

    def getNumMers(self, db_file_name):
        """return the value of numMers in the metadata tables"""
        return self.getMetaField(db_file_name, 'numMers')

    def getNumCons(self, db_file_name):
        """return the value of numCons in the metadata tables"""
        return self.getMetaField(db_file_name, 'numCons')

    def setNumBins(self, db_file_name, numBins):
        """set the number of bins"""
        stoit_col_names = self.getStoitColNames(db_file_name)
        meta_data = (stoit_col_names,
                    len(stoit_col_names.split(',')),
                    self.getMerColNames(db_file_name),
                    self.getMerSize(db_file_name),
                    self.getNumMers(db_file_name),
                    self.getNumCons(db_file_name),
                    numBins,
                    self.isClustered(db_file_name),
                    self.isComplete(db_file_name),
                    self.getGMDBFormat(db_file_name))
        try:
            with tables.open_file(db_file_name, mode='a', rootUEP="/") as h5file:
                self.setMeta(h5file, meta_data, overwrite=True)
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def getNumBins(self, db_file_name):
        """return the value of numBins in the metadata tables"""
        return self.getMetaField(db_file_name, 'numBins')

    def getStoitColNames(self, db_file_name):
        """return the value of stoit_col_names in the metadata tables"""
        return self.getMetaField(db_file_name, 'stoit_col_names')

#------------------------------------------------------------------------------
# GET / SET WORKFLOW FLAGS

    def isClustered(self, db_file_name):
        """Has this data set been clustered?"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                return h5file.root.meta.meta.read()['clustered']
        except:
            print("Error opening database:", db_file_name, exc_info()[0])
            raise

    def setClustered(self, db_file_name, state):
        """Set the state of clustering"""
        stoit_col_names = self.getStoitColNames(db_file_name)
        meta_data = (stoit_col_names,
                    len(stoit_col_names.split(',')),
                    self.getMerColNames(db_file_name),
                    self.getMerSize(db_file_name),
                    self.getNumMers(db_file_name),
                    self.getNumCons(db_file_name),
                    self.getNumBins(db_file_name),
                    state,
                    self.isComplete(db_file_name),
                    self.getGMDBFormat(db_file_name))
        try:
            with tables.open_file(db_file_name, mode='a', rootUEP="/") as h5file:
                self.setMeta(h5file, meta_data, overwrite=True)
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

    def isComplete(self, db_file_name):
        """Has this data set been *completely* clustered?"""
        try:
            with tables.open_file(db_file_name, mode='r') as h5file:
                return h5file.root.meta.meta.read()['complete']
        except:
            print("Error opening database:", db_file_name, exc_info()[0])
            raise

    def setComplete(self, db_file_name, state):
        """Set the state of completion"""
        stoit_col_names = self.getStoitColNames(db_file_name)
        meta_data = (stoit_col_names,
                    len(stoit_col_names.split(',')),
                    self.getMerColNames(db_file_name),
                    self.getMerSize(db_file_name),
                    self.getNumMers(db_file_name),
                    self.getNumCons(db_file_name),
                    self.getNumBins(db_file_name),
                    self.isClustered(db_file_name),
                    state,
                    self.getGMDBFormat(db_file_name))
        try:
            with tables.open_file(db_file_name, mode='a', rootUEP="/") as h5file:
                self.setMeta(h5file, meta_data, overwrite=True)
        except:
            print("Error opening DB:",db_file_name, exc_info()[0])
            raise

#------------------------------------------------------------------------------
# FILE / IO

    def dumpData(self, db_file_name, fields, outFile, separator, useHeaders):
        """Dump data to file"""
        header_strings = []
        data_arrays = []

        if fields == ['all']:
            fields = ['names', 'lengths', 'gc', 'bins', 'coverage', 'tcoverage', 'ncoverage', 'mers']

        num_fields = len(fields)
        data_converters = []

        for field in fields:
            if field == 'names':
                header_strings.append('cid')
                data_arrays.append(self.getContigNames(db_file_name))
                data_converters.append(lambda x : x)

            elif field == 'lengths':
                header_strings.append('length')
                data_arrays.append(self.getContigLengths(db_file_name))
                data_converters.append(lambda x : str(x))

            elif field == 'gc':
                header_strings.append('GCs')
                data_arrays.append(self.getContigGCs(db_file_name))
                data_converters.append(lambda x : str(x))

            elif field == 'bins':
                header_strings.append('bid')
                data_arrays.append(self.getBins(db_file_name))
                data_converters.append(lambda x : str(x))

            elif field == 'coverage':
                stoits = self.getStoitColNames(db_file_name).split(',')
                for stoit in stoits:
                    header_strings.append(stoit)
                data_arrays.append(self.getCoverageProfiles(db_file_name))
                data_converters.append(lambda x : separator.join(["%0.4f" % i for i in x]))

            elif field == 'tcoverage':
                header_strings.append('transformedCoverageX')
                header_strings.append('transformedCoverageY')
                header_strings.append('transformedCoverageZ')
                data_arrays.append(self.getTransformedCoverageProfiles(db_file_name))
                data_converters.append(lambda x : separator.join(["%0.4f" % i for i in x]))

            elif field == 'ncoverage':
                header_strings.append('normalisedCoverage')
                data_arrays.append(self.getNormalisedCoverageProfiles(db_file_name))
                data_converters.append(lambda x : separator.join(["%0.4f" % i for i in x]))

            elif field == 'mers':
                mers = self.getMerColNames(db_file_name).split(',')
                for mer in mers:
                    header_strings.append(mer)
                data_arrays.append(self.getKmerSigs(db_file_name))
                data_converters.append(lambda x : separator.join(["%0.4f" % i for i in x]))

        try:
            with open(outFile, 'w') as fh:
                if useHeaders:
                    header = separator.join(header_strings) + "\n"
                    fh.write(header)

                num_rows = len(data_arrays[0])
                for i in range(num_rows):
                    fh.write(data_converters[0](data_arrays[0][i]))
                    for j in range(1, num_fields):
                        fh.write(separator+data_converters[j](data_arrays[j][i]))
                    fh.write('\n')
        except:
            print("Error opening output file %s for writing" % outFile)
            raise

###############################################################################
###############################################################################
###############################################################################
###############################################################################
class ContigParser:
    '''Main class for reading in and parsing contigs'''
    def __init__(self): pass

    def read_fasta(self, fp): # this is a generator function
        header = None
        seq = None
        while True:
            for l in fp:
                if l[0] == '>': # fasta header line
                    if header is not None:
                        # we have reached a new sequence
                        yield header, ''.join(seq)
                    header = l.rstrip()[1:].partition(' ')[0] # save the header we just saw
                    seq = []
                else:
                    seq.append(l.rstrip().upper())
            # anything left in the barrel?
            if header is not None:
                yield header, ''.join(seq)
            break

    def parse(self, contig_file, cutoff, kse):
        '''Do the heavy lifting of parsing'''
        print("Parsing contigs")

        contig_info = {} # save everything here first so we can sort accordingly
        for cname, seq in self.read_fasta(contig_file):
            if len(seq) >= cutoff:
                contig_info[cname] = (
                    kse.get_k_sig(seq), len(seq), kse.get_gc(seq))

        # sort the contig names here once!
        cnames = np.array(sorted(contig_info.keys()))

        ksigs_df = pd.DataFrame(
            columns=kse.kmer_cols,
            index=cnames)

        contigs_df = pd.DataFrame(
            columns=['cid', 'bid', 'length','gc'],
            index=cnames)

        for cname in cnames:
            k_sig, length, gc = contig_info[cname]
            contigs_df.loc[cname] = [cname, 0, length, gc]
            ksigs_df.loc[cname] = k_sig

        return cnames, contigs_df, ksigs_df

    def getWantedSeqs(self, contig_file, wanted, storage={}):
        """Do the heavy lifting of parsing"""
        print("Parsing contigs")
        for cid,seq in self.read_fasta(contig_file):
            if(cid in wanted):
                storage[cid] = seq
        return storage

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class CoverageTransformer(object):
    """Transform coverage profiles GroopM-style"""

    def __init__(self,
                 numContigs,
                 numStoits,
                 normCoverages,
                 kmerNormPC1,
                 coverageProfiles,
                 stoit_col_names,
                 scaleFactor=1000):
        self.numContigs = numContigs
        self.numStoits = numStoits
        self.normCoverages = normCoverages
        self.kmerNormPC1 = kmerNormPC1
        self.covProfiles = coverageProfiles
        self.stoit_col_names = stoit_col_names
        self.indices = list(range(self.numContigs))
        self.scaleFactor = scaleFactor

        # things we care about!
        self.TCentre = None
        self.transformedCP = np.zeros((self.numContigs,3))
        self.corners = np.zeros((self.numStoits,3))

    def transformCP(self, silent=False, nolog=False):
        """Do the main transformation on the coverage profile data"""
        shrinkFn = np.log10
        if(nolog):
            shrinkFn = lambda x:x

        if(not silent):
            print("    Reticulating splines")
            print("    Dimensionality reduction")

        unit_vectors = [(np.cos(i*2*np.pi/self.numStoits),np.sin(i*2*np.pi/self.numStoits)) for i in range(self.numStoits)]

        # make sure the bams are ordered consistently
        if self.numStoits > 3:
            self.shuffleBAMs()

        for i in range(len(self.indices)):
            shifted_vector = np.array([0.0,0.0])
            try:
                flat_vector = (self.covProfiles[i] / sum(self.covProfiles[i]))
            except FloatingPointError:
                flat_vector = self.covProfiles[i]

            for j in range(self.numStoits):
                shifted_vector[0] += unit_vectors[j][0] * flat_vector[j]
                shifted_vector[1] += unit_vectors[j][1] * flat_vector[j]

            # log scale it towards the centre
            scaling_vector = shifted_vector * self.scaleFactor
            sv_size = np.linalg.norm(scaling_vector)
            if sv_size > 1:
                shifted_vector /= shrinkFn(sv_size)

            self.transformedCP[i,0] = shifted_vector[0]
            self.transformedCP[i,1] = shifted_vector[1]
            # should always work cause we nuked
            # all 0 coverage vecs in parse
            self.transformedCP[i,2] = shrinkFn(self.normCoverages[i])

        # finally scale the matrix to make it equal in all dimensions
        min = np.amin(self.transformedCP, axis=0)
        self.transformedCP -= min
        max = np.amax(self.transformedCP, axis=0)
        max = max / (self.scaleFactor-1)
        self.transformedCP /= max

        # get the corner points
        XYcorners = np.reshape([i for i in np.array(unit_vectors)],
                               (self.numStoits, 2))

        for i in range(self.numStoits):
            self.corners[i,0] = XYcorners[i,0]
            self.corners[i,1] = XYcorners[i,1]

        # shift the corners to match the space
        self.corners -= min
        self.corners /= max

        # scale the corners to fit the plot
        cmin = np.amin(self.corners, axis=0)
        self.corners -= cmin
        cmax = np.amax(self.corners, axis=0)
        cmax = cmax / (self.scaleFactor-1)
        self.corners[:,0] /= cmax[0]
        self.corners[:,1] /= cmax[1]
        for i in range(self.numStoits):
            self.corners[i,2] = self.scaleFactor + 100 # only affect the z axis

        self.TCentre = np.mean(self.corners, axis=0)

    def small2indices(self, index, side):
        """Return the indices of the comparative items
        when given an index into a condensed distance matrix
        """
        step = 0
        while index >= (side-step):
            index = index - side + step
            step += 1
        return (step, step + index + 1)

    def shuffleBAMs(self, ordering=None):
        """Make the data transformation deterministic by reordering the bams"""
        # As Ben pointed out. This is basically the travelling salesman.
        if ordering is None:
            # we will need to deduce the ordering of the contigs
            # first we should make a subset of the total data
            # we'd like to take it down to about 1500 or so RI's
            # but we'd like to do this in a repeatable way
            ideal_contig_num = 1500
            sub_cons = np.arange(self.numContigs)
            while len(sub_cons) > ideal_contig_num:
                # select every second contig when sorted by norm cov
                cov_sorted = np.argsort(self.normCoverages[sub_cons])
                sub_cons = np.array([sub_cons[cov_sorted[i*2]] for i in np.arange(int(len(sub_cons)/2))])

                if len(sub_cons) > ideal_contig_num:
                    # select every second contig when sorted by mer PC1
                    mer_sorted = np.argsort(self.kmerNormPC1[sub_cons])
                    sub_cons = np.array([sub_cons[mer_sorted[i*2]] for i in np.arange(int(len(sub_cons)/2))])

            # now that we have a subset, calculate the distance between each of the untransformed vectors
            num_sc = len(sub_cons)

            # log shift the coverages towards the origin
            sub_covs = np.transpose([self.covProfiles[i]*(np.log10(self.normCoverages[i])/self.normCoverages[i]) for i in sub_cons])
            sq_dists = cdist(sub_covs,sub_covs,'cityblock')
            dists = squareform(sq_dists)

            # we have an all vs all distance matrix. Time to do some dodgy optimization.
            # For this case, we only require locally optimal paths. So we can get away with
            # simply choosing the shortest edges in the list
            # initialise a list of left, right neighbours
            lr_dict = {}
            for i in range(self.numStoits):
                lr_dict[i] = []

            too_big = 10000
            while True:
                closest = np.argmin(dists)
                if dists[closest] == too_big:
                    break
                (i,j) = self.small2indices(closest, self.numStoits-1)
                lr_dict[j].append(i)
                lr_dict[i].append(j)

                # mark these guys as neighbours
                if len(lr_dict[i]) == 2:
                    # no more than 2 neighbours
                    sq_dists[i,:] = too_big
                    sq_dists[:,i] = too_big
                    sq_dists[i,i] = 0.0
                if len(lr_dict[j]) == 2:
                    # no more than 2 neighbours
                    sq_dists[j,:] = too_big
                    sq_dists[:,j] = too_big
                    sq_dists[j,j] = 0.0

                # fix the dist matrix
                sq_dists[j,i] = too_big
                sq_dists[i,j] = too_big
                dists = squareform(sq_dists)

            # now everyone should have two neighbours ( but not always )
            # The global path may be separated into several disjoint rings.
            # so we need to make sure that we get all the nodes in the ordering list
            trier = 0   # start of a new disjoint ring
            ordering = [trier]
            while len(ordering) < len(list(lr_dict.keys())):
                try:
                    adding_index = lr_dict[trier][0]    # ok IF this guy has a registered neighbour
                    if adding_index in ordering:        # NOT ok if the neighbour is already in the list
                        raise IndexError()
                    ordering.append(adding_index)
                    while len(ordering) < len(list(lr_dict.keys())):  # try consume the entire ring
                        # len(ordering) >= 2
                        last = ordering[-1]
                        if lr_dict[last][0] == ordering[-2]:    # bi-directionality means this will always work
                            try:
                                adding_index = lr_dict[last][1] # ok IF this guy has two neighbours
                                if adding_index in ordering:    # NOT ok if the neighbour is already in the list
                                    raise IndexError()
                                ordering.append(adding_index)
                            except IndexError:                  # only one neighbour
                                # stick (2 city system)
                                while(trier in ordering):       # find the next index NOT in the ordering
                                    trier += 1
                                if trier < len(list(lr_dict.keys())): # make sure it makes sense
                                    ordering.append(trier)
                                break
                        else:
                            adding_index = lr_dict[last][0]
                            if adding_index in ordering:
                                raise IndexError()
                            ordering.append(adding_index)
                except IndexError:                  # start a new disjoint ring
                    # single point
                    while(trier in ordering):
                        trier += 1
                    if trier < len(list(lr_dict.keys())): # make sure it makes sense
                        ordering.append(trier)

        # sanity check
        if len(ordering) != self.numStoits:
            print("WATTUP, ordering is looking wrong!")
            print(ordering)
            print(lr_dict)

        # reshuffle the contig order!
        # yay for bubble sort!
        working = np.arange(self.numStoits)
        for i in range(1, self.numStoits):
            # where is this guy in the list
            loc = list(working).index(ordering[i])
            if loc != i:
                # swap the columns
                self.covProfiles[:,[i,loc]] = self.covProfiles[:,[loc,i]]
                self.stoit_col_names[[i,loc]] = self.stoit_col_names[[loc,i]]
                working[[i,loc]] = working[[loc,i]]
