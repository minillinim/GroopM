#!/usr/bin/env python3

import numpy as np
np.seterr(all='raise')

class KmerSigEngine(object):
    '''Simple class for determining kmer signatures'''
    def __init__(self, kmer_len=4):
        self.kmer_len = kmer_len
        self.compl = str.maketrans('ACGT', 'TGCA')

        self.kmer_cols, self.mer_2_idx = self.make_kmers()
        self.num_mers = len(self.kmer_cols)

    def make_kmers(self):
        '''Work out the range of kmers required based on kmer length

        returns a list of sorted kmers and optionally a llo dict
        '''
        # build up the big list
        nucleotides = ('A','C','G','T')
        kmer_list = ['A','C','G','T']
        for i in range(1, self.kmer_len):
            working_list = []
            for mer in kmer_list:
                for char in nucleotides:
                    working_list.append(mer+char)
            kmer_list = working_list

        # pare it down based on lexicographical ordering
        mer_2_idx = {}
        sorted_llmers = []
        kidx = 0
        for kmer in sorted(kmer_list):
            lmer = self.shift_low_lexi(kmer)
            if lmer in mer_2_idx:
                # seen this one
                mer_2_idx[kmer] = mer_2_idx[lmer]
            else:
                sorted_llmers.append(lmer)
                mer_2_idx[kmer] = kidx
                kidx += 1

        return sorted_llmers, mer_2_idx

    def get_gc(self, seq):
        '''Get the GC of a sequence'''
        GCs = 0.
        seq_len = len(seq)
        for base in seq.upper():
            if base == 'N': seq_len -= 1
            if base == 'G' or base == 'C': GCs += 1

        if seq_len == 0: return 0
        return GCs / seq_len

    def shift_low_lexi(self, seq):
        '''Return the lexicographically lowest form of this sequence'''
        rseq = self.rev_comp(seq)
        if(seq < rseq):
            return seq
        return rseq

    def rev_comp(self, seq):
        '''Return the reverse complement of a sequence'''
        # build a dictionary to know what letter to switch to
        return seq.translate(self.compl)[::-1]

    def get_k_sig(self, seq):
        '''Work out kmer signature for a nucleotide sequence

        returns a tuple of floats which is the kmer sig
        '''
        sig = np.zeros(self.num_mers)
        num_mers = len(seq) - self.kmer_len + 1
        for i in range(0, num_mers):
            try:
                sig[self.mer_2_idx[seq[i:i+self.kmer_len]]] += 1.0
            except KeyError:
                # typically due to an N in the sequence. Reduce the number of mers we've seen
                num_mers -= 1

        # normalise by length and return
        if num_mers > 0:
            return sig / num_mers
        else:
            print('***WARNING*** Sequence "%s" is not playing well with the kmer signature engine ' % seq)
            return np.zeros(self.num_mers)
