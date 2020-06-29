# GroopM REBORN?

After leaving this sit idle for several years I've decided to dust off thre code. I hope it's useful for someone

### Overview

GroopM is a metagenomic binning toolset. It leverages spatio-temoral
dynamics (differential coverage) to accurately (and almost automatically)
extract population genomes from multi-sample metagenomic datasets.

GroopM is largely parameter-free. Use: groopm -h for more info.

For installation and usage instructions see : http://ecogenomics.github.io/GroopM/

### Data preparation and running GroopM

Before running GroopM you need to prep your data. A typical workflow looks like this:

    1. Produce NGS data for your environment across mutiple (3+) samples (spearated spatially or temporally or both).
    2. Co-assemble your reads.
    3. For each sample, map the reads against the co-assembly. GroopM needs sorted indexed bam files. If you have 3 samples then you will produce 3 bam files. I use BWA / Samtools for this.
    4. Take your co-assembled contigs and bam files and load them into GroopM using 'groopm parse' saveName contigs.fa bam1.bam bam2.bam...
    5. Keep following the GroopM workflow. See: groopm -h for more info.

### Licence and referencing

Project home page, info on the source tree, documentation, issues and how to contribute, see http://github.com/ecogenomics/GroopM

If you use this software then we'd love you to cite us.
Our paper is now available at https://peerj.com/articles/603. The DOI is http://dx.doi.org/10.7717/peerj.603

Copyright © 2012-2020 Michael Imelfort.

GroopM is licensed under the GNU GPL v3
See LICENSE.txt for further details.
