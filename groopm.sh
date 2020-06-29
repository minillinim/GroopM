#!/usr/bin/env bash
set -ue
#
# This file hides some docker mess from the user
#

groopm_image="groopm:latest"

display_usage() {
  echo "

  Usage for image: ${groopm_image}

  This is a bash wrapper of the GroopM docker image. When run, it will map the current
  folder you're in to /app/data inside the image. So always specify paths to data
  relative to a subset of a SUB-directory of the folder you're in when you it.

  Assuming a folder structure like:

    /path/to/current/folder
    ├── contigs
    │   └── assembly.fasta
    ├── reads
    │   ├── sample1.fastq
    │   ├── ...

  Then you can run:

    /path/to/groopm.sh parse -d contigs/assembly.fa -i reads/sample1.fastq ...

  You can run any command in GroopM like this. See GroopM help below
  "
  docker run --rm -it "${groopm_image}"

  exit ${1}
}

# Process the arguments
subcommand="${1:-help}" && shift || true
if [[ ${subcommand} == "help" ]]; then
  display_usage 0
fi

docker \
  run \
  -v $PWD:/app/data \
  --rm -it \
  "${groopm_image}" \
  /bin/bash -c "cd /app/data && groopm ${subcommand} $*"
