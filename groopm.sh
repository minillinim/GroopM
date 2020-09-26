#!/usr/bin/env bash
set -ue
#
# This file hides some docker mess from the user
#

groopm_image="minillinim/groopm:latest"

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

if [[ ${subcommand} == "build" ]]; then
  docker build -t minillinim/groopm:latest "${1}"
  exit 0
fi

# Add ability to switch to the current user so file perms line up
prepare="groupadd -g$(id -g) -f floop && useradd -u$(id -u) -g$(id -g) floop"
volume_mounts="-v ${PWD}:/app/data"

if [[ ${subcommand} == "dev" ]]; then
  groopm_src="${1}" && shift
  if [ ! -d ${groopm_src} ]; then
    echo "Error: '${groopm_src}' is not a folder"
  fi

  set +u
  subcommand="${1}" && shift || true
  set -u
  if [[ ${subcommand} == "" ]]; then
    docker run --rm -it "${groopm_image}"
    exit 0
  fi
  # add dev install code and file mounts
  volume_mounts="${volume_mounts} -v $(realpath ${groopm_src}):/app"
  prepare="cd /app && python3 setup.py install && ${prepare}"
fi

run_groopm="cd /app/data && su - floop -c \"groopm ${subcommand} $*\""
eval "docker run ${volume_mounts} --rm -it ${groopm_image} /bin/bash -c '${prepare} && ${run_groopm}'"
