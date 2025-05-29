#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
cwdDir=$PWD
export PYTHONUNBUFFERED=1
export PATH=/opt/miniconda3/envs/venv-py3/bin:$PATH
export TS=$(date +%Y%m%d%H%M%S)
export DATE=`date "+%Y%m%d"`
export DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` #add %3N as we want millisecond too

# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/..

# check env
if [ ! -f .env ]; then
    echo `pwd`"/.env file not found"
    exit 1
fi

source .env

if [ -z ${HYPER_PARAM_EPOCH+x} ]; then echo "ERROR, HYPER_PARAM_EPOCH is not defined"; exit 2; else echo "HYPER_PARAM_EPOCH is set to '$HYPER_PARAM_EPOCH'"; fi

if [ ! -d tmp ]; then
    mkdir tmp
fi

# commit changes
cd $baseDir/..
# $baseDir/commit.sh

if [ ! $? -eq 0 ]; then
    echo "Error on commit code before training."
    exit 1 
fi

GIT_COMMIT_SHORT=`git rev-parse --short=9 HEAD`
echo "" >> .env
echo "#AUTO GENERATED" >> .env
echo "GIT_COMMIT_SHORT=$GIT_COMMIT_SHORT" >> .env

# start train
cd $baseDir/../src
python train_example_002.py

# push results
# cd $baseDir/..
# git push origin master