#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
cwdDir=$PWD

cd $baseDir/..
git add --all
git commit -m "Update changes"
git pull origin master

exit $?
