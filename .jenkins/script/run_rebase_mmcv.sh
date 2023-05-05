#!/bin/bash

if [ $# != 3 ];then
  echo "Usage: bash rebase.sh branch1,branch2,branch3 NAME TOKEN"
  exit -1
fi

IFS=","
BRANCHS=$1
NAME=$2
TOKEN=$3

rm -rf mmcv
git clone http://$NAME:$TOKEN@gitlab.software.cambricon.com/neuware/software/framework/openmmlab/mmcv.git
pushd mmcv
  for branch in $BRANCHS
  do
    git checkout $branch
    if [ $? != 0 ];then
      echo "git checkout branch: $branch failed!"
      exit -1
    fi
  done

  git remote add upstream https://github.com/open-mmlab/mmcv.git
  git fetch upstream

  for branch in $BRANCHS
  do
    echo "Rebase branch: $branch"
    git checkout $branch
    git rebase upstream/$branch
    if [ $? != 0 ];then
      echo "git rebase upstream/$branch failed!"
      exit -1
    fi
    git push origin $branch
    if [ $? != 0 ];then
      echo "git push origin/$branch failed!"
      exit -1
    fi
    echo "Rebase branch: $branch succeed!"
  done
popd
rm -rf mmcv
