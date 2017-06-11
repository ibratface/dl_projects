#!/bin/bash

hash aws 2>/dev/null
if [ $? -ne 0 ]; then
    echo >&2 "'aws' command line tool required, but not installed.  Aborting."
    exit 1
fi

if [ -z "$(aws configure get aws_access_key_id)" ]; then
    echo "AWS credentials not configured.  Aborting"
    exit 1
fi

# Check for name argument
if [ -z ${1-x} ]; then
    name=$1
else
    name='kaggle'
fi

region=$(aws configure get region)
settings=$region-$name.sh
