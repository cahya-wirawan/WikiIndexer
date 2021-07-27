#!/usr/bin/env bash
set -e

cd /fastapi/WikiSearch
git pull
cd /fastapi/WikiSearch/app

if [ "$DEBUG" = true ] ; then
    echo 'Debugging - ON'
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
else
    echo 'Debugging - OFF'
    uvicorn main:app --host 0.0.0.0 --port 8000 
fi