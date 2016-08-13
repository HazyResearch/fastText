#!/bin/bash

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | \
  sed -e '/^\(8\|9\)/!d' | \
  sed -e 's/^/__label__/g' | \
  sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
      -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
      -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | \
  tr -s " " | \
  myshuf
}

divider() {
  str="********** $1 **********"
  stars=$(printf '*%.0s' $(seq 1 ${#str}))
  echo -e "\n${stars}\n${str}\n${stars}\n"
}

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/dbpedia.train" ]
then
  divider "Downloading test data. Hang on!"
  wget -c "https://googledrive.com/host/0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k" -O "${DATADIR}/dbpedia_csv.tar.gz"
  tar -xzvf "${DATADIR}/dbpedia_csv.tar.gz" -C "${DATADIR}"
  cat "${DATADIR}/dbpedia_csv/train.csv" | normalize_text > "${DATADIR}/dbpedia.train"
  cat "${DATADIR}/dbpedia_csv/test.csv" | normalize_text > "${DATADIR}/dbpedia.test"
fi

divider "BUILDING"
make clean
make

OPTS="-dim 64 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 10 -thread 12"

divider "TRAINING"
time ./fasttext supervised -input "${DATADIR}/dbpedia.train" -output "${RESULTDIR}/dbpedia" ${OPTS}

divider "TESTING"
./fasttext test "${RESULTDIR}/dbpedia.bin" "${DATADIR}/dbpedia.test"
