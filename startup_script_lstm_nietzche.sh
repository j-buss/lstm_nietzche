#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log.out 2>&1
# Everything below will go to the file 'log.out':
sudo apt-get update
sudo apt-get --yes --force-yes install git
sudo apt-get --yes --force-yes install python3-pip
git clone https://github.com/j-buss/lstm_nietzche.git
cd lstm_nietzche/
pip3 install -r requirements.txt
python3 lstm.py 
gsutil cp -r data* gs://lstm-text-gen-001/
