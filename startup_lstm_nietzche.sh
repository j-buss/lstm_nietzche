#!/bin/bash
# VM Startup Script
sudo apt-get update
sudo apt-get install git
sudo apt-get install python3-pip
git clone https://github.com/j-buss/lstm_nietzche.git
cd lstm_nietzche/
pip3 install -r requirements.txt
python3 lstm.py 
