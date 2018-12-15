#!/bin/bash
apt-get update
apt-get --assume-yes install git
git clone https://github.com/j-buss/lstm_nietzche.git
cd lstm_nietzche/
python3 cpu_perf_test.py
