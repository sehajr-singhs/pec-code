@echo off
cd /d %~dp0
python -u seed_farm.py e3,e6,e0,e5 --seeds 0-9 --workers 8 --threads 2 --profile farm > farm_log.txt 2>&1
