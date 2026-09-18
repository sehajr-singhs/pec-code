@echo off
cd /d %~dp0
python -u seed_farm.py e7 --seeds 0-3 --workers 2 --threads 2 --profile farm > farm_log_e7.txt 2>&1
