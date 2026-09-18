@echo off
cd /d C:\Users\sehaj\OneDrive\Desktop\pec2
python -u seed_farm.py e8 --seeds 0-5 --workers 2 --threads 3 --profile farm > farm_log_e8.txt 2>&1
