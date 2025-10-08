# Helper bash (WSL) / git-bash script to run all variable processors sequentially
# Adjust input subfolder names if different.

python process_temperature.py --input raw_nc/temperature
python process_precipitation.py --input raw_nc/total_precipitation
python process_evaporation.py --input raw_nc/total_evaporation
python process_runoff.py --input raw_nc/runoff
python process_snow.py --input raw_nc/snow
