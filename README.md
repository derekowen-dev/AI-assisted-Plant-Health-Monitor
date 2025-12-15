Plant Monitor (CSCI 490 AIT Final Project)
This repo runs a Raspberry Pi plant monitoring dashboard with light/temp/moisture sensors, captures a photo, and sends it to a laptop inference server for stress/OK photo prediction

-------------------------------setup -------------------------------

pi hardware (I used a Pi 4 model B (2GB RAM))
sensors: BH1750 (lux) + BMP280 (temp/pressure) + ADS1115 + moisture probe + Raspberry Pi camera
also needed: breadboard + jumper wires + soldering kit (to solder pin heads onto sensors)

on the pi: enable I2C + camera in raspi-config, then reboot

on the laptop: install Anaconda/Miniconda (used for the inference server)

------------------------commands to get started (Pi) ------------------------

pyenv local 3.11.9

python -m venv .venv
source .venv/bin/activate

pip install flask requests pillow
pip install adafruit-blinka adafruit-circuitpython-bh1750 adafruit-circuitpython-bmp280 adafruit-circuitpython-ads1x15

export INFER_SERVER="http://      " - change this to your laptop address

python web_app.py

open the dashboard in a browser (local host address)

------------------------commands to get started (laptop on same network as PI) ------------------------

conda create -n leaf_infer python=3.11 -y
conda activate leaf_infer

conda install pytorch torchvision cpuonly -c pytorch -y
pip install flask pillow

python inference_server.py

-------------------------------notes-------------------------------
the pi captures photos with rpicam-still and saves to test.jpg
sensor logs are stored as CSVs in ./data/ (light.csv, temp.csv, moisture.csv)
the pi calls the laptop inference server to compute photo health + confidence