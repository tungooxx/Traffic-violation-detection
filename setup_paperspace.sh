#!/bin/bash
# setup_paperspace.sh - Automates CARLA 0.9.15 installation on Paperspace

echo "[1/4] Updating system and installing dependencies..."
sudo apt-get update && sudo apt-get install -y wget tar python3-pip libomp5 tmux libpng16-16 libjpeg8 libtiff5

echo "[2/4] Downloading CARLA 0.9.15 (this may take a few minutes)..."
# Using a reliable mirror or direct link
wget https://carla-releases.s3.eu-west-1.amazonaws.com/Linux/CARLA_0.9.15.tar.gz

echo "[3/4] Extracting CARLA..."
mkdir -p carla
tar -xzf CARLA_0.9.15.tar.gz -C carla
rm CARLA_0.9.15.tar.gz

echo "[4/4] Installing Python dependencies..."
# Ensure we are using the right pip
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install carla==0.9.15

echo "-------------------------------------------------------"
echo "SETUP COMPLETE!"
echo "To start CARLA in the background, run:"
echo "  tmux new -s carla_server"
echo "  ./carla/CarlaUE4.sh -RenderOffScreen -nosound -opengl"
echo "Then press Ctrl+B then D to detach."
echo "-------------------------------------------------------"
echo "After that, you can run the data generation:"
echo "  python3 carla_data_gen.py --town Town03 --num-vehicles 100"
echo "-------------------------------------------------------"
