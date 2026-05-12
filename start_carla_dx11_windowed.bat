@echo off
cd /d "%~dp0"
start "CARLA" CarlaUE4.exe -dx11 -quality-level=Low -nosound -carla-rpc-port=2000 -windowed -ResX=800 -ResY=600
