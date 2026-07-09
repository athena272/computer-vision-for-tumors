@echo off
cd /d "%~dp0"
echo Abrindo interface web do projeto U-Net...
julia --project -t auto scripts/interface.jl
pause
