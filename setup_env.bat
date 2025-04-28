@echo off
set "CONDA_PATH=C:\ProgramData\Anaconda3"
call "%CONDA_PATH%\Scripts\activate.bat"
call conda create -n stock-dashboards python=3.9 -y
call conda activate stock-dashboards
call conda install pandas numpy pyyaml matplotlib seaborn -y
call conda install -c conda-forge streamlit -y
call conda install -c conda-forge yfinance -y
echo Environment setup complete!
pause 