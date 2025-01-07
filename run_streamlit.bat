@echo off
:: Path to Anaconda installation
set CONDA_PATH=e:\program\anaconda3
:: Activate the environment
call "%CONDA_PATH%\Scripts\activate.bat" health_classify

:: Path to your Python script
streamlit run "e:\program\pythonProject\Smartfarm_project\pig health\healh cassification\healh cassification\app_video.py"

:: Keep the window open after execution
pause
