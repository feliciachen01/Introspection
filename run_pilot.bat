@echo off
rem One-click pilot: conda env "introspect" (first time only), install, tests, 0.5B sweep, report.
rem Everything is written to run_pilot.log in this folder. Don't edit this file while it runs.
setlocal
cd /d "%~dp0"
set LOG=%~dp0run_pilot.log
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo === pilot started %DATE% %TIME% === > "%LOG%"
set CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat
if not exist "%CONDA_BAT%" set CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat
if not exist "%CONDA_BAT%" (
  echo RUN FAILED: could not find conda >> "%LOG%"
  goto :end
)
call "%CONDA_BAT%" env list | findstr /b /c:"introspect " >nul
if errorlevel 1 (
  echo --- creating conda env introspect --- >> "%LOG%"
  call "%CONDA_BAT%" create -n introspect python=3.11 -y >> "%LOG%" 2>&1
)
call "%CONDA_BAT%" activate introspect >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: could not activate env >> "%LOG%"
  goto :end
)
echo --- installing --- >> "%LOG%"
python -m pip install -e ".[dev]" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: pip install >> "%LOG%"
  goto :end
)
echo --- tests --- >> "%LOG%"
python -m pytest -q >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: tests >> "%LOG%"
  goto :end
)
echo --- sweep --- >> "%LOG%"
introspect sweep --config configs/pilot.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: sweep >> "%LOG%"
  goto :end
)
introspect report --config configs/pilot.yaml >> "%LOG%" 2>&1
echo RUN DONE >> "%LOG%"
:end
echo === finished %DATE% %TIME% === >> "%LOG%"
endlocal
