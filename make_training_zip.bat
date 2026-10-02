@echo off
REM Double-click this file. It builds raga.zip, the file you drag into Colab.
REM Safe to run again any time - it always rebuilds from the current committed code.

cd /d "%~dp0"
git archive HEAD -o raga.zip --prefix=raga/

if exist raga.zip (
    echo.
    echo Done. raga.zip is in this folder: %cd%
    echo Now go to Colab and drag it into the Files panel.
) else (
    echo.
    echo Something went wrong - raga.zip was not created.
    echo Make sure this .bat file is inside the Raaga_trial_1 project folder.
)
echo.
pause
