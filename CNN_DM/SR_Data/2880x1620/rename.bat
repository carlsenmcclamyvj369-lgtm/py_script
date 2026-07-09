@echo off
setlocal enabledelayedexpansion

for %%a in (*_pp_*.bmp) do (
	set "oldname=%%~na"
	set "newname=!oldname:_pp_=#!"
	for /f "tokens=1 delims=#" %%i in ("!newname!") do (
		ren "%%a" "%%i.bmp"
	)
)

echo rename finished！
pause

