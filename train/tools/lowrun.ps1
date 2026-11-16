# Run a command at BelowNormal priority, wait, print its output. Usage: lowrun.ps1 -Out <file> -Exe <exe> -ArgStr "<args>"
param([string]$Out, [string]$Exe, [string]$ArgStr)
$p = Start-Process -FilePath $Exe -ArgumentList $ArgStr -NoNewWindow -PassThru -RedirectStandardOutput $Out -RedirectStandardError ($Out + '.err')
try { $p.PriorityClass = 'BelowNormal' } catch {}
$p.WaitForExit()
Get-Content $Out; Get-Content ($Out + '.err')
