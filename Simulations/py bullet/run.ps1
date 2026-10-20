# ==============================================================================
# PyBullet Quadrotor Simulation Runner (PowerShell)
# Automatically detects Python with PyBullet pre-installed.
# No hardcoded Python paths required.
# ==============================================================================

param(
    [ValidateSet("trajectory", "attitude")]
    [string]$mode = "trajectory"
)

$argsList = @("--mode", $mode)

# Collect available Python interpreters
$pythonCandidates = @()

# Check Python available through PATH
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonCandidates += $pythonCommand.Source
}

# Check the Windows Python launcher
$pyCommand = Get-Command py -ErrorAction SilentlyContinue
if ($pyCommand) {
    $pyPath = $pyCommand.Source

    # Ask the launcher for Python 3.12, if installed
    $py312 = & $pyPath -3.12 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $py312) {
        $pythonCandidates += ($py312 | Select-Object -Last 1).Trim()
    }

    # Also check the launcher's default Python
    $pyDefault = & $pyPath -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $pyDefault) {
        $pythonCandidates += ($pyDefault | Select-Object -Last 1).Trim()
    }
}

# Remove duplicate interpreter paths
$pythonCandidates = @($pythonCandidates | Select-Object -Unique)

# Find the first interpreter that can import PyBullet
$selectedPython = $null

foreach ($candidate in $pythonCandidates) {
    if (-not (Test-Path $candidate)) {
        continue
    }

    & $candidate -c "import pybullet" 2>$null

    if ($LASTEXITCODE -eq 0) {
        $selectedPython = $candidate
        break
    }
}

# Stop if no compatible Python environment is found
if (-not $selectedPython) {
    Write-Host ""
    Write-Host "ERROR: No Python interpreter with PyBullet was found." -ForegroundColor Red
    Write-Host "Install Python and the project dependencies, including PyBullet." -ForegroundColor Yellow
    Write-Host "For this project, try: python -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using Python: $selectedPython" -ForegroundColor Cyan
Write-Host "Running $mode simulation..." -ForegroundColor Cyan

# Run the simulation
& $selectedPython main.py @argsList

# Return the simulation's exit code
exit $LASTEXITCODE
```