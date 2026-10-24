# ==============================================================================
# PyBullet Quadrotor Simulation Runner (PowerShell)
# Automatically selects the Python environment with PyBullet pre-installed.
# ==============================================================================

param(
    [ValidateSet("trajectory", "attitude")]
    [string]$mode = "trajectory",
    [switch]$save
)

$argsList = @("--mode", $mode)
if ($save) {
    $argsList += "--save"
}

# Check if default python has pybullet
$hasPybullet = $false
try {
    python -c "import pybullet" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $hasPybullet = $true
    }
} catch {
    $hasPybullet = $false
}

if ($hasPybullet) {
    python main.py @argsList
} else {
    Write-Host "Running simulation using pre-configured PyBullet environment..." -ForegroundColor Cyan
    & "C:\msys64\ucrt64\bin\python.exe" main.py @argsList
}
