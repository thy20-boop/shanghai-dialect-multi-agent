param(
    [string]$OutputDir = "outputs\wu_expert_benchmark"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$OutputRoot = Join-Path $ProjectRoot $OutputDir
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$BenchmarkConfig = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot "wu_expert_benchmark_samples.json") | ConvertFrom-Json
$Samples = $BenchmarkConfig.samples

function Stop-WuServer {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'scripts\.wenet_wu_server:app'
    } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

function Wait-WuServer {
    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:9881/health" -TimeoutSec 2
            if ($health.status -eq "ok") { return }
        } catch {}
        Start-Sleep -Seconds 1
    }
    throw "Wu expert server did not become ready."
}

$Manifest = @()
foreach ($Expert in @("sft", "cpt", "prosody")) {
    Stop-WuServer
    powershell -ExecutionPolicy Bypass -File scripts\start_wenet_wu_expert.ps1 -Expert $Expert -Detached
    Wait-WuServer
    $ExpertDir = Join-Path $OutputRoot $Expert
    New-Item -ItemType Directory -Force -Path $ExpertDir | Out-Null

    foreach ($Sample in $Samples) {
        $body = @{
            text = $Sample.text
            speed = 1.0
            use_text_frontend = $true
            instruction = $BenchmarkConfig.instruction
        } | ConvertTo-Json
        $output = Join-Path $ExpertDir "$($Sample.id).wav"
        Invoke-WebRequest `
            -Uri "http://127.0.0.1:9881/tts" `
            -Method Post `
            -ContentType "application/json; charset=utf-8" `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
            -OutFile $output `
            -TimeoutSec 300
        $Manifest += @{
            expert = $Expert
            id = $Sample.id
            text = $Sample.text
            audio = $output
        }
    }
}
Stop-WuServer
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $OutputRoot "manifest.json")
Write-Host "Benchmark generation complete: $OutputRoot"
