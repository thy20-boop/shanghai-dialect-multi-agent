param(
    [string]$RuntimeRoot = $env:SHANGHAI_WU_RUNTIME
)

$ErrorActionPreference = "Stop"
$legacyRuntime = "D:\wswu_runtime"
if (-not $RuntimeRoot) {
    $RuntimeRoot = if (Test-Path $legacyRuntime) { $legacyRuntime } else { Join-Path $env:LOCALAPPDATA "ShanghaiDialectAgent\wswu_runtime" }
}
$BaseDir = Join-Path $RuntimeRoot "models\CosyVoice2-Wu-SFT-runtime"
$DownloadDir = Join-Path $RuntimeRoot "downloads\WenetSpeech-Wu-Generation"
if (-not (Test-Path (Join-Path $BaseDir "cosyvoice2.yaml"))) {
    throw "Base CosyVoice2-Wu-SFT runtime not found: $BaseDir"
}
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

function Download-VariantWeight {
    param([string]$RemotePath, [string]$LocalName)
    $target = Join-Path $DownloadDir $LocalName
    if (-not (Test-Path $target) -or (Get-Item $target).Length -lt 1000000000) {
        $url = "https://hf-mirror.com/ASLP-lab/WenetSpeech-Wu-Speech-Generation/resolve/main/$RemotePath`?download=true"
        & curl.exe -L --retry 20 --retry-delay 5 -C - -o $target $url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $RemotePath"
        }
    }
    return $target
}

function New-VariantRuntime {
    param([string]$TargetDir, [string]$WeightPath)
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    Get-ChildItem -LiteralPath $BaseDir -Recurse -File | Where-Object { $_.Name -ne "llm.pt" } | ForEach-Object {
        $relative = $_.FullName.Substring($BaseDir.Length).TrimStart('\')
        $destination = Join-Path $TargetDir $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        if (-not (Test-Path $destination)) {
            New-Item -ItemType HardLink -Path $destination -Target $_.FullName | Out-Null
        }
    }
    $llm = Join-Path $TargetDir "llm.pt"
    if (Test-Path $llm) {
        Remove-Item -LiteralPath $llm -Force
    }
    New-Item -ItemType HardLink -Path $llm -Target $WeightPath | Out-Null
}

$cpt = Download-VariantWeight "CosyVoice2-Wu-CPT/CPT.pt" "CosyVoice2-Wu-CPT.pt"
$prosody = Download-VariantWeight "CosyVoice2-Wu-instruct-prosody/instruct_Pro.pt" "CosyVoice2-Wu-instruct-prosody.pt"

New-VariantRuntime (Join-Path $RuntimeRoot "models\CosyVoice2-Wu-CPT-runtime") $cpt
New-VariantRuntime (Join-Path $RuntimeRoot "models\CosyVoice2-Wu-instruct-prosody-runtime") $prosody

Write-Host "Wu expert variants are ready."
Write-Host "  SFT:     $BaseDir"
Write-Host "  CPT:     $(Join-Path $RuntimeRoot 'models\CosyVoice2-Wu-CPT-runtime')"
Write-Host "  Prosody: $(Join-Path $RuntimeRoot 'models\CosyVoice2-Wu-instruct-prosody-runtime')"
