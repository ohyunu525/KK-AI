<#
.SYNOPSIS
Run one independent seed with the frozen dropout020 experiment settings.
.EXAMPLE
.\Modelexperiment11\run_fixed_seed.ps1 -Seed 42
.EXAMPLE
.\Modelexperiment11\run_fixed_seed.ps1 -Seed 42 -Resume
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 4294967295)]
    [long]$Seed,
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'cuda',
    [switch]$Resume,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$experimentName = "seed${Seed}_g05_sweep_dropout020"
$resultsRoot = Join-Path $PSScriptRoot 'fraction_sweep_results'
$checkpointsRoot = Join-Path $PSScriptRoot 'fraction_sweep_checkpoints'
$resultDirectory = Join-Path $resultsRoot $experimentName
$checkpointDirectory = Join-Path $checkpointsRoot $experimentName
$launchDirectory = Join-Path (Join-Path $PSScriptRoot 'fraction_sweep_launches') $experimentName

# Pin the confirmed executable sources and dataset, independently of other seeds.
$requiredHashes = [ordered]@{
    'Codes/ModelExperiment10.py' = '20c2f9e26fee48187fa57e5a55750a9654a93724d11d8f83dfadedb3e598de38'
    'Codes/NewLearning9.py' = 'c026690468ff6e13d6e14534d33ddf6b66b8ab02e48288cb66d1817dacb8c307'
    'Models/charge_dataset_5charges_v9.npz' = 'f90880bbbcd31c528c0603a2aa8043c0ddea567c393d83fc59ae3afd3ac24863'
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python not found: $pythonPath"
}
foreach ($relativePath in $requiredHashes.Keys) {
    $inputPath = Join-Path $projectRoot $relativePath
    $actualHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $requiredHashes[$relativePath]) {
        throw "Frozen source/data hash mismatch: $relativePath. Do not mix changed code or data into this experiment."
    }
}

$trainingArguments = @(
    '-u', (Join-Path $projectRoot 'Codes\ModelExperiment10.py'),
    '--data', (Join-Path $projectRoot 'Models\charge_dataset_5charges_v9.npz'),
    '--models', 'g05_sign_only,g05_full_reconstruction',
    '--fractions', '0.0,0.1,0.25,0.5,0.75,1.0',
    '--seeds', $Seed.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--epochs', '150', '--batch-size', '128',
    '--learning-rate', '0.001', '--weight-decay', '0.0001',
    '--structure-dropout', '0.2', '--early-stopping-patience', '20',
    '--early-stopping-min-delta', '0', '--early-stopping-min-epochs', '0',
    '--experiment-name', $experimentName,
    '--results-root', $resultsRoot, '--checkpoint-root', $checkpointsRoot,
    '--device', $Device
)

if ($DryRun) {
    [ordered]@{
        seed = $Seed; models = @('g05_sign_only', 'g05_full_reconstruction')
        fractions = @(0.0, 0.1, 0.25, 0.5, 0.75, 1.0); expected_runs = 12
        mode = $(if ($Resume) { 'resume_this_seed_only' } else { 'fresh' })
        executable = $pythonPath; arguments = $trainingArguments
        results = $resultDirectory; checkpoints = $checkpointDirectory
        required_sha256 = $requiredHashes
    } | ConvertTo-Json -Depth 8
    return
}

if ($Resume) {
    if (-not (Test-Path -LiteralPath (Join-Path $resultDirectory 'protocol.json') -PathType Leaf) -or
        -not (Test-Path -LiteralPath $checkpointDirectory -PathType Container)) {
        throw 'Resume requires this seed''s existing protocol and checkpoint directory.'
    }
} else {
    foreach ($outputDirectory in @($resultDirectory, $checkpointDirectory)) {
        if (Test-Path -LiteralPath $outputDirectory) {
            throw "Fresh training refuses an existing output directory: $outputDirectory. Use -Resume only to continue this same experiment."
        }
    }
}

# Launch records are outside the trainer's initially empty output directories.
New-Item -ItemType Directory -Path $launchDirectory -Force | Out-Null
$sourceDirectory = Join-Path $launchDirectory 'sources'
New-Item -ItemType Directory -Path $sourceDirectory -Force | Out-Null
foreach ($relativePath in @('Codes/ModelExperiment10.py', 'Codes/NewLearning9.py')) {
    $snapshotPath = Join-Path $sourceDirectory (Split-Path -Leaf $relativePath)
    if (Test-Path -LiteralPath $snapshotPath) {
        $snapshotHash = (Get-FileHash -LiteralPath $snapshotPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($snapshotHash -ne $requiredHashes[$relativePath]) {
            throw "Existing source snapshot differs: $snapshotPath"
        }
    } else {
        Copy-Item -LiteralPath (Join-Path $projectRoot $relativePath) -Destination $snapshotPath
    }
}
$attempt = '{0}_{1}' -f [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'), [Guid]::NewGuid().ToString('N').Substring(0, 8)
$logPath = Join-Path $launchDirectory "$attempt.log"
$manifestPath = Join-Path $launchDirectory "$attempt.json"
$manifest = [ordered]@{
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    seed = $Seed; mode = $(if ($Resume) { 'resume_this_seed_only' } else { 'fresh' })
    expected_runs = 12; models = @('g05_sign_only', 'g05_full_reconstruction')
    fractions = @(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    training = @{ max_epochs = 150; batch_size = 128; learning_rate = 0.001; weight_decay = 0.0001;
        structure_dropout = 0.2; early_stopping_patience = 20; early_stopping_min_delta = 0.0;
        early_stopping_min_epochs = 0; optimizer = 'AdamW'; split_seed = 42 }
    checkpoint_selection = @('validation_loss.structure', 'validation_loss.total')
    primary_reporting_checkpoint = 'best_structure.pt'
    test_policy = 'Evaluate both validation-selected checkpoints after each run finishes; no tuning or selection from test metrics.'
    executable = $pythonPath; arguments = $trainingArguments
    results = $resultDirectory; checkpoints = $checkpointDirectory; console_log = $logPath
    required_sha256 = $requiredHashes
    launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Output "Independent seed $Seed | fresh=$(-not $Resume) | 12 runs | fixed dropout020"
Write-Output "Launch record: $manifestPath"
& $pythonPath @trainingArguments 2>&1 | Tee-Object -FilePath $logPath
$trainingExitCode = $LASTEXITCODE
[ordered]@{
    finished_at_utc = [DateTime]::UtcNow.ToString('o'); seed = $Seed
    exit_code = $trainingExitCode; launch_manifest = $manifestPath
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchDirectory "$attempt.exit.json") -Encoding UTF8
exit $trainingExitCode
