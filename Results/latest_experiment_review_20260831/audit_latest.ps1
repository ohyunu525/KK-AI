param([string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path)
$ErrorActionPreference = 'Stop'

function Read-Json([string]$Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}
function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Mean($Values) { ($Values | Measure-Object -Average).Average }
function Stdev($Values) {
    $valuesArray = @($Values)
    $average = Mean $valuesArray
    [math]::Sqrt((($valuesArray | ForEach-Object { ($_ - $average) * ($_ - $average) } | Measure-Object -Sum).Sum) / ($valuesArray.Count - 1))
}
function Minimum-Row($History, [string]$Objective) {
    $History | Sort-Object @{Expression={ $_.validation.$Objective }}, epoch | Select-Object -First 1
}
function Replay($History) {
    $bestStructure = [double]::PositiveInfinity
    $bestTotal = [double]::PositiveInfinity
    $lastImproved = 0
    foreach ($row in $History) {
        $improved = $row.validation.structure -lt $bestStructure -or $row.validation.total -lt $bestTotal
        $bestStructure = [math]::Min($bestStructure, $row.validation.structure)
        $bestTotal = [math]::Min($bestTotal, $row.validation.total)
        if ($improved) { $lastImproved = $row.epoch }
        if ($row.epoch - $lastImproved -ge 20) { break }
    }
    [pscustomobject]@{epoch=$row.epoch; last_improvement_epoch=$lastImproved; bad_epochs=$row.epoch-$lastImproved; best_structure=$bestStructure; best_total=$bestTotal}
}

$pilotRoot = Join-Path $ProjectRoot 'Results\model_experiment10_validation'
$contexts = @()
$runs = @()
foreach ($directory in @('screening_seed43', 'confirmation_seeds41_42', 'confirmation_sign_only43')) {
    $runRoot = Join-Path $pilotRoot $directory
    $context = Read-Json (Join-Path $runRoot 'context.json')
    $contexts += $context
    foreach ($file in Get-ChildItem -LiteralPath $runRoot -File -Filter 'g05_*.json') {
        $run = Read-Json $file.FullName
        Require ($run.history.Count -eq $run.epochs_completed) "History length: $($run.run_name)"
        for ($i = 0; $i -lt $run.history.Count; $i++) {
            $row = $run.history[$i]
            Require ($row.epoch -eq $i + 1) "Epoch sequence: $($run.run_name)"
            foreach ($phase in @('train','validation')) {
                foreach ($metric in @('structure','total','position','magnitude','relative_sign','global_sign')) {
                    $value = [double]$row.$phase.$metric
                    Require (-not [double]::IsNaN($value) -and -not [double]::IsInfinity($value)) "Nonfinite loss: $($run.run_name)"
                }
                $loss = $row.$phase
                Require ([math]::Abs($loss.structure - ($loss.position + $loss.magnitude + $loss.relative_sign)) -lt 1e-6) 'Structure loss sum mismatch'
                Require ([math]::Abs($loss.total - ($loss.structure + $loss.global_sign)) -lt 1e-6) 'Total loss sum mismatch'
            }
        }
        foreach ($objective in @('structure','total')) {
            $minimum = Minimum-Row $run.history $objective
            Require ($minimum.epoch -eq $run."best_${objective}_epoch") "Best epoch: $($run.run_name)"
            Require ($minimum.validation.$objective -eq $run."best_${objective}_loss") "Best loss: $($run.run_name)"
        }
        $replayed = Replay $run.history
        Require ($replayed.epoch -eq $run.epochs_completed -and $replayed.bad_epochs -eq 20) "Early stop: $($run.run_name)"
        Require ($replayed.last_improvement_epoch -eq $run.early_stopping.last_improvement_epoch) 'Stored stop state mismatch'
        Require ($run.stop_reason -eq 'early_stopping') 'Unexpected stop reason'
        $runs += $run
    }
}
Require ($runs.Count -eq 14) 'Expected fourteen completed pilots'
$contextChecks = [ordered]@{}
foreach ($field in @('data_sha256','source_sha256','environment','normalization','split_indices')) {
    $canonical = @($contexts | ForEach-Object { $_.$field | ConvertTo-Json -Depth 5 -Compress })
    $contextChecks[$field] = @($canonical | Select-Object -Unique).Count -eq 1
    Require $contextChecks[$field] "Pilot context differs: $field"
}
foreach ($context in $contexts) {
    Require ($context.split_indices.train.Count -eq 8000 -and $context.split_indices.validation.Count -eq 1000) 'Split size mismatch'
    Require (@(@($context.split_indices.train) + @($context.split_indices.validation) | Select-Object -Unique).Count -eq 9000) 'Duplicate or overlapping split indices'
}

$pairs = @()
$means = @()
$components = @()
$globalChecks = @()
foreach ($model in @('g05_sign_only','g05_full_reconstruction')) {
    foreach ($seed in @(41,42,43)) {
        $baseline = $runs | Where-Object { $_.model -eq $model -and $_.seed -eq $seed -and $_.variant -eq 'baseline' }
        $dropout = $runs | Where-Object { $_.model -eq $model -and $_.seed -eq $seed -and $_.variant -eq 'dropout010' }
        $pairs += [pscustomobject]@{model=$model;seed=$seed;baseline=$baseline.best_structure_loss;dropout010=$dropout.best_structure_loss;improvement_pct=100*($baseline.best_structure_loss-$dropout.best_structure_loss)/$baseline.best_structure_loss}
        if ($model -eq 'g05_sign_only') {
            $maximum = 0.0
            for ($i=0; $i -lt [math]::Min($baseline.history.Count,$dropout.history.Count); $i++) {
                foreach ($phase in @('train','validation')) {
                    $maximum = [math]::Max($maximum,[math]::Abs($baseline.history[$i].$phase.global_sign-$dropout.history[$i].$phase.global_sign))
                }
            }
            $globalChecks += [pscustomobject]@{seed=$seed;max_common_epoch_global_loss_difference=$maximum}
        }
    }
    $modelPairs = @($pairs | Where-Object model -eq $model)
    $means += [pscustomobject]@{model=$model;baseline_structure=Mean $modelPairs.baseline;dropout010_structure=Mean $modelPairs.dropout010;mean_paired_improvement_pct=Mean $modelPairs.improvement_pct;sample_std_improvement_pct=Stdev $modelPairs.improvement_pct}
    foreach ($selection in @('structure','total')) {
        foreach ($metric in @('position','magnitude','relative_sign','global_sign','structure','total')) {
            $averages = @{}
            foreach ($variant in @('baseline','dropout010')) {
                $selectedRuns = @($runs | Where-Object { $_.model -eq $model -and $_.variant -eq $variant })
                $values = @($selectedRuns | ForEach-Object { $_.history[$_."best_${selection}_epoch"-1].validation.$metric })
                $averages[$variant] = Mean $values
            }
            $components += [pscustomobject]@{model=$model;selection=$selection;metric=$metric;baseline=$averages.baseline;dropout010=$averages.dropout010;relative_change_pct=100*($averages.dropout010-$averages.baseline)/$averages.baseline}
        }
    }
}
$published = Read-Json (Join-Path $pilotRoot 'validation_summary.json')
foreach ($pair in $pairs) {
    $saved = $published.confirmation_pairs | Where-Object { $_.model -eq $pair.model -and $_.seed -eq $pair.seed }
    Require ([math]::Abs($saved.structure_improvement_pct-$pair.improvement_pct) -lt 1e-10) 'Published paired improvement mismatch'
}
$csvRows = Import-Csv -LiteralPath (Join-Path $pilotRoot 'pilot_runs.csv')
Require ($csvRows.Count -eq 14) 'Pilot CSV row count mismatch'
foreach ($run in $runs) {
    $csvRow = @($csvRows | Where-Object run_name -eq $run.run_name)
    Require ($csvRow.Count -eq 1) 'Missing or duplicate pilot CSV row'
    foreach ($field in @('best_structure_loss','best_total_loss','best_structure_epoch','best_total_epoch','epochs_completed')) {
        Require ([math]::Abs([double]$csvRow[0].$field-[double]$run.$field) -lt 1e-12) "Pilot CSV differs: $field"
    }
}

$historical = @()
foreach ($seed in @(41,42,43)) {
    $directory = Join-Path $ProjectRoot "Results\new_learning9_experiments\5point_routing_v1_seed$seed\runs"
    foreach ($file in Get-ChildItem -LiteralPath $directory -Recurse -File -Filter history.json) {
        $history = Read-Json $file.FullName
        $replayed = Replay $history
        $bestStructure = Minimum-Row $history 'structure'
        $bestTotal = Minimum-Row $history 'total'
        $historical += [pscustomobject]@{run=$file.Directory.Name;original=$history.Count;stop=$replayed.epoch;both_preserved=([math]::Max($bestStructure.epoch,$bestTotal.epoch) -le $replayed.epoch)}
    }
}
Require ($historical.Count -eq 36 -and @($historical | Where-Object { -not $_.both_preserved }).Count -eq 0) 'Historical replay differs'
$originalEpochs = ($historical.original | Measure-Object -Sum).Sum
$replayedEpochs = ($historical.stop | Measure-Object -Sum).Sum
$runSnapshots = @($runs | ForEach-Object {
    [pscustomobject]@{run=$_.run_name;epochs_completed=$_.epochs_completed;seconds=$_.elapsed_seconds;best_structure_epoch=$_.best_structure_epoch;best_total_epoch=$_.best_total_epoch;best_structure=$_.history[$_.best_structure_epoch-1].validation;best_total=$_.history[$_.best_total_epoch-1].validation;last=$_.history[-1]}
})
Add-Type -AssemblyName System.IO.Compression.FileSystem
$actualDataPath = Join-Path $ProjectRoot 'Models\charge_dataset_5charges_v9.npz'
$archive = [System.IO.Compression.ZipFile]::OpenRead($actualDataPath)
try {
    $arrayHeaders = foreach ($entry in $archive.Entries | Where-Object { $_.FullName -in @('G00.npy','G05.npy','target.npy') }) {
        $stream = $entry.Open()
        try {
            $reader = [System.IO.BinaryReader]::new($stream)
            $magic = $reader.ReadBytes(8)
            $length = if ($magic[6] -eq 1) { $reader.ReadUInt16() } else { $reader.ReadUInt32() }
            [pscustomobject]@{name=$entry.FullName;header=[System.Text.Encoding]::UTF8.GetString($reader.ReadBytes($length)).Trim()}
        } finally { $stream.Dispose() }
    }
} finally { $archive.Dispose() }
$currentSource = foreach ($name in @('Codes\ModelExperiment10.py','Codes\NewLearning9.py','Results\model_experiment10_validation\validation_pilot.py')) {
    $path = Join-Path $ProjectRoot $name
    $lf = [System.IO.File]::ReadAllText($path).Replace("`r`n","`n")
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $lfHash = [BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($lf))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
    [pscustomobject]@{path=$name;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant();lf_sha256=$lfHash}
}
[ordered]@{
    analysis_type='Read-only recomputation of stored experiment histories; no training or inference'
    context_checks=$contextChecks
    batches=@($contexts | ForEach-Object { [pscustomobject]@{created_at=$_.created_at;arguments=$_.arguments;data_sha256=$_.data_sha256;source_sha256=$_.source_sha256;pilot_sha256=$_.pilot_sha256} })
    pilot_integrity=[ordered]@{completed_runs=$runs.Count;history_epochs=($runs.epochs_completed | Measure-Object -Sum).Sum;all_minima_match=$true;all_losses_finite=$true;all_stop_states_match=$true;published_improvements_match=$true;pilot_csv_matches=$true;recorded_loop_seconds=($runs.elapsed_seconds | Measure-Object -Sum).Sum}
    pairs=$pairs
    model_means=$means
    selected_loss_components=$components
    sign_only_global_trajectory=$globalChecks
    historical_replay=[ordered]@{runs=$historical.Count;original_epochs=$originalEpochs;replay_epochs=$replayedEpochs;epoch_reduction_pct=100*(1-$replayedEpochs/$originalEpochs);all_original_best_preserved=$true;min_stop=($historical.stop | Measure-Object -Minimum).Minimum;max_stop=($historical.stop | Measure-Object -Maximum).Maximum}
    run_snapshots=$runSnapshots
    current_default_data=[ordered]@{path=$actualDataPath;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $actualDataPath).Hash.ToLowerInvariant();array_headers=$arrayHeaders;note='The root charge_dataset.npz is a separate two-charge file, not the v10 default.'}
    current_source=$currentSource
} | ConvertTo-Json -Depth 12
