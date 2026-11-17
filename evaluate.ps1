param(
    [Parameter(Mandatory)][ValidatePattern('^Day[1-9][0-9]*$')][string]$Day,
    [Parameter(Mandatory)][ValidateRange(1, 999)][int]$Number,
    [Parameter(Mandatory)][ValidateLength(1, 500)][string]$Note,
    [ValidateSet('run', 'submit')][string[]]$Mode = @('run', 'submit'),
    [ValidateSet('cuda', 'python', 'cute')][string]$Language = 'cuda',
    [ValidatePattern('^[\w.-]+$')][string]$Source = 'code.cpp',
    [ValidateRange(1, 9999)][int]$Trial = 1
)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$folder = Join-Path $repo "$Day/$Number"
$cache = Join-Path $repo ".cache/$Day"
$null = [IO.Directory]::CreateDirectory($cache)
$problem = Get-Content -LiteralPath (Join-Path $folder 'problem.json') -Raw -Encoding utf8 | ConvertFrom-Json
$contest = $problem.contest.id
$slug = $problem.slug
$path = Join-Path $folder $Source
$code = [IO.File]::ReadAllText($path)
$sha = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($code))).ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($Note)) { throw '版本备注应写明本次优化尝试' }
$relay = $env:MOCK_RELAY_URL
if (!$relay -or ([uri]$relay).Scheme -ne 'https') { throw 'MOCK_RELAY_URL 需要填写 HTTPS 地址' }
if (!$env:MOCK_RELAY_LOGIN_FILE) { throw '请通过 MOCK_RELAY_LOGIN_FILE 指定登录文件' }
$login = [IO.File]::ReadAllText($env:MOCK_RELAY_LOGIN_FILE)
$headers = @{'X-Relay-Request'='1'}
$null = Invoke-RestMethod "$relay/api/login" -Method Post -ContentType 'application/json' -Headers $headers -Body $login -SessionVariable session -MaximumRedirection 0 -TimeoutSec 25
$request = @{WebSession=$session; MaximumRedirection=0; TimeoutSec=25}
foreach ($action in $Mode) {
    $suffix = if ($Trial -gt 1) { "-trial$Trial" } else { '' }
    $languageTag = if ($Language -ne 'cuda') { "-$Language" } else { '' }
    $file = Join-Path $cache "$slug-$action-$sha$languageTag$suffix.json"
    $lock = [IO.File]::Open("$file.lock", 'OpenOrCreate', 'ReadWrite', 'None')
    try {
        if (Test-Path -LiteralPath $file) {
            $record = Get-Content -LiteralPath $file -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
            if ($record.note -ne $Note) { throw '同一次评测须沿用原备注，独立复测可指定新的 Trial 并说明目的' }
        } else {
            $record = [ordered]@{contest=$contest; problem=$slug; mode=$action; language=$Language; note=$Note; sha256=$sha; client_key=[Guid]::NewGuid().ToString(); id=$null}
            $record | ConvertTo-Json | Set-Content -LiteralPath $file -Encoding utf8
        }
        if (!$record.id) {
            if ($action -eq 'submit') { & (Join-Path $repo 'sync_team_best.ps1') -Day $Day -Session $session }
            $body = @{contest=$contest; problem=$slug; language=$Language; mode=$action; code=$code; note=$Note; client_key=$record.client_key; start_queue=$true} | ConvertTo-Json -Compress
            $created = Invoke-RestMethod "$relay/api/jobs" @request -Method Post -ContentType 'application/json; charset=utf-8' -Headers $headers -Body $body
            if (!$created.id) { throw '平台未返回任务编号' }
            $record.id = $created.id
            $record | ConvertTo-Json | Set-Content -LiteralPath $file -Encoding utf8
        }
        Write-Output "$action $slug $($record.id)"
        do {
            $job = Invoke-RestMethod "$relay/api/jobs/$($record.id)" @request
            if ($job.job) { $job = $job.job }
            if ($job.id -ne $record.id -or $job.contest -ne $contest -or $job.problem -ne $slug -or
                $job.sha256 -ne $sha -or $job.note -ne $Note) { throw '任务编号、源码或版本备注不符' }
            $pending = $job.state -in 'queued','posting','confirming','streaming'
            if ($pending) { Start-Sleep -Seconds 4 }
        } while ($pending)
        $record.state = $job.state
        $record.submission_id = $job.submission_id
        $record.job_id = $job.job_id
        $record.notice = $job.notice
        $record.result = if ($job.result -is [string] -and $job.result) { $job.result | ConvertFrom-Json } else { $job.result }
        $record | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $file -Encoding utf8
        $result = $record.result
        $floor = @{}
        if ($action -eq 'submit' -and $result.status -eq 'ACCEPTED') {
            $floor = @{ScoredProblem=$slug; MinimumScoreUnits=[long]$result.contest_score_units}
        }
        & (Join-Path $repo 'sync_team_best.ps1') -Day $Day -Session $session @floor
        if ($action -eq 'submit' -and $null -ne $result.contest_score_units) {
            Write-Output "$($result.status) $($result.passed_tests)/$($result.total_tests) $($result.total_runtime_ms) ms $($result.contest_score_units / 10000) 分"
        } else { Write-Output $result.status }
        $expected = if ($action -eq 'run') { 'PASSED' } else { 'ACCEPTED' }
        if ($record.state -ne 'done' -or $result.status -ne $expected) { throw "$slug $action $($record.state) $($result.status) $($record.notice)" }
    } finally {
        $lock.Dispose()
    }
}
