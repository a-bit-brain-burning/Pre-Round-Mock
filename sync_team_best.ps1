param(
    [Parameter(Mandatory)][ValidatePattern('^Day[1-9][0-9]*$')][string]$Day,
    [Microsoft.PowerShell.Commands.WebRequestSession]$Session,
    [string]$ScoredProblem,
    [long]$MinimumScoreUnits = 0
)
$ErrorActionPreference = 'Stop'
$dayDir = Join-Path $PSScriptRoot $Day
$first = Get-Content -LiteralPath (Join-Path $dayDir '1/problem.json') -Raw -Encoding utf8 | ConvertFrom-Json
$contest = $first.contest.id
$relay = $env:MOCK_RELAY_URL
if (!$relay -or ([uri]$relay).Scheme -ne 'https') { throw 'MOCK_RELAY_URL 需要填写 HTTPS 地址' }
if (!$Session) {
    if (!$env:MOCK_RELAY_LOGIN_FILE) { throw '请通过 MOCK_RELAY_LOGIN_FILE 指定登录文件' }
    $login = [IO.File]::ReadAllText($env:MOCK_RELAY_LOGIN_FILE)
    $null = Invoke-RestMethod "$relay/api/login" -Method Post -ContentType 'application/json' -Headers @{'X-Relay-Request'='1'} -Body $login -SessionVariable Session -MaximumRedirection 0 -TimeoutSec 25
}
$request = @{WebSession=$Session; MaximumRedirection=0; TimeoutSec=25}
$deadline = [DateTimeOffset]::UtcNow.AddSeconds(90)
do {
    $catalog = Invoke-RestMethod "$relay/api/contest?contest=$contest&refresh=1" @request
    if ($catalog.id -ne $contest -or !$catalog.tasks) { throw '比赛目录与本地题面不符' }
    $scored = $catalog.tasks | Where-Object slug -eq $ScoredProblem
    if ([long]$scored.mine.scoreUnits -ge $MinimumScoreUnits) { break }
    if ([DateTimeOffset]::UtcNow -ge $deadline) { throw '提交已结束，计分表仍待更新；可使用原任务继续核对' }
    Start-Sleep -Seconds 4
} while ($true)
$utf8 = [Text.UTF8Encoding]::new($false)
for ($index = 0; $index -lt $catalog.tasks.Count; $index++) {
    $task = $catalog.tasks[$index]
    $folder = Join-Path $dayDir ($index + 1)
    $problem = Get-Content -LiteralPath (Join-Path $folder 'problem.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($task.slug -ne $problem.slug -or $task.label -ne $problem.contest.label) { throw '题号与本地目录不符' }
    $score = [long]$task.mine.scoreUnits
    if ($score -le 0) { continue }
    $previous = Get-ChildItem -LiteralPath $folder -Filter 'team-best-*.json' | ForEach-Object {
        Get-Content -LiteralPath $_.FullName -Raw -Encoding utf8 | ConvertFrom-Json
    } | Sort-Object score_units -Descending | Select-Object -First 1
    if ($previous) {
        $saved = Get-FileHash -LiteralPath (Join-Path $folder $previous.source) -Algorithm SHA256
        if ($saved.Hash -ne $previous.sha256) { throw '已归档的队内源码与回执不符' }
        if ($previous.score_units -ge $score) { continue }
    }
    $detailDeadline = [DateTimeOffset]::UtcNow.AddSeconds(90)
    do {
        $best = Invoke-RestMethod "$relay/api/best-submission?contest=$contest&slug=$($task.slug)" @request
        $result = if ($best.result -is [string]) { $best.result | ConvertFrom-Json } else { $best.result }
        if ([long]$result.contest_score_units -ge $score) { break }
        if ([DateTimeOffset]::UtcNow -ge $detailDeadline) { throw '队内成绩已更新，提交详情仍待同步' }
        Start-Sleep -Seconds 4
    } while ($true)
    if ($best.contest -ne $contest -or $best.problem -ne $task.slug -or $best.mode -ne 'submit' -or
        $best.status -ne 'ACCEPTED' -or $result.status -ne 'ACCEPTED' -or
        $best.id -ne $best.submission_id -or !$best.submission_id -or
        [long]$result.contest_score_units -lt $score -or !$best.code) { throw '最高分回执与比赛记录不符' }
    $bytes = $utf8.GetBytes($best.code)
    $sha = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
    if ($sha -ne $best.sha256) { throw '平台返回的源码与 SHA-256 不符' }
    $note = $best.note
    if ($best.local_job_id) {
        $job = Invoke-RestMethod "$relay/api/jobs/$($best.local_job_id)" @request
        if ($job.job) { $job = $job.job }
        if ($job.submission_id -ne $best.submission_id -or $job.sha256 -ne $sha) { throw '队内任务与最高分提交不符' }
        $details = if ($job.result -is [string]) { $job.result | ConvertFrom-Json } else { $job.result }
        if ($details.status -ne 'ACCEPTED' -or $details.contest_score_units -ne $result.contest_score_units) {
            throw '队内任务的分数与最高分提交不符'
        }
        $result = $details
        $note = $job.note
    }
    $extension = @{cuda='cu'; python='py'; triton='py'; cute='py'}[$best.language]
    if (!$extension) { throw "未知语言 $($best.language)" }
    $name = "team-$($sha.Substring(0,12)).$extension"
    $source = Join-Path $folder $name
    if (!(Test-Path -LiteralPath $source)) { [IO.File]::WriteAllBytes($source, $bytes) }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $sha) { throw '本地源码与平台 SHA-256 不符' }
    $record = [ordered]@{
        contest=$contest; problem=$task.slug; submission_id=$best.submission_id; relay_job=$best.local_job_id
        score_units=[long]$result.contest_score_units; previous_score_units=$previous.score_units
        observed_at=[DateTimeOffset]::UtcNow.ToString('o'); source=$name; sha256=$sha
        note=$note; result=$result
    }
    $receipt = Join-Path $folder "team-best-$($best.submission_id).json"
    [IO.File]::WriteAllText($receipt, (($record | ConvertTo-Json -Depth 32) + [Environment]::NewLine), $utf8)
    Write-Output "$Day $($task.label) $($record.score_units / 10000) $($best.submission_id) $name"
}
