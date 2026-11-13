<#
.SYNOPSIS
  SEI on this machine: set up, start, stop, ask.

.DESCRIPTION
  sei setup [-ToolsDir D:\sei-tools]   Python venv, Neo4j, and the local AI models.
  sei start <project folder>           Start Neo4j, the models and the SEI server; open the console.
  sei stop                             Stop the SEI server and Neo4j.
  sei status                           What is running, and what the server reports.
  sei ask "question"                   Ask the project `sei start` was given, from the terminal.
  sei index                            Re-index that project from scratch (Sync Project).

  Everything large (Neo4j, Ollama and its models, downloads, logs) lives in the
  tools folder, never in the repository and never on C: unless you put it there.
  The folder is remembered in .local\config.json.

  Works on Windows PowerShell 5.1: no &&, no ternaries, no null-coalescing.
#>
param(
    [Parameter(Position = 0)] [string] $Command = "help",
    [Parameter(Position = 1)] [string] $Target = "",
    [string] $ToolsDir = "",
    [string] $Repo = "",
    [int] $Port = 8000,
    [switch] $NoBrowser,
    [switch] $NoModels
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LocalDir = Join-Path $Root ".local"
$ConfigFile = Join-Path $LocalDir "config.json"

$Neo4jVersion = "5.26.0"            # pinned, as docker-compose.yml pins it
$Neo4jZip = "neo4j-community-$Neo4jVersion-windows.zip"
$Neo4jUrl = "https://dist.neo4j.org/$Neo4jZip"
$Neo4jPassword = "sei-dev-password"
$AnswerModel = "qwen2.5-coder:3b"   # config.LOCAL_LLM_MODEL
$EmbedModel = "nomic-embed-text"    # config.LOCAL_EMBED_MODEL

# --------------------------------------------------------------------------
# Where things live
# --------------------------------------------------------------------------

function Read-Config {
    if (Test-Path $ConfigFile) { return Get-Content $ConfigFile -Raw | ConvertFrom-Json }
    return $null
}

function Get-ToolsDir {
    if ($ToolsDir -ne "") { return $ToolsDir }
    if ($env:SEI_TOOLS_DIR) { return $env:SEI_TOOLS_DIR }
    $config = Read-Config
    if ($config -and $config.tools_dir) { return ($config.tools_dir -replace "/", "\") }
    return (Join-Path $LocalDir "tools")
}

function Save-Config([string] $tools) {
    New-Item -ItemType Directory -Force $LocalDir | Out-Null
    @{ tools_dir = ($tools -replace "\\", "/") } | ConvertTo-Json | Out-File $ConfigFile -Encoding ascii
}

$Tools = Get-ToolsDir
$Neo4jHome = Join-Path $Tools "neo4j\neo4j-community-$Neo4jVersion"
# A second, disposable server for the test suite, which wipes what it uses.
$Neo4jTestHome = Join-Path $Tools "neo4j-test\neo4j-community-$Neo4jVersion"
$TestBoltPort = 7688
$OllamaDir = Join-Path $Tools "ollama"
$ModelsDir = Join-Path $Tools "ollama-models"
$LogDir = Join-Path $Tools "logs"
$RunDir = Join-Path $Tools "run"
$ServerState = Join-Path $RunDir "server.json"

function Say([string] $text) { Write-Host "sei: $text" }

function New-Token {
    # The server answers only requests carrying this run's token (api/cli.py).
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    return ([Convert]::ToBase64String($bytes) -replace '\+', '-' -replace '/', '_').TrimEnd('=')
}

function Get-AuthHeaders($state) {
    if ($state -and $state.token) { return @{ Authorization = "Bearer $($state.token)" } }
    return @{}
}

function Test-Port([int] $port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $attempt = $client.BeginConnect("127.0.0.1", $port, $null, $null)
        $open = $attempt.AsyncWaitHandle.WaitOne(500) -and $client.Connected
        return $open
    } catch { return $false } finally { $client.Close() }
}

function Wait-Port([int] $port, [int] $seconds, [string] $what) {
    for ($i = 0; $i -lt $seconds; $i++) {
        if (Test-Port $port) { return $true }
        Start-Sleep -Seconds 1
    }
    Say "$what did not answer on port $port within $seconds s."
    return $false
}

function Stop-PortOwner([int] $port, [string] $what) {
    $owners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($owner in $owners) {
        if ($owner -and $owner -ne 0) {
            Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            Say "stopped $what (pid $owner)."
        }
    }
    if ($owners.Count -eq 0) { Say "$what was not running." }
}

function Find-Ollama {
    $local = Join-Path $OllamaDir "ollama.exe"
    if (Test-Path $local) { return $local }
    $onPath = Get-Command ollama -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

function Invoke-Quiet([string] $exe, [string[]] $arguments) {
    # Windows PowerShell 5.1 turns a native program's stderr into error records
    # the moment it is redirected, and "Stop" then aborts the script on what
    # was only a banner. Java prints its settings to stderr; so does ollama.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { return @(& $exe @arguments 2>&1 | ForEach-Object { "$_" }) }
    catch { return @() }
    finally { $ErrorActionPreference = $previous }
}

function Find-Java {
    if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME "bin\java.exe"))) { return $env:JAVA_HOME }
    $java = Get-Command java -ErrorAction SilentlyContinue
    if (-not $java) { return $null }
    # A shim on PATH (Oracle's javapath) hides the real home; ask the JVM.
    $homeLine = Invoke-Quiet $java.Source @("-XshowSettings:properties", "-version") |
        Where-Object { $_ -match "java\.home" } | Select-Object -First 1
    if ($homeLine) { return ($homeLine -split "=", 2)[1].Trim() }
    return $null
}

# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

function Invoke-Setup {
    Save-Config $Tools
    foreach ($dir in @($Tools, $LogDir, $RunDir, $ModelsDir, (Join-Path $Tools "downloads"))) {
        New-Item -ItemType Directory -Force $dir | Out-Null
    }
    Say "tools folder: $Tools"

    # -- Python ------------------------------------------------------------
    if (-not (Test-Path $Python)) {
        Say "creating the Python environment (.venv)..."
        python -m venv (Join-Path $Root ".venv")
    }
    Say "installing Python packages..."
    & $Python -m pip install -q --disable-pip-version-check -r (Join-Path $Root "requirements.txt")

    # -- Neo4j -------------------------------------------------------------
    if (-not (Find-Java)) {
        Say "Java was not found. Neo4j $Neo4jVersion needs Java 17 or 21 (newer versions run with a warning)."
        Say "Install a JDK, then run 'sei setup' again."
        exit 1
    }
    if (-not (Test-Path (Join-Path $Neo4jHome "bin\neo4j.bat"))) {
        $downloads = Join-Path $Tools "downloads\neo4j"
        New-Item -ItemType Directory -Force $downloads | Out-Null
        $zip = Join-Path $downloads $Neo4jZip
        if (-not (Test-Path $zip)) {
            Say "downloading Neo4j $Neo4jVersion (about 160 MB) from dist.neo4j.org..."
            Invoke-WebRequest -Uri $Neo4jUrl -OutFile $zip -UseBasicParsing
        }
        $published = (Invoke-WebRequest -Uri "$Neo4jUrl.sha256" -UseBasicParsing).Content
        if ($published -is [byte[]]) { $published = [System.Text.Encoding]::ASCII.GetString($published) }
        $published = $published.Trim().Split(" ")[0]
        $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $published.ToLower()) {
            Say "Neo4j download does not match its published SHA-256. Delete $zip and try again."
            exit 1
        }
        Say "checksum verified; extracting..."
        Expand-Archive -Path $zip -DestinationPath (Join-Path $Tools "neo4j") -Force
        Add-Content -Path (Join-Path $Neo4jHome "conf\neo4j.conf") -Encoding ascii -Value @"

# --- SEI local server (mirrors docker-compose.yml) ---
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1G
server.memory.pagecache.size=512m
server.config.strict_validation.enabled=false
"@
        $env:JAVA_HOME = Find-Java
        & (Join-Path $Neo4jHome "bin\neo4j-admin.bat") dbms set-initial-password $Neo4jPassword | Out-Null
        Say "Neo4j installed."
    } else {
        Say "Neo4j is already installed."
    }
    Install-Neo4jTest

    # -- local models --------------------------------------------------------
    if ($NoModels) { Say "skipping local models (-NoModels)."; return }
    $ollama = Find-Ollama
    if (-not $ollama) {
        Say "Ollama is not installed. For local answers and search, install it from https://ollama.com/download"
        Say "  (to keep it off C:, run:  OllamaSetup.exe /DIR=`"$OllamaDir`")"
        Say "  and set the user variable OLLAMA_MODELS=$ModelsDir before its first start."
        Say "Then run 'sei setup' again. Without it SEI still runs, with keyword and graph search only."
        return
    }
    Start-Ollama $ollama | Out-Null
    foreach ($model in @($EmbedModel, $AnswerModel)) {
        Say "pulling $model (skipped if already present)..."
        & $ollama pull $model
    }
    Say "setup complete. Next:  sei start <path to your project>"
}

# --------------------------------------------------------------------------
# start / stop
# --------------------------------------------------------------------------

function Install-Neo4jTest {
    # The same distribution without its data, listening on 7688/7475. The
    # suite refuses to wipe a database holding someone's data; this one never does.
    if (Test-Path (Join-Path $Neo4jTestHome "bin\neo4j.bat")) { Say "test Neo4j is already installed."; return }
    New-Item -ItemType Directory -Force $Neo4jTestHome | Out-Null
    Get-ChildItem $Neo4jHome | Where-Object { $_.Name -notin @("data", "logs", "run") } |
        ForEach-Object { Copy-Item -Recurse -Force $_.FullName $Neo4jTestHome }
    foreach ($dir in @("data", "logs", "run")) { New-Item -ItemType Directory -Force (Join-Path $Neo4jTestHome $dir) | Out-Null }
    $conf = Join-Path $Neo4jTestHome "conf\neo4j.conf"
    $text = (Get-Content $conf -Raw) -replace "(?s)\r?\n# --- SEI test server.*$", ""
    Set-Content -Path $conf -Encoding ascii -Value ($text + @"

# --- SEI test server: a second instance so pytest never touches the app's data ---
server.bolt.listen_address=:$TestBoltPort
server.bolt.advertised_address=:$TestBoltPort
server.http.listen_address=:7475
server.http.advertised_address=:7475
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1G
server.memory.pagecache.size=256m
"@)
    $env:JAVA_HOME = Find-Java
    & (Join-Path $Neo4jTestHome "bin\neo4j-admin.bat") dbms set-initial-password $Neo4jPassword | Out-Null
    Say "test Neo4j installed (port $TestBoltPort)."
}

function Start-Neo4jTest {
    if (Test-Port $TestBoltPort) { return $true }
    if (-not (Test-Path (Join-Path $Neo4jTestHome "bin\neo4j.bat"))) {
        Say "the test Neo4j is not installed. Run 'sei setup' (or: docker compose up -d neo4j-test)."
        return $false
    }
    $env:JAVA_HOME = Find-Java
    New-Item -ItemType Directory -Force $LogDir | Out-Null
    Say "starting the test Neo4j..."
    Start-Process -FilePath (Join-Path $Neo4jTestHome "bin\neo4j.bat") -ArgumentList "console" `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "neo4j-test.log") `
        -RedirectStandardError (Join-Path $LogDir "neo4j-test.err.log") | Out-Null
    return (Wait-Port $TestBoltPort 120 "The test Neo4j")
}

function Invoke-Test([string[]] $extra) {
    if (-not (Start-Neo4jTest)) { exit 1 }
    $env:SEI_REQUIRE_NEO4J = "1"
    $env:SEI_TEST_NEO4J_URI = "bolt://localhost:$TestBoltPort"
    Push-Location $Root
    try { & $Python -m pytest tests/ @extra; exit $LASTEXITCODE } finally { Pop-Location }
}

function Start-Neo4j {
    if (Test-Port 7687) { Say "Neo4j is already running."; return $true }
    if (-not (Test-Path (Join-Path $Neo4jHome "bin\neo4j.bat"))) {
        Say "Neo4j is not installed in $Tools. Run 'sei setup' first (or start your own: docker compose up -d)."
        return $false
    }
    $javaHome = Find-Java
    if (-not $javaHome) { Say "Java was not found; Neo4j cannot start."; return $false }
    $env:JAVA_HOME = $javaHome
    New-Item -ItemType Directory -Force $LogDir | Out-Null
    Say "starting Neo4j..."
    Start-Process -FilePath (Join-Path $Neo4jHome "bin\neo4j.bat") -ArgumentList "console" `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "neo4j.log") `
        -RedirectStandardError (Join-Path $LogDir "neo4j.err.log") | Out-Null
    return (Wait-Port 7687 120 "Neo4j")
}

function Start-Ollama([string] $ollama) {
    if (Test-Port 11434) { return $true }
    $env:OLLAMA_MODELS = $ModelsDir
    New-Item -ItemType Directory -Force $LogDir | Out-Null
    Say "starting the local model server..."
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "ollama.log") `
        -RedirectStandardError (Join-Path $LogDir "ollama.err.log") | Out-Null
    return (Wait-Port 11434 60 "The model server")
}

function Invoke-Start {
    if ($Target -eq "") { Say "which project?  sei start <path to your project>"; exit 2 }
    if (-not (Test-Path $Target -PathType Container)) { Say "'$Target' is not a folder."; exit 2 }
    if (-not (Test-Path $Python)) { Say "no .venv yet. Run 'sei setup' first."; exit 1 }
    $project = (Resolve-Path $Target).Path
    $repoId = $Repo
    if ($repoId -eq "") { $repoId = (Split-Path $project -Leaf) -replace "[^A-Za-z0-9_.-]", "_" }

    if (Test-Port $Port) {
        Say "port $Port is already in use. Run 'sei stop' first, or pick another with -Port."
        exit 1
    }
    if (-not (Start-Neo4j)) { exit 1 }

    $ollama = Find-Ollama
    if ($ollama) { Start-Ollama $ollama | Out-Null }
    else { Say "Ollama not found: running without local models (keyword and graph search, no written answers)." }

    New-Item -ItemType Directory -Force $RunDir | Out-Null
    $out = Join-Path $LogDir "sei.log"
    $err = Join-Path $LogDir "sei.err.log"
    Say "starting SEI for '$repoId' ($project)..."
    $env:PYTHONIOENCODING = "utf-8"
    $token = New-Token
    $env:SEI_API_TOKEN = $token
    $server = Start-Process -FilePath $Python -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -ArgumentList @("-m", "api.cli", "--repo", $repoId, "--root", "`"$project`"", "--serve", "--watch", "--port", "$Port") `
        -RedirectStandardOutput $out -RedirectStandardError $err
    @{ pid = $server.Id; repo = $repoId; root = $project; port = $Port; token = $token } |
        ConvertTo-Json | Out-File $ServerState -Encoding ascii

    if (-not (Wait-Port $Port 90 "SEI")) {
        Say "it did not start. Last lines of ${err}:"
        if (Test-Path $err) { Get-Content $err -Tail 15 }
        exit 1
    }
    $url = "http://127.0.0.1:$Port/?token=$token"
    Say "running at $url"
    Say "the first index of a project, and the first answer after a boot, take a minute or two:"
    Say "the page header shows 'indexing...' until it is ready.  Logs: $LogDir"
    if (-not $NoBrowser) { Start-Process $url }
}

function Invoke-Stop {
    $state = $null
    if (Test-Path $ServerState) { $state = Get-Content $ServerState -Raw | ConvertFrom-Json }
    $serverPort = $Port
    if ($state -and $state.port) { $serverPort = [int] $state.port }
    Stop-PortOwner $serverPort "SEI"
    if (Test-Path $ServerState) { Remove-Item $ServerState -Force }
    if ((Test-Port 7687) -and (Test-Path $Python)) {
        # Neo4j run in a console has no stop command: its process is ended. A
        # checkpoint first leaves no transaction log to replay at the next start.
        Push-Location $Root
        try {
            Invoke-Quiet $Python @("-c", "import config; from neo4j import GraphDatabase; d = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)); d.execute_query('CALL db.checkpoint()'); d.close()") | Out-Null
        } finally { Pop-Location }
    }
    Stop-PortOwner 7687 "Neo4j"
    if (Test-Port $TestBoltPort) { Stop-PortOwner $TestBoltPort "test Neo4j" }
    $ollama = Find-Ollama
    if ($ollama -and (Test-Port 11434)) {
        # Free the GPU; the model server itself is left for other programs.
        Invoke-Quiet $ollama @("stop", $AnswerModel) | Out-Null
        Invoke-Quiet $ollama @("stop", $EmbedModel) | Out-Null
        Say "unloaded the local models (the model server stays available)."
    }
}

function Get-State {
    if (-not (Test-Path $ServerState)) { Say "SEI is not running. Start it with:  sei start <project>"; exit 1 }
    return Get-Content $ServerState -Raw | ConvertFrom-Json
}

function Invoke-Status {
    $neo = "stopped"; if (Test-Port 7687) { $neo = "running" }
    $models = "stopped"; if (Test-Port 11434) { $models = "running" }
    Say "tools folder : $Tools"
    Say "Neo4j        : $neo"
    Say "model server : $models"
    if (-not (Test-Path $ServerState)) { Say "SEI server   : stopped"; return }
    $state = Get-Content $ServerState -Raw | ConvertFrom-Json
    if (-not (Test-Port ([int] $state.port))) { Say "SEI server   : stopped (stale state for $($state.repo))"; return }
    Say "SEI server   : http://127.0.0.1:$($state.port)  ($($state.repo): $($state.root))"
    if ($state.token) { Say "console      : http://127.0.0.1:$($state.port)/?token=$($state.token)" }
    try {
        $status = Invoke-RestMethod -Uri "http://127.0.0.1:$($state.port)/status" -TimeoutSec 10 `
            -Headers (Get-AuthHeaders $state)
        Say "answers      : $($status.llm)"
        Say "search       : $($status.embeddings)  (vector search: $($status.vector_search))"
        if ($status.graph) {
            Say "index        : $($status.graph.files) files, $($status.graph.symbols) symbols, $($status.graph.relationships) links"
        }
        if ($status.watcher) { Say "auto-index   : $($status.watcher.state)" }
    } catch { Say "the server did not answer /status: $($_.Exception.Message)" }
}

function Invoke-Cli([string[]] $extra) {
    $state = Get-State
    $env:PYTHONIOENCODING = "utf-8"
    & $Python -m api.cli --repo $state.repo --root $state.root @extra
}

function Invoke-Index {
    # The running server owns the index: it holds the per-repository lock that
    # keeps a re-index and a save from interleaving. A second process writing
    # the same graph would not see that lock, so the request goes to the server.
    $state = Get-State
    if (-not (Test-Port ([int] $state.port))) {
        Say "the server is not running. Start it with:  sei start `"$($state.root)`""
        exit 1
    }
    Say "re-indexing $($state.repo): re-reading every file and re-linking every edge..."
    $body = @{ repo = $state.repo } | ConvertTo-Json
    $started = Get-Date
    $reply = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$($state.port)/index/sync" `
        -ContentType "application/json" -Body $body -TimeoutSec 3600 -Headers (Get-AuthHeaders $state)
    if (-not $reply.accepted) { Say "refused: $($reply.reason)"; exit 1 }
    Say ("done in {0:N0} s." -f ((Get-Date) - $started).TotalSeconds)
    Invoke-Status
}

switch ($Command.ToLower()) {
    "setup"  { Invoke-Setup }
    "start"  { Invoke-Start }
    "stop"   { Invoke-Stop }
    "status" { Invoke-Status }
    "ask"    {
        if ($Target -eq "") { Say 'ask what?  sei ask "what calls main"'; exit 2 }
        Push-Location $Root; try { Invoke-Cli @($Target) } finally { Pop-Location }
    }
    "index"  { Invoke-Index }
    "test"   { if ($Target -ne "") { Invoke-Test @($Target) } else { Invoke-Test @() } }
    default  {
        Write-Host @"
SEI - ask questions about a codebase, on this machine.

  sei setup                    install what is missing (Python packages, Neo4j, local models)
  sei start <project folder>   start everything and open the console in the browser
  sei status                   what is running, and what the index holds
  sei ask "question"           ask the running project from the terminal
  sei index                    re-index the running project from scratch
  sei stop                     stop the server and Neo4j, unload the models
  sei test ["pytest args"]     run the test suite against its own Neo4j (port 7688)

Options:  -ToolsDir <folder>  where large files live (now: $Tools)
          -Port <n>           console port (default 8000)      -NoBrowser   do not open the page
          -Repo <name>        index name (default: the folder's name)
"@
    }
}
