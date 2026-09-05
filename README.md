# calibrated-models

An OpenAI-compatible endpoint for local GGUF models on one NVIDIA card, sized to the
machine it runs on.

The machine with the card runs one `llama-server` process — the **router**. It starts
with no model loaded; a request naming a model in its `model` field makes the router load
that model, keep it resident, and unload it after some minutes idle. How many stay
resident at once is `models_max`, one unless the settings file says otherwise, so
switching costs a load.

*Calibrated* is the point. How long a window a model can hold, how many of its experts
stay in system memory, how many threads it runs, how much memory the prompt cache may
have — none of that is a constant taken from somebody else's machine. It is worked out
against this card, this processor and this much memory, every time you run `calibrate`.

Nothing about a machine is in this repository. What it holds is a settings template, and
programs that read the machine.

---

## Ten commands

| run it as | on which machine | what it does |
|---|---|---|
| `python -m cm` | either | Prints the ten below in the order they are run, so that which one comes next is not something to open this file for |
| `python -m cm.install llamacpp` | the one with the card | Installs the newest llama.cpp release published for this machine's CUDA version, and removes the ones past keeping |
| `python -m cm.scan` | the one with the card | Adds an entry to the settings file for every model in the library that has none yet, named after the file, and brings every entry's sampler values up to date with `recommended.toml` in this repository. An entry marked `manual = true` is left alone; nothing is ever removed |
| `python -m cm.models` | the one with the card | The models the settings file names: whether the file is in the library, which repository it came from, and the commit that repository is at now. Downloads nothing |
| `python -m cm.calibrate` | the one with the card | Works out where each model in the settings file sits on this card, and writes `llamacpp.models.ini` — the preset the router reads. Loads nothing; it asks the estimator, which reads GGUF headers, so it takes seconds |
| `python -m cm.router start` | the one with the card | Runs the server of the newest release unpacked here, on the preset `calibrate` wrote, and reports what it serves. `stop` ends the server that is holding the configured port |
| `python -m cm.autostart` | the one with the card | Registers the scheduled task that starts the router at boot, or `--remove`s it. Needs an elevated session; stores no password |
| `python -m cm.firewall` | the one with the card | Admits the local subnet to the router's port, so other machines can call it, or `--remove`s that rule. Needs an elevated session |
| `python -m cm.vram` | the one with the card | What the router can load right now, and what to close so it can load the rest. A screen that keeps reading while you close things |
| `python -m cm.install pi` | any machine that calls it | Installs the pi agent and the context-policy extension, or brings both up to date |
| `python -m cm.pi <host>` | any machine that calls it | Points the pi agent at the router and rewrites its model list from what the router actually serves |

They share no state of their own. `scan` writes the entries `calibrate` places,
`calibrate` writes a file `router` hands to the
server, `install llamacpp` unpacks the releases both of them run, `autostart` and
`firewall` make the router something the network can reach without anybody logging in,
`install pi` puts the agent on the machine that calls, and `pi` asks the router over
HTTP and copies nothing from the machine with the card.

## What runs where

`calibrate` and `vram` read this machine through Windows: the card through `nvidia-smi`,
the installed memory through `GlobalMemoryStatusEx`, the cores and their efficiency
classes through `GetLogicalProcessorInformationEx`, and, for `vram`, video memory per
process through the performance counters. They say so and stop on any other system rather
than guessing.

`router`, `autostart` and `firewall` read no card, but they belong to that machine all
the same: `router` starts and stops Windows processes, `autostart` hands a task to the
Windows scheduler, and `firewall` asks netsh what the machine admits. The last two are
the only ones that need an elevated session, and where they do not have one they ask
Windows for it rather than half-doing the work.

`install`, `models` and `pi` have none of that. One runs npm and pi, one looks in the
library and asks Hugging Face what its repositories hold, and one speaks HTTP to the
router and writes two JSON files. All three run on **Windows, macOS and Linux** alike —
which is the shape of the thing: one machine holds the card, any number of machines call
it.

## Requirements

Python 3.11 or newer, and nothing else — no packages to install, standard library only.

On the machine with the card, additionally: an NVIDIA driver with `nvidia-smi` on the
path, and a [llama.cpp](https://github.com/ggml-org/llama.cpp) build carrying
`llama-server` and `llama-fit-params` -- which is what `install llamacpp` puts there,
so it is not something to arrange first.

On a machine that calls it: npm, if you want `install pi` to put pi there. Any other
OpenAI-compatible client works with no script at all.

Nothing here is installed. The programs run from the directory this repository is in,
and that directory is the whole of it: the llama.cpp releases go into `.llamacpp` inside
it, the settings file, the preset and the logs sit beside them. Nothing but the models
lives anywhere else.

## Running the programs

From the root of this repository, so that `cm` is importable:

```bash
python -m cm.calibrate
```

The interpreter is named differently from system to system, and that is the only
difference:

| | |
|---|---|
| Windows | `python -m cm.calibrate`, or `py -3 -m cm.calibrate` |
| macOS, Linux | `python3 -m cm.pi gpu-box` |

Every program takes `--help`, and all but `install pi` and `pi` take `--settings`: the
settings file to read. Those two are pointed at a router by its host name and read no
settings file at all. Left out, it is `settings.toml` in the directory you are standing
in, which is the whole reason the programs are run from the root of this repository.
Naming a file
elsewhere moves two things with it and leaves one where it is -- the preset and the logs
are looked for beside the file named, and the llama.cpp releases stay in `.llamacpp`
inside this repository, because that is where `install llamacpp` put them.

The rest of what they take, in full:

| command | flag | what it does |
|---|---|---|
| `install llamacpp` | `--check` | report what is installed and what is available, and download nothing |
| | `--force` | install even where the newest release is already here -- after an install that was interrupted or came out damaged |
| `router start` | `--print-command` | print what would be run, and start nothing |
| | `--foreground` | run the server in this console instead of detaching, so that whatever started this stays alive with it |
| | `--startup-check-seconds` | how long to wait before reporting what it serves. Five; `0` reports nothing and returns as soon as it is started |
| `autostart` | `--print-command` | print what would be registered, down to the task XML, and register nothing |
| | `--start-now` | start the task once it is registered, instead of waiting for the next boot |
| | `--remove` | remove the task, leaving the router to be started by hand |
| `firewall` | `--remove` | take the rule away, leaving the router reachable from this machine only |
| `install pi` | `--config` | pi's model configuration, where it is not `~/.pi/agent/models.json` |
| `pi <host>` | `--port` | the router's port, unless the address already carries one. 18081 |
| | `--config` | as above |
| | `--provider` | what to call the provider in pi's file, where none points at this router yet. `llamacpp-cuda` |
| | `--backups` | how many dated copies of pi's files to keep. Ten |
| | `--preview` | print what would be written, and write nothing |

`scan`, `models`, `calibrate`, `vram` and `router stop` take `--settings` and `--help`
and nothing else: there is nothing in them to vary.

`python -m cm` takes nothing at all. It prints every command in the order they are run,
which is the one thing `--help` on any single command cannot tell you.

---

## The whole path, on a machine that has none of it

**What has to be there first:** Python 3.11 or newer and an NVIDIA driver with
`nvidia-smi` on the path. llama.cpp does not: step 1 puts it there. Nor does the
repository need installing -- clone or unpack it anywhere and run the programs from that
directory, which is where everything they make will be.

The whole path is nine commands, and the rest of this section is what each one does.
`python -m cm` prints this list in the console, so it is not something to come back here
for:

```bash
python -m cm.install llamacpp         # 1. the settings file, and llama.cpp
python -m cm.scan                     # 2. an entry per GGUF in the library
python -m cm.models                   #    what is here, what is not, where from
python -m cm.calibrate                #    the preset, measured on this card
python -m cm.router start             # 3. serve
python -m cm.autostart                # 4. and at every boot        (elevated)
python -m cm.firewall                 #    and from the local network (elevated)
python -m cm.install pi               # 5. on whatever machine calls it
python -m cm.pi <the machine with the card>
```

**1. The settings file, and llama.cpp.**

```bash
python -m cm.install llamacpp
```

On a machine that has no `settings.toml` it copies one from `settings.template.toml`
beside it, and says so. Nothing in that copy has to be edited before the download, which
is what lets one command do both: the releases go into `.llamacpp` in this directory
whatever the file says, and every other key has a default or is asked for.

Then it downloads the newest release published for `cuda_version` and the CUDA runtime
beside it -- some 515 MB, 670 MB unpacked -- unpacks it into
`.llamacpp\b10754-cuda13.3`, and moves it into place only once the server in it reports
the build number it was downloaded as. `--check` first says what it would do and
downloads nothing.

Two keys in that copy are worth a look before the download rather than after.

`cuda_version` picks between the archives the project publishes -- one per CUDA version
-- so it has to be one this machine's driver supports. `nvidia-smi` prints the highest it
supports in its top right corner; take that or lower, and quote it, because `13.30`
unquoted is the same number as `13.3` and a different archive. The default is `13.3`.

`model_root` is the directory the weights sit under. Left commented out it falls back to
LM Studio's library, since that is where these files land anyway and LM Studio records
the folder it downloads into. Where LM Studio is not installed there is nothing to fall
back on and nothing on the machine to read it off, so `install` asks while it is making
the copy:

```
There is no LM Studio library at C:\Users\you\.lmstudio\models, so nothing on this
machine says where the models are.
Directory the models are under: D:\models
```

and writes that answer into the copy as `model_root`. A directory that is not there yet
is taken, with a line saying it will read as empty until it is. Where nobody is at the
keyboard -- a pipe, a script -- it refuses rather than guessing, and says to copy the
template by hand. Either way what the file says wins over the library.

**2. The models.** Nothing to type: `scan` writes an entry for every GGUF in the library
that has none yet, and keeps every entry's sampler values in step with what this
repository publishes for that model.

```bash
python -m cm.scan
```

The library is laid out the way LM Studio keeps it and the way a Hugging Face repository
unpacks -- publisher, repository, file -- and that says everything an entry needs. So
`file` is where the file sits, and the key is built from it: the model, the quantisation
of its weights, and the variant the repository directory carries where the file name
drops it.

```toml
[models.'qwen3.8-27b-ud-iq4xs']
file = 'unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ4_XS.gguf'

[models.'qwen3.8-27b-ud-iq4xs'.settings]
# qwen3.8-27b, over neutral defaults for whatever it leaves open:
# https://huggingface.co/Qwen/Qwen3.8-27B
# Change any of it here; manual = true above keeps scan out of this block for good.
min-p            = '0'
temp             = '1.0'
top-k            = '20'
top-p            = '0.95'
```

The key is what you will ask the router for, so it says which file answered: two
quantisations of one model are two names, and `Qwen3.8-27B-Q4_K_M.gguf` beside that one
is `qwen3.8-27b-q4km` rather than a coin toss. Rename it if you would rather type
something shorter -- an entry belongs to its `file`, not to its key, and `scan` will not
write a second one for a file that already has an entry.

Files that only sit beside a model are left out: `mmproj-*.gguf` is a vision projector,
unused here and about a gibibyte of video memory to pair, and `mtp-*.gguf` is a
prediction head, which this reads out of the model's own file instead.

### Where the sampler values come from

The numbers in that block are not invented per machine. They live in
[`recommended.toml`](recommended.toml) in this repository, one row per model, each row
saying where it was read from:

```toml
[recommended.'qwen3.8-27b']
source = 'https://huggingface.co/...'

[recommended.'qwen3.8-27b'.settings]
temp  = '1.0'
top-k = '20'
```

A row is matched on the model's **stem** -- the file name with the quantisation left
off -- so two quantisations of one model, and two publishers' repackagings of it, are
one row: `unsloth\Qwen3.8-27B-GGUF\...UD-IQ4_XS.gguf` and
`lmstudio-community\Qwen3.8-27B-GGUF\...Q4_K_M.gguf` are both `qwen3.8-27b`. A pattern
may hold a `*`, and the longest one that matches wins, so a row for a single release
sits inside a row for its family. Two matching patterns of the same length are refused
rather than ranked.

That is what makes the file worth having: what a model should run with is a property of
the model, not of a machine, so a correction is made once here and reaches every machine
that pulls it. Vendors usually publish two sets, one for general or chat use and one the
agentic and code benchmarks they report were run at; the second is the one taken, even
where it is the louder of the two — these models are run by an agent, not typed at.

A row holds only what its page states, so it can be read against that page line for
line. What the page leaves open is filled in from a neutral floor — `min-p 0`, the two
penalties off — rather than from llama.cpp's own `min-p 0.05`, `top-k 40` and
`temp 0.8`, which are not silence but somebody else's answer.

What happens to an entry when `scan` runs is decided by the entry:

| the entry | what happens |
|---|---|
| there is none for the file | one is added, with what the repository recommends |
| there is one | its settings block is brought up to date |
| there is one, marked `manual = true` | nothing at all |
| there is one, and its settings are written some other way | nothing at all, and the run says which entry and why |

The last is for a settings block `scan` cannot replace: one whose header carries a
comment after it, one written as an inline table or as dotted keys, one that sits under
some other table. Rewriting any of those would leave what is there and add a second
block besides, and a file that declares one table twice reads as no file at all — so the
entry is left exactly as it stands and named in what the run reports.

So change whatever you like. It stays changed until the next `scan`; mark the entry
`manual = true` and it stays changed for good. A model nothing in the repository covers
keeps whatever it has, and a new entry for one starts on neutral values rather than on
anybody's recommendation — the block says which of the two it is.

Everything between the settings block's header and whatever follows it belongs to
`scan` and is rewritten, the provenance comment included. A note of your own goes above
the header, where it survives. Nothing else in the file is touched: `scan` never removes
an entry, never reorders one, and never edits a key other than its settings.

Deleting the block does not leave a model neutral. It leaves it on llama.cpp's own
defaults -- `temp 0.8`, `top-k 40`, `min-p 0.05` -- which are nobody's recommendation
for these models and appear nowhere in the file to be disagreed with.

Nothing about the card goes in an entry: what window the model gets, what its attention
cache is held at and how much of it stays in system memory are measured by `calibrate`
below, and writing one of them here is refused rather than obeyed.

To take a model out of service, set it aside rather than deleting its entry:

```toml
[models.'qwen3.8-27b-q4km']
hidden = true
```

Nothing is placed or served for it, and `scan` still counts its file as named. Deleting
the entry instead would leave the file unnamed, and the next `scan` would write it back.
Both flags go **above** the settings block: TOML gives a bare key to the last header
above it, so `hidden` written under `[models.'x'.settings]` is a sampler value that
hides nothing. That is refused rather than ignored.

Then:

```bash
python -m cm.models
```

It lists what the settings file names -- whether the file is in the library, which
repository it came from, and the commit that repository is at now -- and downloads
nothing. A repository is re-uploaded under the same file names, so a file already here
and the one published now can differ without differing in size, and what settles it is
the commit. Which build to take is yours; fetch what is missing from the pages it
prints, keeping the `publisher\repository\file` layout the entries name. Then:

```bash
python -m cm.calibrate
```

It prints what it settled on for each and writes `llamacpp.models.ini` beside the
settings file. Nothing is loaded: it asks the estimator, which reads GGUF headers, so it
takes seconds per model.

**3. The router.**

```bash
python -m cm.router start
```

It runs the server of the newest release unpacked in `.llamacpp`, detached, with
the preset from step 2 and nothing else -- no model is loaded until a request names one.
Then it waits five seconds and reports what the router says it serves, so a start that
did not survive is a message rather than a silence. `--print-command` shows what would
be run and starts nothing; `python -m cm.router stop` stops what is serving.

It leaves four files in `log_dir` -- a `logs` directory beside the settings file
unless the file names another. `router.log` is the server's own log, which is where
a slow request or a model that would not load is explained; `router.stdout.log` and
`router.stderr.log` hold what it wrote before it had that log open, which is where a
build that will not start says why; and `router.pid` records the process this copy
started, for a person to read — nothing looks for it, because what has to go is decided
by the port and the executable rather than by a number written down earlier. A start
that does not survive prints the last lines of the first three itself, so the ordinary
way to read them is not to have to.

Starting takes priority over whatever is serving already: a start stops the router that
is running, the scheduled task included, because two of them cannot hold one port and
one card. Running it at boot is step 4.

**4. At boot, and reachable.** Two things the machine does rather than a person,
and the only two that need an elevated session:

```bash
python -m cm.autostart
python -m cm.firewall
```

Run them from an ordinary console. Where the session is not elevated they ask Windows
for the rights before reading or writing anything, so what you answer the prompt for is
the whole of the work rather than the second half of it. The elevated run happens with
no window of its own: what it wrote and the code it returned come back to the console
you are standing in. Declining the prompt changes nothing.

The first registers a scheduled task that starts the router at boot, and starts it
again if it fails. It runs as the account it was registered from, with no password
stored anywhere -- S4U, which is all a service needs that reads local files and
binds a local port -- and asks for no elevation of its own at run time.
`--print-command` prints the task it would register, down to the XML, and registers
nothing.

The second admits the local subnet to the router's port. Windows turns away an
inbound connection to a newly bound port, so until that rule exists a router
listening on every address answers this machine and nothing else. The local subnet
and not any address: nothing stands in front of this endpoint, and whoever reaches
it can load models, spend the card and read whatever a conversation carries.

Both take `--remove`.

**5. The client.** On the machine that will call it -- which may be this one:

```bash
python -m cm.install pi
python -m cm.pi <the machine with the card>
```

The first installs pi -- the npm package `@earendil-works/pi-coding-agent` -- or
updates it if it is already there, and installs the context-policy extension with
pi's own installer. Unless a git checkout of it is
already where pi loads extensions from: pi loads that instead of the package, whoever put
it there did so in order to edit it, and it is left alone rather than taken over.

The second lists what the router serves, rewrites that provider in
`~/.pi/agent/models.json`, and puts the two compaction numbers in `settings.json`
beside it behind what the
[context-policy](https://github.com/somewhat-useful/context-policy) extension will
compute -- see below. Every other provider in that file is left exactly as it was, and a
dated backup is kept. `--preview` prints what it would write and writes nothing.

Any other OpenAI-compatible client needs no script at all: point it at
`http://<the machine>:18081/v1`.

---

## Afterwards

**Another model.** Add its entry, fetch the file, and rebuild the preset. The preset is
handed to the router when it starts, so start it again to have the new model offered:

```bash
python -m cm.models
python -m cm.calibrate
python -m cm.router start
```

**A newer llama.cpp.**

```bash
python -m cm.install llamacpp
python -m cm.calibrate
python -m cm.router start
```

The first installs whatever the project has published since, carrying the CUDA runtime
over from the release already there rather than downloading four hundred megabytes of it
again, and removes the releases past `keep_releases` -- except one a server is running
from. The highest build number present is what runs, so a start is what puts the router
on it.

`calibrate` again because the placements were measured by asking that build's
`llama-fit-params` what fits. A build that packs its buffers differently answers
differently, and the preset would still be holding the old answer. It costs seconds and
loads nothing.

**Which models are served right now**, without any client:

```bash
curl http://localhost:18081/v1/models
```

**Stopping.** `python -m cm.router stop` ends the process that is **both** running
`llama-server.exe` **and** listening on the port the settings file names. Both, because
neither on its own is a reason to end anything:

- a server on a port of its own is serving somebody. Another copy of these scripts, or a
  worker the router started — which runs the same executable and is its child;
- something on this port that is not the server is somebody's program, and ending it is
  your decision rather than a start's. It is reported instead.

No pid file comes into it: a number written down earlier says nothing about what is
holding anything now. The port is read from the settings file, and the listening sockets
from Windows directly rather than off `netstat`, whose state words are in the language
Windows was installed in — a port read as free because a word did not match in English
is a router that starts and cannot bind.

Afterwards the port is checked again, and this time for **anything at all**: what a
start needs is to bind, and a port held by something nobody aimed at stops it just as
hard. If something still has it, `stop` says so and names it rather than reporting
success.

A start does the same clearing before it runs, which is why two routers cannot end up
on one port. `stop` also ends the scheduled task first, since ending only the server
would have the task start another. The task itself stays registered, so the router comes
back at the next boot.

**Taking it off the machine.** `autostart --remove` and `firewall --remove`, both
elevated, undo what needed elevation to do. Everything else is files, and all of them
are in this directory: `.llamacpp`, the preset, the logs and the settings file. On a
machine that calls, `~/.pi` if `install pi` put pi there. `npm uninstall -g @earendil-works/pi-coding-agent` removes pi itself.

## Watching the card

`calibrate` measures against the card's total, so what it decides does not move with what
happens to be open -- the preset is the same whether it was built with a browser running
or not. Whether a model fits *right now* is the other question, and its answer changes
every time a window opens:

```bash
python -m cm.vram
```

A screen rather than a report. It reads the card again on every keystroke and once a
second with nothing pressed, so closing something is watched happening rather than asked
about afterwards:

```
NVIDIA GeForce RTX 4080 SUPER   16376 MiB   in use 13996   free 2380
  the router holds 9550 MiB and hands it back when it loads: 11930 to place in

> qwen3.8-45k-q8                13890 MiB   short by 1960 MiB
  qwen3.8-88k-q4                12640 MiB   short by 710 MiB
  gemma4-12b                     9310 MiB   loads now, 2620 MiB to spare

  [x] chrome                          9184    3480 MiB
  [ ] Code                           21556    1960 MiB
  [ ] Discord                         4472     620 MiB

  the other 1420 MiB is Windows, the compositor and this window -- not on offer
  310 MiB of that is this window, and comes back when it is closed
  marked 1 program, 3480 MiB -- enough by these figures
  [tab] next model   [shift-tab] back   [space] mark   [enter] close marked   [r] refresh   [q] quit
```

The first line is the card. The second is there only while the router is holding a model:
that memory comes back the moment it is asked for a different one, so what a profile has
to fit into is the free memory plus what the router is holding -- `11930` here, and that
is the figure the list is measured against, not the `2380` the card calls free.

Then every profile the preset offers, in the order the preset lists them, with what each
holds once loaded. The screen opens on the first one that does not fit: a profile that
already loads leaves nothing to decide.

Below them, what is holding the card, largest first, with each program's pid and a box.
The boxes start filled in -- `vram` proposes a browser first and then whatever holds the
most, taking programs until the shortfall is covered and not one further. `[space]` adds
or removes one, `[tab]` moves to another profile and proposes a fresh set for that one,
`[r]` reads everything again, and nothing at all is closed until `[enter]`.

The two lines under the list account for the rest of the card. What Windows, the
compositor and this console hold is a number rather than rows, because none of it is
something a person can give up -- except the part belonging to this window, which is said
out loud, because it does come back once the model is loaded and the window is closed.

`[enter]` closes whole programs, and only ones a person could close by hand: a program
with a window is asked to close it, exactly as clicking its corner would, and one holding
the card with nothing on screen is ended instead -- a hidden window frees nothing when it
is asked and does not answer. Each is given five seconds to go, and one that does not
go is on the list again at the next draw, still open -- nothing is taken on trust,
the card is simply read again.

## When it refuses

Every program refuses in the same shape: one line saying what it will not do, and the
command that deals with it. The ones worth knowing in advance:

| it says | it means |
|---|---|
| `settings.toml not found at ...` | you are not in the directory the settings file is in, or `install llamacpp` has not run to make one |
| `model_root is not set and there is no LM Studio library at ...` | uncomment `model_root` and name the directory the weights are under |
| `No release carries llama-b<number>-bin-win-cuda-<version>-x64.zip` | `cuda_version` names a flavour the project no longer builds. It prints the newest tags it saw |
| `No llama.cpp release under ... carries llama-server.exe` | step 1 has not happened: nothing is unpacked in `.llamacpp` yet |
| `The preset the router reads was not found: ...` | `calibrate` has not run since the settings file changed |
| `qwen3.8: ctx-size is derived; remove it` | a placement was written into the settings file by hand. Those are measured, and one written down would be obeyed silently and wrongly |
| `Port N is still held by process M` | either a router this session may not signal — one started by the scheduled task, under another logon session — or something that is not `llama-server.exe` and was left alone deliberately. It prints the elevated `Stop-Process` line either way |
| `The prompt was declined, so nothing was changed` | `autostart` or `firewall` asked Windows for the rights it needs and the prompt was answered no. Run it again and accept, or run it from a session that is already elevated |
| `pi was not found, and neither was npm` | `install pi` needs Node.js. It prints the winget line |

---

## The settings file

`settings.toml` is yours. `install llamacpp` copies it from the template where there is
none, and nothing writes it after that. It holds three kinds of thing:

**Where things are.** `model_root`, and `preset_path` — the file `calibrate` writes.
Where llama.cpp is is not among them: the releases are in `.llamacpp` beside this file.

**Which llama.cpp.** `cuda_version` picks between the archives the project publishes for
each CUDA version, and `keep_releases` says how many unpacked releases to keep when
`install llamacpp` installs a newer one. Both have a default — `13.3` and two —
and the file only has to name them to say something else.

**How the router runs.** `listen_host` and `port` are what it binds, `models_max` how
many models may be resident at once, `sleep_idle_seconds` how long an idle one is kept
before the card is given back, and `log_dir` where it writes. All five have defaults —
every address, 18081, one, a quarter of an hour, and a `logs` directory beside this
file — and a machine serving one card has no reason to name any of them.

**Judgements about the work**, which no machine can make for you:

| key | what it decides |
|---|---|
| `reserve_mib` | video memory to leave for everything that is not a model |
| `min_ctx_tokens` | below which a window stops being worth serving |
| `ample_ctx_tokens` | past which a coarser attention cache buys nothing worth having |
| `cache_ram` | how much system memory the prompt cache may hold |

`cache_ram` is written the way anyone would write it — `32`, `32G`, `32Gb`, `32GiB` for a
size in gibibytes, `50%` for a share of the memory installed. Leave it out and it is
worked out: what the machine has, less the weights the heaviest profile keeps off the
card, less a share for the system. Half the memory, which is what an older version of this
used, is not the answer on a machine whose job is serving models with nobody logged in.

**The models**: one entry per GGUF, written and kept up to date by
[`scan`](#running-the-programs) and yours to change afterwards. Everything beyond `file`
is optional — the sampler settings, which come from `recommended.toml` in this
repository, `cache` and `mtp` to narrow which profiles `calibrate` may offer,
`hidden = true` to set a model aside without deleting the entry that names its file, and
`manual = true` to keep `scan` out of that entry's settings for good.

What may **not** be in it: `ctx-size`, `cache-type-k`, `cache-type-v`, `gpu-layers`,
`n-cpu-moe`, the `spec-*` keys, `threads` and `cache-ram`. Those are worked out against
this machine, and one written by hand is refused rather than obeyed — a placement quietly
overridden is the failure this program exists to prevent, and it would not even look like
one: the router would start, serve the model, and hold a different window than the card was
measured for.

The settings file is yours and is not in the repository. The template is.

## What the machine decides

| what | from |
|---|---|
| the window each profile holds, and the attention cache it holds it at | asked of `llama-fit-params` against the card's total memory, searched until the placement lands near `reserve_mib` |
| how many expert layers of a mixture stay in system memory | the same search, along the other lever |
| `threads` and `threads-batch` | the logical processors of the fastest cores. Not all of them: the thread pool is synchronised by a barrier, so work spread onto slow cores is work the rest wait for |
| `cache-ram` | the memory installed, less what the heaviest placement keeps outside the card, less a share for the system |

A placement is never narrowed to fit around a browser: what fits is worked out against the
card's total, so it does not matter what is open while `calibrate` runs. Whether a model
fits *right now* is a different question, and that is what
[`vram`](#watching-the-card) is for.

## Calling it

```
http://<server>:18081/v1
```

The API key is ignored. The model id is a profile name from `llamacpp.models.ini`, and the
list of them is at `/v1/models`.

```bash
curl http://<server>:18081/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"<id>\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}"
```

**There is no authentication.** Keep it on a network you trust.

A model that reasons returns its trace in `message.reasoning_content`, so a client with a
tight `max_tokens` can get an empty `content`: the budget went on thinking. That is why
`pi` is given a reply cap per model rather than one number for all of them.

## Where a conversation gets cut

pi compacts on two global numbers, and one pair cannot suit windows running from 28k to
262k. The context-policy extension computes both per model instead, from the window and
the reply cap the model is listed with — which is what `pi` writes into pi's model list.

But an extension cannot switch pi's own compaction off, only get there first. So the
policy governs a model only while pi's own reserve stays *under* the one the policy
computed, and pi keeps no more than the policy would keep. `python -m cm.pi` works
out the smallest of each across everything the router serves and reports it:

```
The context-policy extension will hold back at least 8400 tokens and keep at most 6860,
over these models. pi's own compaction has to stay behind that.
```

Numbers already behind that are left alone — they are also pi's fallback when summarising
fails, so there is no reason to sit one token off the line. Only a number that would take
the extension out of the decision is corrected. `/context-policy` inside pi prints the
same figures per model and flags a mismatch with a line starting `!`.

## Tests

```bash
python -m unittest discover -s tests -t tests -q
```

No network, no card and no files of yours are touched: everything the machine says arrives
as text written into the tests.
