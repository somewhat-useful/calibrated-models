# calibrated-models

An OpenAI-compatible endpoint for local GGUF models on the NVIDIA cards of one machine,
and on a card another machine lends it over the network, sized to the machines they
are in.

The machine with the card runs one `llama-server` process — the **router**. It starts
with no model loaded; a request naming a model in its `model` field makes the router load
that model, keep it resident, and unload it after some minutes idle. How many stay
resident at once is `models_max`, one unless the settings file says otherwise, so
switching costs a load.

*Calibrated* is the point. How long a window a model can hold, how many of its experts
stay in system memory, how many threads it runs, how much memory the prompt cache may
have — none of that is a constant taken from somebody else's machine. It is worked out
against these cards, this processor and this much memory, every time you run
`calibrate`.

Nothing about a machine is in this repository. What it holds is a settings template, and
programs that read the machine.

---

## Thirteen commands

| run it as | on which machine | what it does |
|---|---|---|
| `python -m cm` | either | Prints the thirteen below in the order they are run, so that which one comes next is not something to open this file for |
| `python -m cm.install llamacpp` | the one with the card, and one lending its card | Installs the newest llama.cpp release that runs on this machine's driver and cards -- or, with `--build`, the build named -- records it in the settings file as the build everything here runs, and removes the releases past keeping. `--check` says which build that would be |
| `python -m cm.scan` | the one with the card | Adds an entry to the settings file for every model in the library that has none yet, named after the file, and brings every entry's sampler values up to date with `recommended.toml` in this repository. An entry marked `manual = true` is left alone; an entry whose file is gone is taken out, whatever else it says. `--force` also keys every entry the way the library names its file |
| `python -m cm.models` | the one with the card | The models the settings file names: whether the file is in the library, which repository it came from, and the commit that repository is at now. Downloads nothing |
| `python -m cm.calibrate` | the one with the card | Works out where each model in the settings file sits on this card, and writes `llamacpp.models.ini` — the preset the router reads. Loads nothing; it asks the estimator, which reads GGUF headers, so it takes seconds |
| `python -m cm.router start` | the one with the card | Runs the server of the build the settings file records, on the preset `calibrate` wrote, and reports what it serves. `stop` ends the server that is holding the configured port |
| `python -m cm.autostart` | the one with the card | Registers the scheduled task that starts the router at boot, or `--remove`s it. Needs an elevated session; stores no password |
| `python -m cm.firewall` | the one with the card | Admits the local subnet to the router's port, so other machines can call it, or `--remove`s that rule. Needs an elevated session |
| `python -m cm.vram` | the one with the card | What the router can load right now, and what to close so it can load the rest. A screen that keeps reading while you close things |
| `python -m cm.install pi` | any machine that calls it | Installs the pi agent and the context-policy extension, or brings both up to date |
| `python -m cm.pi <host>` | any machine that calls it | Points the pi agent at the router and rewrites its model list from what the router actually serves |
| `python -m cm.install slave` | a machine lending its card | Registers the scheduled task that starts its RPC worker at boot, from the release `install llamacpp` put there, and admits the local subnet to the worker's port, then prints what to type on the machine with the router. Needs an elevated session; stores no password |
| `python -m cm.slave start` | a machine lending its card | Starts the worker by hand, lending the card with the most memory. `stop` ends the worker holding its port |
| `python -m cm.install master <host> <memory>` | the one with the router | Names the slave in the settings file — where its worker listens and how much memory its card has — and says whether it answers. `calibrate` then places models across that card too |

They share no state of their own. `scan` writes the entries `calibrate` places,
`calibrate` writes a file `router` hands to the
server, `install llamacpp` unpacks the releases both of them run and records which build
that is, `autostart` and
`firewall` make the router something the network can reach without anybody logging in,
`install pi` puts the agent on the machine that calls, and `pi` asks the router over
HTTP and copies nothing from the machine with the card. `install master` writes the one
table `calibrate` reads about a slave, and the slave is told nothing about the router at
all: whatever connects to its worker is served.

## What runs where

`calibrate` and `vram` read this machine through Windows: the card through `nvidia-smi`,
the installed memory through `GlobalMemoryStatusEx`, the cores and their efficiency
classes through `GetLogicalProcessorInformationEx`, and, for `vram`, video memory per
process through the performance counters. They say so and stop on any other system rather
than guessing.

`router`, `autostart` and `firewall` read no card, but they belong to that machine all
the same: `router` starts and stops Windows processes, `autostart` hands a task to the
Windows scheduler, and `firewall` asks netsh what the machine admits. The last two need
an elevated session, and where they do not have one they ask Windows for it rather than
half-doing the work.

`install slave` and `slave` belong to a machine lending its card, and to Windows for the
same reasons: they read its card through `nvidia-smi`, hand its worker to the
scheduler, ask netsh to admit the worker's port, and start and stop Windows processes.
`install slave` needs an elevated session too, and asks for it the same way.
`install master` reads no card and needs no rights: it writes the settings file on the
router's machine and knocks once on the slave's port.

`install pi`, `models` and `pi` have none of that. One runs npm and pi, one looks in the
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

On a machine lending its card: an NVIDIA driver with `nvidia-smi` on the path, and a
copy of this repository to run `install llamacpp` and `install slave` from.

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
| `install llamacpp` | `--check` | say which build would be installed -- `Build: b11070` -- and download and record nothing |
| | `--build` | install this build rather than the newest: the one the other machine's `--check` printed, so that the router and a slave run the same. Pruning leaves it for you to remove |
| | `--force` | download even where the release is already here -- after an install that was interrupted or came out damaged |
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
| `install slave` | `--port` | the port the worker listens on. 50052 |
| | `--print-command` | print what would be registered, down to the task XML, and change nothing |
| | `--remove` | stop starting the worker at boot and close its port, leaving llama.cpp installed |
| `slave start` | `--port` | as above |
| | `--print-command` | print what would be run, and start nothing |
| | `--foreground` | run the worker in this console instead of detaching — which is how the scheduled task runs it |
| `slave stop` | `--port` | as above |
| `install master` | `--port` | the worker's port, unless the address already carries one. 50052 |
| | `--reserve` | MiB to leave on the slave's card for its own machine. 2048 |
| | `--remove` | take the slave out of the settings file |

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

The whole path is nine commands, and three more where another machine lends its card; the
rest of this section is what each one does. `python -m cm` prints this list in the
console, so it is not something to come back here for:

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
python -m cm.install llamacpp         # 6. on a machine lending its card, --build N
python -m cm.install slave            #    there too                  (elevated)
python -m cm.install master <it> 12G  #    here, naming it; then calibrate again
```

**1. The settings file, and llama.cpp.**

```bash
python -m cm.install llamacpp
```

On a machine that has no `settings.toml` it copies one from `settings.template.toml`
beside it, and says so. Nothing in that copy has to be edited before the download, which
is what lets one command do both: the releases go into `.llamacpp` in this directory
whatever the file says, and every other key has a default or is asked for by the command
that needs it. So the same command puts llama.cpp on a machine lending its card, which
never needs to know where any model is.

Then it asks the NVIDIA driver which CUDA version it runs and `nvidia-smi` which cards
there are, reads which CUDA versions the project builds for Windows, and takes the newest
of those that runs here. A driver runs a build for its own CUDA version or an older one;
one of the same major version runs a newer build as well, on the cards the build carries
finished code for -- RTX 3000, 4000 and 5000 -- but not on the ones it carries only PTX
for, such as an RTX 2070, which need a driver at least as new as the build. Which cards
those are is read from the project's own source at that build's tag,
`ggml/src/ggml-cuda/CMakeLists.txt`, so a card the project adds is known without an
update here; where that cannot be read it says so and takes only a build no newer than
the driver, which runs on any card. Where the newest published does not run here, the
newest that does is taken, and it says which driver the newest needs. It downloads the
newest release built for that version and the CUDA runtime beside it -- some 550 MB,
700 MB unpacked -- unpacks it into `.llamacpp\b11070-cuda13.4`, and moves it into place
only once the server in it reports the build number it was downloaded as. Then it
records that build in the settings file, as `llamacpp_build = 11070`, and that is the
build the router, `calibrate` and a slave's worker run from then on -- not whichever is
newest in `.llamacpp`. `--check` says which build it would install and downloads and
records nothing.

`--build 11065` installs that build rather than the newest, and where it is already
unpacked downloads nothing and only records it: that is how a machine is moved back
onto a build it has. A build installed this way is never removed by pruning; it goes
when you delete it.

Two keys in that copy are worth a look before the download rather than after.

`cuda_version` is left out unless a machine has to stay on one CUDA version -- a slave
and the router kept alike, say. Left out, the version follows the project: it moves its
builds from one CUDA version to the next and stops publishing the old one, and `install`
moves with it, never to one that does not run here. Named, that version is taken while
the project publishes it and it runs here; where either stops being true, `install`
takes the newest that runs instead and says what it stood in for, rather than
installing nothing. Write it the way the archives do, in quotes: `'13.4'`.

`model_root` is the directory the weights sit under. Left commented out it falls back to
LM Studio's library, since that is where these files land anyway and LM Studio records
the folder it downloads into. Where LM Studio is not installed there is nothing to fall
back on and nothing on the machine to read it off, so `scan`, the first command that
needs the models, asks:

```
There is no LM Studio library at C:\Users\you\.lmstudio\models, so nothing on this
machine says where the models are.
Directory the models are under: D:\models
```

and writes that answer into the settings file as `model_root`. Where nobody is at the
keyboard -- a pipe, a script -- it refuses rather than guessing, and says to set
`model_root` by hand. Either way what the file says wins over the library.

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

A name already in the file stays, however it was arrived at, which is how a key written
when a model was the only one of its kind outlives the day a second quantisation arrives
beside it. `--force` is where that is given up on purpose:

```bash
python -m cm.scan --force
```

Every entry is then keyed the way the library would name its file today, so that a name
says which file it is rather than what it happened to be called first. Only the entry's
two headers change; the file it names, the values under it and anything written around
them stay. An entry marked `manual = true` is not renamed either, and neither is one
whose file sits outside the library. Every profile of a renamed model is served under a
new name afterwards, so the preset has to be written again:

```bash
python -m cm.calibrate
```

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
| there is one keyed some other way, and `--force` | it is keyed the way the library names its file |
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
the header, where it survives. Nothing else in the file is touched: `scan` never
reorders an entry, and never edits a key other than its settings.

One entry it does take out: the one naming a file that is not there. Such an entry
names nothing, and `hidden` or `manual` makes no difference to that -- both its tables
go and the run says which entry and where the file was, along with the note above the
header: it was about the model named under it. An entry that cannot be cut whole -- one
written twice, or carrying a table besides its settings -- is left alone and named
instead.

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

It runs the server of the build `install llamacpp` recorded, from `.llamacpp`, detached,
with the preset from step 2 and nothing else -- no model is loaded until a request names
one.
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

**6. Another machine's card**, where there is one to lend. The router and the worker on
that machine only talk when they are the same build, so first ask both which build they
would install, on each of them:

```bash
python -m cm.install llamacpp --check
```

and take the lower of the two numbers it prints. Then the first two commands on that
machine, from a copy of this repository, and the rest on this one:

```bash
python -m cm.install llamacpp --build 11065
python -m cm.install slave
python -m cm.install llamacpp --build 11065
python -m cm.install master <that machine> 12G
python -m cm.calibrate
python -m cm.router start
```

The first puts that build there; the second asks for the rights it needs, starts its
worker at every boot and says what to type here. Here the same build is installed, the
slave written into the settings file, the models placed again, and the router started
on what that wrote. What each of them does, and what `calibrate` does with the card, is
[below](#several-cards-and-a-card-lent-over-the-network).

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
over from a release of the same CUDA version already there rather than downloading four
hundred megabytes of it again, records it as the build to run, and removes the releases
past `keep_releases` -- except the one recorded, one a server is running from, and every
one installed with `--build`. A start is what puts the router on it.

`calibrate` again because the placements were measured by asking that build's
`llama-fit-params` what fits. A build that packs its buffers differently answers
differently, and the preset would still be holding the old answer. It costs seconds and
loads nothing.

Where another machine lends its card, the two move together: the router and the worker
talk to each other, and nothing here checks that two builds still agree on how. Run
`python -m cm.install llamacpp --check` on both, install the lower build on both with
`--build`, and start the worker there again with `python -m cm.slave start`.

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
On a machine lending its card, `install slave --remove`, elevated, and `slave stop`; the
rest there is files in that copy of this directory. On the router's machine,
`install master --remove` takes the slave out of the settings file.

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

> qwen3.8-45k-nomtp             13890 MiB   short by 1960 MiB
  qwen3.8-88k-q4-nomtp          12640 MiB   short by 710 MiB
  gemma4-12b                     9310 MiB   loads now, 2620 MiB to spare

  [x] chrome                          9184    3480 MiB
  [ ] Code                           21556    1960 MiB
  [ ] Discord                         4472     620 MiB

  the other 1420 MiB is Windows, the compositor and this window -- not on offer
  310 MiB of that is this window, and comes back when it is closed
  marked 1 program, 3480 MiB -- enough by these figures
  [tab] next model   [shift-tab] back   [space] mark   [enter] close marked   [r] refresh   [q] quit
```

The card is the one a desktop is drawn on, since what holds any other is nothing a person
closes a window to give back: a machine's only card, or of several the one the desktop
draws on. Each profile's figure is what it holds on that card -- nothing, for a profile
that does not use it -- and each program's is what it holds there. Where several cards
draw a desktop, or none of several does, `vram` says so and stops.

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
| `Which cards b... carries finished code for could not be read` | the project's source at that tag is not where it was, or reads differently. Nothing breaks: only a build no newer than the driver is taken until this is revisited |
| `Nothing published lately runs here` | the NVIDIA driver is too old for anything the project builds for Windows now, on one of the cards it names. Update it |
| `No release carries a Windows CUDA archive` | the project has renamed its archives, and this needs revisiting. It prints the newest tags it saw |
| `No llama.cpp release under ... carries llama-server.exe` | step 1 has not happened: nothing is unpacked in `.llamacpp` yet |
| `No llama.cpp release under ... carries ggml-rpc-server.exe` | on a machine lending its card: `install llamacpp` has not run there yet |
| `The settings file records build N as the one to run, and no release of it ... carries ...` | that build was deleted by hand, or the settings file came from another machine. It prints the `--build` line that puts it back |
| `The preset the router reads was not found: ...` | `calibrate` has not run since the settings file changed |
| `qwen3.8: ctx-size is derived; remove it` | a placement was written into the settings file by hand. Those are measured, and one written down would be obeyed silently and wrongly |
| `Port N is still held by process M` | either a router this session may not signal — one started by the scheduled task, under another logon session — or something that is not `llama-server.exe` and was left alone deliberately. It prints the elevated `Stop-Process` line either way |
| `The prompt was declined, so nothing was changed` | `autostart` or `firewall` asked Windows for the rights it needs and the prompt was answered no. Run it again and accept, or run it from a session that is already elevated |
| `pi was not found, and neither was npm` | `install pi` needs Node.js. It prints the winget line |

---

## The settings file

`settings.toml` is yours. `install llamacpp` copies it from the template where there is
none and records in it the build it installs; `scan` writes the model entries and
`install master` the slave, and nothing else writes it. It holds three kinds of thing:

**Where things are.** `model_root`, and `preset_path` — the file `calibrate` writes.
Where llama.cpp is is not among them: the releases are in `.llamacpp` beside this file.

**Which llama.cpp.** `llamacpp_build` is the build everything here runs, and
`install llamacpp` writes it. `cuda_version` pins the CUDA version of the archives it
takes, where it would otherwise take the newest that runs here, and `keep_releases` says
how many unpacked releases to keep when it installs a newer one. Neither of the two has
to be named: without them the version follows the project, the driver and the cards,
and two releases are kept.

**How the router runs.** `listen_host` and `port` are what it binds, `models_max` how
many models may be resident at once, `sleep_idle_seconds` how long an idle one is kept
before the card is given back, and `log_dir` where it writes. All five have defaults —
every address, 18081, one, a quarter of an hour, and a `logs` directory beside this
file — and a machine serving one card has no reason to name any of them.

**Judgements about the work**, which no machine can make for you:

| key | what it decides |
|---|---|
| `reserve_mib` | video memory to leave for everything that is not a model, on a machine with one card |
| `reserve_multi_gpu_mib` | the same where the machine has several cards, on each card a desktop draws on. 2048 |
| `reserve_no_desktop_mib` | the same on each card none draws on, where the machine has several. 768 |
| `min_ctx_tokens` | below which a window stops being worth serving |
| `ample_ctx_tokens` | past which a coarser attention cache buys nothing worth having on a card alone |
| `cache_ram` | how much system memory the prompt cache may hold |

`cache_ram` is written the way anyone would write it — `32`, `32G`, `32Gb`, `32GiB` for a
size in gibibytes, `50%` for a share of the memory installed. Leave it out and it is
worked out for each model on its own: what the machine has, less the weights that model
keeps off the card, less a share for the system. Written down by name, it is room the
weights may not take either: a model is placed within what is left after it, and every
profile is given the size as written. Half the memory, which is what an older version of
this used, is not the answer on a machine whose job is serving models with nobody logged
in.

**The slave**, where another machine lends its card: a `[slave]` table saying where its
worker listens, how much memory its card has and how much of it to leave.
`install master` writes it and takes it out again — see
[below](#several-cards-and-a-card-lent-over-the-network).

**The models**: one entry per GGUF, written and kept up to date by
[`scan`](#running-the-programs) and yours to change afterwards. Everything beyond `file`
is optional — the sampler settings, which come from `recommended.toml` in this
repository, `cache` and `mtp` to narrow which profiles `calibrate` may offer,
`hidden = true` to set a model aside without deleting the entry that names its file, and
`manual = true` to keep `scan` out of that entry's settings for good.

What may **not** be in it: `ctx-size`, `cache-type-k`, `cache-type-v`, `gpu-layers`,
`n-cpu-moe`, the `spec-*` keys, `threads`, `cache-ram`, and what says where the layers go
and how they run: `device`, `split-mode`, `tensor-split`, `ubatch-size`, `rpc` and
`override-tensor`. Those are worked out against this machine, and one written by hand is
refused rather than obeyed — a placement quietly overridden is the failure this program
exists to prevent, and it would not even look like one: the router would start, serve the
model, and hold a different window than the card was measured for.

The settings file is yours and is not in the repository. The template is.

## What the machine decides

| what | from |
|---|---|
| the window each profile holds | asked of `llama-fit-params` against each card's total memory, with what it is never told about a prediction head added, searched until the placement lands near each card's reserve |
| which cards a profile runs on | the fastest card alone first, then each card added in front of it, then a slave's card, each only where it holds a longer window than every one before |
| the attention cache | `q8_0`, with `q4_0` beside it only on a card alone while `q8_0` holds less than `ample_ctx_tokens` there -- or instead of it, where `q8_0` does not fit a machine's only card |
| whether a prediction head runs | both ways on a machine with one card; with a second card, always where the file carries one |
| how many expert layers of a mixture stay in system memory | the same search, along the other lever -- and with several cards, none until every card holds layers of it |
| how many layers each card holds, where there are several | laid out from the fastest card back, each taking layers while that brings it nearer its reserve |
| the micro-batch a profile runs at | `ubatch-size` from `[shared]`, halved down to 128 only where the window it buys is worth the prefill it costs |
| whether several cards run pieces of a prompt at once | yes, unless the window without it is at least 30% longer |
| `threads` and `threads-batch` | the logical processors of the fastest cores. Not all of them: the thread pool is synchronised by a barrier, so work spread onto slow cores is work the rest wait for |
| `cache-ram` | per profile: the memory installed, less what that model keeps outside the card, less a share for the system. `[*]` carries the same worked out for the heaviest model, which is what one with no profile of its own is left |

A placement is never narrowed to fit around a browser: what fits is worked out against
each card's total, so it does not matter what is open while `calibrate` runs. Whether a model
fits *right now* is a different question, and that is what
[`vram`](#watching-the-card) is for.

## Several cards, and a card lent over the network

**Several cards in the machine** need nothing set. `calibrate` reads every card
`nvidia-smi` lists and adds them one at a time, the fastest first. Nothing a card reports
says how fast it is, so the latest generation stands for the fastest, and of one
generation the card with more memory goes first; whether a desktop draws on it
decides nothing. The fastest card alone is placed first: those are the quickest
profiles, with the compromises one card makes. Then the next card is added in front of
it, and a profile across both is written only where it holds a longer window than the
fastest card alone did for the same cache and head. Every card added costs speed, so it
has to buy window; the profiles add up, and which to load is chosen by name. A slave's
card is the last one added.

A profile's name is the model's key, the window in thousands of tokens where the model
has more than one profile, and then what that profile gives up against the best way the
model runs: `-q4` for a coarser attention cache, `-nomtp` where the file carries a
prediction head this profile does not run, `-2gpu` for two of the machine's cards and
`-3gpu` for three, and `-rpc` where a slave's card is among them. A `q8_0` cache, a head
that runs and one card are not written.

With the desktop on an 8 GiB card and a 16 GiB one beside it, `qwen3.8` gets
`qwen3.8-25k` and `qwen3.8-57k-q4` on the 16 GiB card alone and `qwen3.8-120k-2gpu`
across both, while `gemma4-12b`, which holds its whole trained window on the 16 GiB
card, gets that one profile and nothing across two. `ornith-1.5-35b`, a mixture the
16 GiB card alone holds only with experts in system memory, gets one profile across both
cards, `ornith-1.5-35b-2gpu`, with the experts of 2 layers there.

Three rules follow the cards. Where the machine has a second card, every profile runs
the prediction head a file carries: dropping it would buy window the second card buys
better. A coarser `q4_0` cache is offered only on a card alone, beside `q8_0` while
`q8_0` holds less than `ample_ctx_tokens` there; a profile across several devices is
always `q8_0`. A model whose `q8_0` does not fit on the fastest card at all gets a
`q4_0` profile of that card only on a machine with one card -- with several, it is
placed across the cards or not at all. And experts read from system memory are slower
than on any card, so a mixture the fastest card cannot hold whole is not offloaded
there: it is placed across the cards after it, and only once every card of the machine
holds layers of it do experts stay in system memory.

System memory is the other bound, the cards being one of them. What a placement leaves
there may not be more than this machine has for weights -- what it has beyond the
system's own share of 8 GiB, less a `cache_ram` written down by name -- because weights
counted on but not held are read off the disk for every token, or fail to be locked
where `load-mode` asks for them to be. A model whose heaviest profile wants more than
that is left out of the preset, and the run says so under its name.

Not everything the estimator counts there is held there. An architecture may mark a
tensor as read row by row -- the table of per-layer embeddings is one, and on a 87 GiB
file it is 27 of them -- and llama.cpp maps it and fetches the rows a token asks for,
whatever `load-mode` says. The estimator counts it as memory all the same, its
memory-fit pass allocating nothing and mapping nothing, so the model's own file is read
here and those tensors come off what a placement is said to need. `lazy-mode = off` in
the shared block asks llama.cpp to hold them after all, and then nothing is taken off.

The cards of a profile run slowest first, the order they were added in reversed, and a
model's layers pass through them in that order. The last card is the one that works
hardest -- it takes the most layers, the output and the prediction head -- so it is the
fastest. The layers are
laid out from that end: the last card takes blocks for as long as each one brings what it
leaves free nearer its reserve, the card before it does the same with what remains, and
the first card takes the rest. Every card of a profile holds at least one block.

The numbers are `nvidia-smi`'s. CUDA on its own counts the fastest card first; every
process these programs start is told to count in bus order, as `nvidia-smi` does, so
`CUDA1` in a profile is the card `nvidia-smi` calls 1.

A profile on two cards, as `calibrate` writes it, with the sampler values left out:

```ini
[qwen3.8-120k-2gpu]
; VRAM REQUIRED: 6185 MiB on CUDA1, 15446 MiB on CUDA0, held from the moment this profile loads
model = D:\models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ4_XS.gguf
cache-type-k = q8_0
cache-type-v = q8_0
ctx-size = 120000
device = CUDA1,CUDA0
fit = off
gpu-layers = 99
spec-draft-n-max = 3
spec-draft-type-k = q8_0
spec-draft-type-v = q8_0
spec-type = draft-mtp
split-mode = layer
tensor-split = 21,45
```

`tensor-split` counts layers per card, in the order `device` names the cards: 21 on the
older one and 45 on the newer, the output and the head's own layer among the 45. Loaded,
this profile took 6067 MiB of the 8 GiB card and 15430 of the 16 GiB one -- a little
under what its requirement says, which is the side `calibrate` errs on. A profile on one card of
several names that card too, `device = CUDA0` with `split-mode = none`: left unnamed,
llama.cpp would take the first card it counts.

What each card is left:

| the card | keeps about |
|---|---|
| a machine's only card | `reserve_mib` |
| of several cards, one a desktop draws on | `reserve_multi_gpu_mib`: 2048, enough to work at the desktop while the rest serve |
| of several cards, one no desktop draws on | `reserve_no_desktop_mib`: 768, what the card keeps from everybody, so a card the monitors are moved off serves with all the rest of its memory |
| a card lent over the network | `reserve_mib` under `[slave]`: 2048, everything its own machine keeps |

A reserve is the whole of what a card is left: what the driver holds for itself is inside
that figure rather than taken off beside it, which is why one number is enough to write
down. Every one of these is a target to land near rather than a line to clear: a layer
that fits by landing a few megabytes under goes on.

Where nobody is logged in, no desktop draws on any card and every card of several is left
what a card without one is. That is a machine given over to serving, and logging out of it
is how to ask for that.

Which card a desktop draws on is read from Windows' own counters, as the card the
compositor holds video memory on. Not from the monitors: over a remote session Windows
detaches the machine's own, and `nvidia-smi` then reports none on any card while the
desktop goes on holding what it holds. So `calibrate` writes the same profiles run from
another machine as it does at this one.

The micro-batch is searched as well, on one card as on several. `ubatch-size` in
`[shared]` is where the search starts, and each halving of it down to 128 is placed too.
A halving costs roughly a tenth of prefill speed, so it is taken only where it brings the
window more than a tenth of `ample_ctx_tokens` nearer that length -- past it a window buys
nothing -- and a profile that runs at a smaller one says `ubatch-size` in its section.

On two cards or more, llama.cpp runs pieces of a prompt on the cards at once, holding
extra working buffers to do it, and generates several per cent faster for it. `calibrate`
gives that up only where the window without it is at least 30% longer, and a section that
gives it up carries an `override-tensor` naming a tensor no model has: any override at
all is what turns it off.

A prediction head costs more than the estimator can be told about: `llama-fit-params`
takes no `--spec-type`. So on top of its estimate `calibrate` adds what the server holds
for a head -- its weights and its cache, the working buffers of the draft context on the
last card, and on every card a snapshot of each recurrent layer's state for every token
the head drafts, which is what a rejected draft is rolled back to.

**A card lent over the network.** Another machine with an NVIDIA card -- a laptop, say --
can lend it to this one. It runs llama.cpp's RPC worker, which offers that card to
whatever connects, and a profile that uses it keeps some of its layers there and reaches
them across the network.

That machine needs an NVIDIA driver and a copy of this repository, and two commands, run
from that copy in an ordinary console:

```bash
python -m cm.install llamacpp --build 11065
python -m cm.install slave
```

The first installs llama.cpp there as it does on any machine -- the build the router
runs, which `--check` on both machines settles -- and records it in a settings file of
that machine's own. The second installs nothing, and refuses where there is nothing
installed to run; it registers the scheduled task `llama.cpp RPC worker`, which starts
the worker at every boot the way `autostart` starts the router, as that account and with
no password stored, and admits the local subnet to the worker's port, 50052, the way
`firewall` does. Like those two it asks Windows for the rights before doing any of it.
Of the settings file there, the worker reads `llamacpp_build` and `log_dir` and nothing
else. Then it says what to type on the machine with the router:

```
Lending NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227 MiB, on port 50052.
Start it now rather than at the next boot: python -m cm.slave start

On the machine with the router:
  python -m cm.install master <this machine's name or address>:50052 12G
```

The worker lends the card with the most memory and nothing else. Offered the processor as
well, it would let the router place layers in that machine's system memory, at the far end
of a network cable.

`python -m cm.slave start` starts the worker by hand and `python -m cm.slave stop` ends
it, the way `router start` and `router stop` treat the router: a start stops a worker
already on the port first, the scheduled task included, and what the worker writes goes
to `slave.stdout.log` and `slave.stderr.log` in `log_dir`.

**Nothing on that port is authenticated or encrypted.** Whoever reaches it can have the
card run whatever they send it, which is why the rule admits the local subnet and nothing
wider.

On the machine with the router, name it and place the models again:

```bash
python -m cm.install master <that machine> 12G
python -m cm.calibrate
python -m cm.router start
```

`install master` writes one table at the end of the settings file:

```toml
[slave]
address     = '<that machine>:50052'
memory      = '12G'
reserve_mib = 2048
```

`memory` is the round figure `install slave` printed. Nothing on the router's machine can
read a card on another one, so it is taken as written. `reserve_mib` there is everything
the other machine keeps on its card, the driver included, in one figure -- nothing else is
taken off it -- and `--reserve` writes another. `--remove` takes the table out, and a
second `install master` replaces the first, so there is one slave at most. It says whether
the worker answers, and writes the table either way.

`calibrate` then adds the slave's card last, in front of all of the machine's cards. In
front, because a card reached over the network is slower than any card in the machine,
so it is the one that takes what the others leave. A profile across the slave is written
only where its window is longer than every profile before it gives the same model with
the same cache and head, beside those rather than instead of them. Its name ends in
`-rpc` and says nothing of the cards beside it -- `qwen3.8-259k-rpc`, or
`ornith-1.5-35b-rpc` for a model with that one profile -- so what a client asks for says
it needs the other machine. Its section names the worker it reaches:

```ini
device = RPC0,CUDA1,CUDA0
rpc = <that machine>:50052
split-mode = layer
```

With a slave in the chain llama.cpp does not run pieces of a prompt on the cards at once,
so there is no second way to place those profiles.

A slave that does not answer when `calibrate` runs is said first, and that run writes no
profile using its card; the machine's own profiles are written as always. Run `calibrate`
again once it answers. A profile across the slave needs its worker running whenever it
loads.

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
