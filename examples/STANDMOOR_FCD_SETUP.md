# Re-running the archived Standardized Moorings and Fishing Co-Design analyses

This guide recreates the environment used for the Standardized Moorings (StandMoor) and Fishing
Co-Design (FCD) WAVES runs, which were archived in March 2026.

## Pinned versions

| Package | Source | Version / ref | Install |
|---|---|---|---|
| WAVES | `https://github.com/shousner/WAVES` | tag `archive/iea-49-standmoor-fcd` (commit `8259343`) | editable |
| WOMBAT | `https://github.com/NLRWindSystems/WOMBAT` | tag `v0.11.3` (commit `253f941`) | editable |
| ORBIT (`orbit-nrel`) | `https://github.com/NLRWindSystems/ORBIT` | `Standard_Mooring` branch, commit `5102892df18e047eec0d88a16dba60e06105dc2c` (Mar 5, 2026) | editable |

### Notes

**WAVES**

- The code is Stein's fork of NREL's `analysis/iea-49` branch. That branch was frozen in May 2026.
- The fork changed WAVES source code in only one place: a 9-line patch in `waves/project.py`. It
  registers ORBIT's `CustomMooringSystemDesign` and `MooringSystemSupplyChain` phases before
  ORBIT's `ProjectManager` is created. Later ORBIT Standard_Mooring branch commits (Nov 2026) handle this registration in ORBIT itself.
- Everything else the fork adds is analysis inputs and scripts: the `library/Standardized_Moorings*`
  and `library/FCD_*` folders, `examples/runWAVES.py`, and
  `examples/standmoor-example-floating-deep.ipynb`. They were kept in the WAVES repo for
  convenience; they don't depend on being there.
- The version may show as `0.6.1` (the code's `__version__`) or `0.5.3` (old install metadata).
  Both refer to the same code.

**ORBIT**

- The `Standard_Mooring` branch holds in-progress ORBIT work. It adds the
  `CustomMooringSystemDesign` and `MooringSystemSupplyChain` phases and new installation logic
  for shared mooring components.
- The version may show as `1.2.4` or `1.2.1`. The branch never had a PyPI release, so identify it
  by commit rather than version number.

**Other packages**

- FLORIS, LandBOSSE, pandas and so on were not pinned when the runs were made. A new install gets
  whatever pip resolves (WAVES requires `floris>=4.2.2`), so results may differ slightly from the
  archived ones (Stein's waves-env listed `FLORIS 4.2`)
- The versions installed in the original `waves-env` (Python 3.11.13) were recorded in [standmoor_fcd_requirements.txt](standmoor_fcd_requirements.txt). Key versions:
  numpy 1.26.4, pandas 2.3.2, polars 1.32.3, pyarrow 21.0.0, scipy 1.16.1, simpy 4.1.1,
  nrel-landbosse 2.6.1, FLORIS 4.2.x. To match them, run
  `pip install -r standmoor_fcd_requirements.txt` after the editable installs.
- In `waves-env`, the editable installs report their versions from old install metadata
  (`waves 0.5.3`, `orbit-nrel 1.2.1`). The code itself reports 0.6.1 and 1.2.4.

### Other ways to run the analyses

The table above is the setup the archived runs used, and the steps below follow it. These
alternatives should also work:

| Component | Primary | Alternatives |
|---|---|---|
| ORBIT | `Standard_Mooring` @ `5102892` (Mar 5, 2026) | `Standard_Mooring` @ `3be5f015d1b3a7da83b594e636ad2d3e1a9dc672` (Nov 21, 2025): an earlier stage of the same work, which could be checked out if the Mar 5 commit shows different results. |
| WAVES code | this fork @ `archive/iea-49-standmoor-fcd` | NREL's `analysis/iea-49` branch, or WAVES `v0.6.1`. Without the fork's patch, these work only if the installed ORBIT registers the custom mooring phases itself (Nov 2026 or later), or if the run script registers them before creating `Project`. |
| Project libraries and scripts | inside this fork | Copy the `library/Standardized_Moorings*` and `library/FCD_*` folders and `examples/runWAVES.py` to any location and use them with any of the WAVES options above. The library paths in `runWAVES.py` are relative, so run it from the folder that contains `library/` or update the paths. |
| WOMBAT | `v0.11.3` (editable) | `pip install wombat==0.11.3` from PyPI gives the same code. |

Mixing in newer WAVES or WOMBAT releases (WAVES 0.7+, WOMBAT 0.13+) changes input formats and
model behavior. Treat that as a new analysis rather than a re-run.

## 1. Clone the repositories

The commands below assume this layout. Adjust the paths if yours differ.

```text
C:\Code\WAVES    <- shousner/WAVES (this repo)
C:\Code\WOMBAT
C:\Code\ORBIT
```

```powershell
cd C:\Code
# Skip any clone that already exists
git clone https://github.com/shousner/WAVES.git
git clone https://github.com/NLRWindSystems/WOMBAT.git
git clone https://github.com/NLRWindSystems/ORBIT.git
```

## 2. Check out the pinned versions

Use a throwaway local branch in each repo so new commits never land on a detached HEAD.

```powershell
# WAVES
cd C:\Code\WAVES
git fetch origin --tags
git switch -c rerun/standmoor-fcd archive/iea-49-standmoor-fcd

# WOMBAT
cd C:\Code\WOMBAT
git fetch origin --tags
git switch -c rerun/v0.11.3 v0.11.3

# ORBIT
cd C:\Code\ORBIT
git fetch origin Standard_Mooring
git switch -c rerun/standard-mooring 5102892df18e047eec0d88a16dba60e06105dc2c
```

Check that each repo is at the expected commit:

```powershell
git -C C:\Code\WAVES  --no-pager log -1 --oneline   # 8259343 Updating StandMoor libraries ...
git -C C:\Code\WOMBAT --no-pager log -1 --oneline   # 253f941 Merge pull request #207 ... v0.11.3 Release
git -C C:\Code\ORBIT  --no-pager log -1 --oneline   # 5102892 Bug fix to turbine index value ...
```

## 3. Create the environment and install

Install ORBIT and WOMBAT **before** WAVES. WAVES requires `wombat>=0.11` and
`orbit-nrel>=1.2.1`; the editable installs already satisfy both, so pip won't replace them
with PyPI releases.

```powershell
conda create -n standmoor-env python=3.11 -y
conda activate standmoor-env

pip install -e C:\Code\ORBIT
pip install -e C:\Code\WOMBAT
pip install -e C:\Code\WAVES
```

## 4. Verify the environment

```powershell
python -c "import waves, wombat, ORBIT; print('WAVES ', waves.__version__, waves.__file__); print('WOMBAT', wombat.__version__, wombat.__file__); print('ORBIT ', ORBIT.__version__, ORBIT.__file__)"
python -c "from ORBIT.phases.design import CustomMooringSystemDesign; from ORBIT.phases.install import MooringSystemSupplyChain; print('custom mooring phases OK')"
pip list --editable
```

Expected:

- `WAVES 0.6.1 C:\Code\WAVES\waves\__init__.py`
- `WOMBAT 0.11.3 C:\Code\WOMBAT\wombat\__init__.py`
- `ORBIT 1.2.4 C:\Code\ORBIT\ORBIT\__init__.py`
- `custom mooring phases OK`

Save a record of the environment alongside any new results:

```powershell
pip freeze > C:\Code\WAVES\library\<case_folder>\results\env.txt
```

## 5. Run the analyses

All runs use `examples/runWAVES.py`. Its library paths are relative (`library/...`), so **run
it from the WAVES repo root**:

```powershell
cd C:\Code\WAVES
python examples\runWAVES.py
```

Settings near the bottom of `runWAVES.py`:

| Variable | Meaning |
|---|---|
| `run` | `1` = run WAVES simulations; `0` = only plot existing results |
| `NUM_RUNS` | number of WOMBAT simulations per case (archived runs used `10`) |
| `case` | which comparison to run (see below) |

| `case` | Project | Library folders (baseline vs. alternative) |
|---|---|---|
| 1 | StandMoor: Humboldt | `Standardized_Moorings` vs `Standardized_Moorings2` |
| 2 | StandMoor: Morro Bay | `Standardized_Moorings3` vs `Standardized_Moorings4` |
| 3 | FCD: Gulf of Maine | `FCD_Baseline` vs `FCD_Interrow` |
| 4 | FCD: Mid-Atlantic | `FCD_MidAtlantic_Baseline` vs `FCD_MidAtlantic_Interrow` |

Outputs go to `library/<case_folder>/results/` and `library/<case_folder>/results/figures/`.
**Running again overwrites the archived results**, so copy them aside first if you want to compare.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'CustomMooringSystemDesign'` | PyPI ORBIT is installed instead of the `Standard_Mooring` checkout | `pip uninstall orbit-nrel -y; pip install -e C:\Code\ORBIT`, then re-run step 4 |
| `__file__` points to `site-packages` instead of `C:\Code\...` | a package was reinstalled from PyPI (often by a later `pip install`) | reinstall that package with `pip install -e <path>`; add `--no-deps` to later installs |
| WAVES shows version 0.5.3 | out-of-date `WAVES.egg-info` metadata | `pip install -e C:\Code\WAVES` again; the imported `waves.__version__` is what counts |
| `FileNotFoundError` for `library/...` | script started from the wrong folder | `cd C:\Code\WAVES` before running |
| ORBIT/WOMBAT results differ from the archived results | wrong commit checked out, or package drift (FLORIS, pandas) | `git -C <repo> --no-pager log -1 --oneline`; compare `pip freeze` against a saved `env.txt` |
| `ValueError` while registering phases | phase already registered | handled by the `try/except` in `waves/project.py`; safe to ignore |

## Git commands

### Where am I?

```powershell
git status -sb                         # current branch and uncommitted changes
git --no-pager log -1 --oneline        # current commit
git branch -vv                         # local branches and what they track
git remote -v                          # origin = shousner fork, upstream = NREL
git tag -l "archive/*"                 # archive tags
```

### Go back to the archived state

```powershell
git stash                              # or commit, if you've made changes you want to keep
git switch --detach archive/iea-49-standmoor-fcd
```

### Save changes made while re-running (without touching the archive)

```powershell
git switch -c rerun/<description>      # if you're not already on a rerun/ branch
git add <files>
git commit -m "Describe the change"
git push -u origin rerun/<description>
```

The archive tag never moves, so the original state is always recoverable.

### Undo mistakes

```powershell
git reflog -n 20                       # every recent position of HEAD
git reset --hard <sha>                 # return to one of them (discards uncommitted changes)
git rebase --abort                     # stop a rebase that hit conflicts
git merge --abort                      # stop a merge that hit conflicts
git restore <file>                     # discard uncommitted edits to one file
```

### Rules of thumb

- Don't use `git pull --rebase` on branches whose commits are already on the fork. Use a plain
  `git pull` (merge).
- Only pull a branch into its own counterpart (e.g. `upstream analysis/iea-49` into
  `analysis/iea-49`).
- Run `git fetch` and look (`git --no-pager log --oneline HEAD..<remote>/<branch>`) before you
  merge anything.
- Tag any state that produced results: `git tag <name>; git push origin <name>`.
