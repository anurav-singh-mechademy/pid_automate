# P&ID tag extraction and description generation

Two pipelines that run on P&ID drawings (PDF, PNG, JPG):

1. **Tag extraction** finds every instrument tag on each sheet (`PIT-9010A`, `034-TICA-0813`) and writes a mastersheet.
2. **Description generation (descgen)** writes a description for each extracted tag in the client's style
   (`Fuel Gas Compressor A Stg 2 Suction Pressure`), with the drawing evidence behind every word.

Both are cold start: no equipment, line or tag lists are inputs. Everything runs locally with open-weight models
(RapidOCR / PaddleOCR for text, RF-DETR for instrument bubbles); nothing is sent to a hosted service.

## What is in this repository

| Path | Contents |
|---|---|
| `batch/tag_pipeline.py` | per-page tag extraction: render, orientation, tiled OCR, bubble sizing, RF-DETR detection, text-to-bubble, tag building |
| `batch/batch_tags.py` | runs tag extraction over a folder (resumable, de-duplicates identical pages), writes the mastersheet |
| `batch/tag_filter.py` | precision filter: learns the batch's own tag formats, repairs / splits reads, marks drop reasons (never deletes) |
| `batch/tag_viewer.py` | one-command extraction + interactive HTML overlay with a clickable tag list |
| `batch/pdftext.py` | exact text from the PDF itself (text layer, AutoCAD SHX comments) where a page has it |
| `batch/pid_registry.py`, `batch/eval_*.py`, `batch/bench_tags.py`, `batch/survey_pdfs.py` | data registry and evaluation scripts (need the client data, see below) |
| `descgen/` | description generation: `run.py` (entry point), `sheet.py` (reading a sheet), `plant.py` (equipment register), `describe.py` (finders), `onpipe.py` / `auxsys.py` / `lines.py` / `trace.py` (lines and pipes), `rules.py` (domain tables), `style.py` (drafting-style profiles), `bench.py` (scoring), `desc_split_v2.json` (frozen dev / test split) |
| `descgraph/`, `pidgraph/` | sheet graphs (symbols, pipes, signal lines) and the local LLM judge (`descgraph/judge.py`) |
| `descgen_lab/judge_current.py` | LLM scoring of descgen outputs |
| `models/rfdetr_pid.pth` | RF-DETR P&ID symbol detector (third-party fine-tune, used as is; class ids are 1-based). Stored with Git LFS |

**Not included:** client drawings, client tag lists and reference names (the ground truth), run outputs, caches, notebooks
and earlier experiments. The pipelines run without them; the evaluation scripts need them (see *Evaluation*).

## Setup

```bash
git lfs install && git lfs pull                      # model weights (brew install git-lfs if missing)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt            # RapidOCR downloads its OCR models on first use
```

## 1. Tag extraction

```bash
# a folder of PDFs (loose, or inside .zip archives) -> <out>/
.venv/bin/python batch/batch_tags.py <pdf_folder> <out>

# precision filter over the whole batch (adds keep / drop_reason; writes mastersheet_filtered.csv)
.venv/bin/python -c "import sys; sys.path.insert(0, 'batch'); import tag_filter; tag_filter.filter_mastersheet('<out>')"
```

Or both steps in one command, from any mix of PDF files and folders, plus the interactive viewer:

```bash
.venv/bin/python batch/tag_viewer.py drawing1.pdf drawing2.pdf <pdf_folder> --out <out>
.venv/bin/python batch/tag_viewer.py --from <out>     # viewer only, for an existing output folder
```

Outputs in `<out>/`:

| File | Contents |
|---|---|
| `mastersheet.csv` / `.xlsx` | every tag read on every sheet: tag, source (bubble / stack / text), confidence, bbox, `keep`, `drop_reason` |
| `mastersheet_filtered.csv` | kept tags only |
| `sheets/tags_<sheet>.csv`, `sheets/tags_<sheet>_overlay.png` | per-sheet tags and the overlay image |
| `tag_viewer.html` | overlay on the left, tag list on the right; click a tag to zoom to it |
| `ledger.csv`, `inventory.csv` | per-page status (re-running skips pages already done) and duplicate pages |

## 2. Description generation (descgen)

Runs on a tag-extraction output folder and describes every kept bubble tag:

```bash
.venv/bin/python -m descgen.run <out>                    # -> <out>/descriptions.csv
.venv/bin/python -m descgen.run <out> --loop-register    # also fill bare displays from the same loop on other sheets
.venv/bin/python -m descgen.run <out> --no-register      # skip the cross-sheet equipment-name register
```

`descriptions.csv`: sheet, tag, description, confidence, review (why it may need a look), and `pieces`, the JSON of
every word group with its value, source on the drawing and confidence. Full-page OCR is cached in `<out>/ocr/`.

New clients or drafting houses are added as a profile in `descgen/style.py` (layout, tag formats, wording) and rows in
`descgen/rules.py` (domain knowledge); the finders are general.

## Evaluation (needs the client data)

The data is not in the repository. With it in place (paths are in each script's docstring):

```bash
.venv/bin/python batch/pid_registry.py                    # pid_registry.csv: every sheet, its ground truth, its split
.venv/bin/python batch/bench_tags.py [--ablate]           # tag precision / recall on the judged sets
.venv/bin/python batch/eval_bog_pi.py lngc_bog_a_output lngc/PI_tags_BOG_Anurav.csv    # tags vs a PI tag list
.venv/bin/python -m descgen.bench --set dev               # descriptions vs client names: dev | test | dev2 | test2
.venv/bin/python -m descgen_lab.judge_current dev=<results folder> test=<results folder>   # local LLM judge
```

### `pid_registry.csv`

One row per unique drawing page (320: 189 DECP, 78 probably DECP vendor drawings from the same delivery, 41 LNG Canada,
12 PID2Graph OPEN100), with its source folder, client, file type, and which ground truth exists for it. Regenerate with
`batch/pid_registry.py`; read it before any experiment.

The `tag_split` and `desc_split` columns give each sheet's dev / test assignment.
