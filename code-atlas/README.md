# vLLM-Spyre Code Atlas (Local)

Local-first code atlas for comparing integration seams across:

- upstream `vllm`
- `vllm-spyre` (today)
- `vllm-spyre-next` (inside `vllm-spyre` repo)
- `torch-spyre`

The web app renders:

- first-principles and architecture-layer walkthrough
- seam-by-seam narrative with explicit decision target and answer
- claim-driven story flow (3-5 story snippets: Contract, Call site, Data/Lifecycle, Divergence/Example)
- two-layer annotations per snippet (`What this proves` and `Implications and risks`)
- compare panel (Upstream | Spyre today | Spyre-next | torch-spyre) after the story chain
- evidence drawer for all remaining snippets and full extraction plan
- seam-level evidence plan (ordered targets with `target_id`, `evidence_kind`, anchor type, match status)
- inline snippets with pinned permalinks plus extraction rationale
- search, diffs, and curation export

Extraction is policy-driven and typed:

- `symbol` anchors for Python API/class/function seams
- `literal` / `contains_all` anchors for explicit line-level evidence
- `range` anchors only when explicit ranges are intentionally curated
- regex extractors are disabled by default (`allow_regex_targets=false`)
- strict generation fails when required targets do not match (`strict_target_matches=true`)

## Layout

```
code-atlas/
  README.md
  config.example.json
  config.json
  seams.json
  seam_narratives.json
  story.json
  scripts/
    generate_index.py
    extract_snippets.py
    build_permalinks.py
    serve.sh
  web/
    index.html
    app.js
    styles.css
    vendor/
      prism.js
      prism.css
      diff.js
    data/
      *.json + report.md (generated)
  data/
    *.json + report.md (generated)
```

## Configure repo roots

Edit `config.json` if needed:

```json
{
  "repo_roots": {
    "vllm": "../vllm",
    "vllm-spyre": "../vllm-spyre",
    "torch-spyre": ".."
  },
  "seam_spec": "./seams.json",
  "seam_narratives": "./seam_narratives.json",
  "story_spec": "./story.json",
  "allow_regex_targets": false,
  "strict_target_matches": true
}
```

## Generate index/snippets/symbols (single command)

```bash
python3 scripts/generate_index.py --config config.json
```

This writes generated data to both:

- `data/`
- `web/data/`

## Run locally

```bash
python3 -m http.server 8000 --directory web
```

Open:

- [http://localhost:8000](http://localhost:8000)

## Validation checklist

- App loads locally with no console errors.
- Home view shows purpose, reading path, architecture layers, caveats, and glossary.
- Search works across snippets/symbols/grep hits.
- Seam map shows upstream + Spyre snippets.
- Seam detail panel shows question/why/checklist/pitfalls/repo lens.
- Snippet cards show repo, file, commit, lines, permalink, extraction rationale, and inline code.
- Diff page supports snippet-to-snippet and file-to-file comparison.
- Curation allows starring snippets and exporting Markdown bundle.
- `snippets.json` contains at least 30 snippets.

## Notes

- No external APIs are used.
- No modifications are made to `vllm/`, `vllm-spyre/`, or `torch_spyre/` source trees.
- Snippet extraction reads `git show <HEAD>:<path>` blobs, so inline code is pinned to the same commit as permalinks.
- Repo status metadata (branch, dirty tree, and whether `HEAD` appears in locally fetched `origin/*` refs) is included to make ref drift explicit.
