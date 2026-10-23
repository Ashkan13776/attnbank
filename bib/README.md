# Per-family bibliography snippets

`attnbank/references.bib` is **generated**. Do not edit it.

Add entries to the snippet for your family — `bib/<family>.bib` — and run

```bash
python tools/merge_bib.py
```

from `lib/`. The merge concatenates every snippet, rejects duplicate keys,
and writes `attnbank/references.bib`.

This exists because the library is expanded by several people (or agents) at
once, and ten contributors appending to one `.bib` is the reliable way to
corrupt it. One file each, merged deterministically.

Every entry must be **verified**: the paper exists, and the title, authors and
year match the source you checked. A method whose `references` key does not
resolve fails the conformance suite.
