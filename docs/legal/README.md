# Copyright and source attribution

This page records how vLLM-GR marks authorship and third-party code. It follows
the convention of vLLM-Omni. It is an engineering convention, not a legal
determination of ownership.

## File headers

Original project code uses exactly these two lines:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
```

No year is added, because no verified year range exists.

A file adapted from another project keeps the project line and records its
origin in plain comments below it. Upstream copyright is written as a plain
comment, and only when the upstream file carries one:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright 2025 The xLLM Authors. All Rights Reserved.
# Adapted from https://github.com/xLLM-AI/xllm-ops
```

- Use the repository URL after `Adapted from`. Commit-pinned revisions and the
  upstream path of each file are listed in the root `NOTICE`.
- If the adapted file is under a different license, say so in
  `SPDX-License-Identifier` (for example `Apache-2.0 AND BSD-3-Clause`).
- A file copied without changes keeps the upstream `SPDX-FileCopyrightText` and
  does not get the project line.
- Do not add import dates, per-file change lists, or repeated "Adapted from"
  blocks; the change history lives in git.

SPDX completeness does not prove who owns the code. The root `LICENSE` and
`NOTICE` do not replace file-level notices.

## Third-party license texts

The root `NOTICE` lists what was taken from each upstream project, including the
revision and the upstream path of every adapted file. It reproduces the MIT and
BSD-3-Clause texts, so they are present in every distribution. Apache-2.0
upstream code needs no separate copy of the license text: that text is the root
`LICENSE`, and `NOTICE` records each upstream's own copyright line.

## Checks

```text
pre-commit run check-spdx-header --all-files
python -m unittest tests.legal.test_spdx_header
python tools/generate_grpc_proto.py --check
```

The SPDX hook requires a non-empty `SPDX-License-Identifier` holding a valid SPDX
expression and a non-empty `SPDX-FileCopyrightText`, and accepts several
copyright lines. It adds a header with the project holder to a file that has
neither line. It does not guess a license: a file with a copyright line but no
license line is reported for a person to resolve.

It does not verify that the named holder is correct for a given file. A copied
file that lacks a header would be labelled as project code, so check provenance
before relying on it.
