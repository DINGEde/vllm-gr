# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Behaviour of tools/pre_commit/check_spdx_header.py."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/pre_commit/check_spdx_header.py"
LICENSE = "# SPDX-License-Identifier: Apache-2.0\n"
PROJECT = "# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project\n"
UPSTREAM = "# SPDX-FileCopyrightText: Copyright 2025 The xLLM Authors. All Rights Reserved.\n"
SLASH_LICENSE = "// SPDX-License-Identifier: Apache-2.0\n"
SLASH_PROJECT = "// SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project\n"


class SPDXHeaderTests(unittest.TestCase):
    def run_hook(self, content: str, name: str = "sample.py") -> tuple[int, str]:
        """Run the hook on one file; return its exit code and the file afterwards."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(content, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True
            )
            return result.returncode, path.read_text(encoding="utf-8")

    def test_complete_headers_pass_unchanged(self) -> None:
        for content in [
            LICENSE + PROJECT + "x = 1\n",
            # several copyright holders
            LICENSE + PROJECT + UPSTREAM + "x = 1\n",
            # compound license expression
            "# SPDX-License-Identifier: Apache-2.0 AND MIT\n" + PROJECT + "x = 1\n",
        ]:
            with self.subTest(content=content):
                self.assertEqual(self.run_hook(content), (0, content))

    def test_missing_header_is_added_with_project_holder(self) -> None:
        code, text = self.run_hook("x = 1\n")
        self.assertEqual(code, 1)
        self.assertEqual(text, LICENSE + PROJECT + "x = 1\n")

    def test_missing_copyright_is_added_to_license_only_file(self) -> None:
        code, text = self.run_hook(LICENSE + "x = 1\n")
        self.assertEqual(code, 1)
        self.assertEqual(text, LICENSE + PROJECT + "x = 1\n")

    def test_copyright_without_license_is_reported_not_guessed(self) -> None:
        content = UPSTREAM + "x = 1\n"
        self.assertEqual(self.run_hook(content), (1, content))

    def test_empty_values_are_rejected(self) -> None:
        for content in [
            "# SPDX-License-Identifier:\n" + PROJECT + "x = 1\n",
            LICENSE + "# SPDX-FileCopyrightText:\nx = 1\n",
        ]:
            with self.subTest(content=content):
                self.assertEqual(self.run_hook(content), (1, content))

    def test_invalid_license_expression_is_rejected(self) -> None:
        for expression in ["NOT-A-LICENSE", "Apache-2.0 AND", "Apache-2.0 MIT"]:
            content = f"# SPDX-License-Identifier: {expression}\n" + PROJECT + "x = 1\n"
            with self.subTest(expression=expression):
                self.assertEqual(self.run_hook(content), (1, content))

    def test_shebang_is_preserved_when_header_is_added(self) -> None:
        code, text = self.run_hook("#!/usr/bin/env python\nx = 1\n")
        self.assertEqual(code, 1)
        self.assertTrue(text.startswith("#!/usr/bin/env python\n" + LICENSE + PROJECT))

    def test_proto_header_uses_slash_comments(self) -> None:
        # A "#" header would make the schema unparsable.
        code, text = self.run_hook('syntax = "proto3";\n', "sample.proto")
        self.assertEqual(code, 1)
        self.assertEqual(text, SLASH_LICENSE + SLASH_PROJECT + 'syntax = "proto3";\n')

    def test_complete_proto_header_passes_unchanged(self) -> None:
        content = SLASH_LICENSE + SLASH_PROJECT + 'syntax = "proto3";\n'
        self.assertEqual(self.run_hook(content, "sample.proto"), (0, content))

    def test_unknown_file_type_is_reported_not_rewritten(self) -> None:
        content = "some text\n"
        self.assertEqual(self.run_hook(content, "sample.txt"), (1, content))


if __name__ == "__main__":
    unittest.main()


pytestmark = pytest.mark.cpu_test
