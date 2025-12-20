"""Property-based tests for MCP path security denylist enforcement."""

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st


class TestSensitiveDenylistEnforcement:
    """
    **Feature: mcp-path-security, Property 3: Sensitive Denylist Enforcement**
    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        user_patterns=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
                min_size=1,
                max_size=10,
            ),
            min_size=0,
            max_size=5,
        )
    )
    def test_sensitive_files_never_yielded(self, user_patterns: list[str]):
        """Sensitive files should be excluded regardless of user patterns."""
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            sensitive_files = [
                ".ssh/id_rsa",
                ".ssh/id_ed25519",
                ".gnupg/private-keys-v1.d/key.key",
                "config/.env",
                "config/.env.local",
                "certs/server.pem",
                "certs/server.key",
                "certs/ca.crt",
                "secrets/keystore.p12",
                ".netrc",
                ".npmrc",
                ".pypirc",
            ]

            normal_files = ["src/main.py", "src/utils.py", "tests/test_main.py"]

            for rel_path in sensitive_files + normal_files:
                file_path = tmpdir_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(f"# Content of {rel_path}\n", encoding="utf-8")

            scanner = FileScanner(extensions={".py", ".pem", ".key", ".crt", ".p12"}, ignore_patterns=user_patterns)
            scanned_files = list(scanner.scan(tmpdir_path))

            for rel_path in sensitive_files:
                file_path = (tmpdir_path / rel_path).resolve()
                assert file_path not in {sf.path for sf in scanned_files}

    @settings(max_examples=100, deadline=None)
    @given(
        sensitive_pattern=st.sampled_from(
            [".ssh", ".gnupg", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", ".env", ".netrc", ".npmrc", ".pypirc"]
        )
    )
    def test_exact_match_patterns_excluded(self, sensitive_pattern: str):
        """Exact sensitive patterns should be excluded regardless of depth."""
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            locations = [
                sensitive_pattern,
                f"subdir/{sensitive_pattern}",
                f"deep/nested/path/{sensitive_pattern}",
            ]

            for loc in locations:
                file_path = tmpdir_path / loc
                file_path.parent.mkdir(parents=True, exist_ok=True)
                if not sensitive_pattern.startswith("."):
                    file_path.write_text("sensitive content", encoding="utf-8")
                else:
                    file_path.mkdir(exist_ok=True)
                    (file_path / "test.py").write_text("# test", encoding="utf-8")

            normal_file = tmpdir_path / "normal.py"
            normal_file.write_text("# normal file", encoding="utf-8")

            scanner = FileScanner(extensions={".py", ""}, ignore_patterns=[])
            scanned_files = list(scanner.scan(tmpdir_path))

            for sf in scanned_files:
                path_parts = sf.path.parts
                assert sensitive_pattern not in path_parts

    @settings(max_examples=100, deadline=None)
    @given(extension=st.sampled_from([".pem", ".key", ".p12", ".pfx", ".crt", ".keystore"]))
    def test_glob_pattern_extensions_excluded(self, extension: str):
        """Sensitive extensions should always be excluded."""
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sensitive_files = [
                f"server{extension}",
                f"client{extension}",
                f"ca{extension}",
                f"subdir/nested{extension}",
            ]

            for rel_path in sensitive_files:
                file_path = tmpdir_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("sensitive content", encoding="utf-8")

            normal_file = tmpdir_path / "normal.py"
            normal_file.write_text("# normal", encoding="utf-8")

            scanner = FileScanner(extensions={".py", extension}, ignore_patterns=[])
            scanned_files = list(scanner.scan(tmpdir_path))

            for sf in scanned_files:
                assert sf.path.suffix != extension

    @settings(max_examples=100, deadline=None)
    @given(
        env_variant=st.sampled_from(
            [".env", ".env.local", ".env.production", ".env.development", ".env.test", ".env.staging"]
        )
    )
    def test_env_file_variants_excluded(self, env_variant: str):
        """Any .env variant should be excluded from scanning."""
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            locations = [env_variant, f"config/{env_variant}", f"app/settings/{env_variant}"]

            for loc in locations:
                file_path = tmpdir_path / loc
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("SECRET_KEY=xxx", encoding="utf-8")

            normal_file = tmpdir_path / "app.py"
            normal_file.write_text("# app", encoding="utf-8")

            scanner = FileScanner(extensions={".py", ""}, ignore_patterns=[])
            scanned_files = list(scanner.scan(tmpdir_path))

            for sf in scanned_files:
                assert not sf.path.name.startswith(".env")

    def test_denylist_cannot_be_overridden_by_empty_patterns(self):
        """Sensitive denylist should apply even with empty ignore patterns."""
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            sensitive_file = tmpdir_path / "id_rsa"
            sensitive_file.write_text("private key content", encoding="utf-8")
            env_file = tmpdir_path / ".env"
            env_file.write_text("SECRET=xxx", encoding="utf-8")
            normal_file = tmpdir_path / "main.py"
            normal_file.write_text("# main", encoding="utf-8")

            scanner = FileScanner(extensions={".py", ""}, ignore_patterns=[])
            scanned_files = list(scanner.scan(tmpdir_path))
            scanned_names = {sf.path.name for sf in scanned_files}

            assert "id_rsa" not in scanned_names
            assert ".env" not in scanned_names
            assert "main.py" in scanned_names


class TestSensitiveDenylistPatterns:
    """
    **Feature: mcp-path-security, Property 4: Error Message Path Inclusion**
    **Validates: Requirements 2.4**
    """

    def test_required_ssh_patterns_in_denylist(self):
        """Verify SSH-related patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        ssh_patterns = {".ssh", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
        for pattern in ssh_patterns:
            assert pattern in SENSITIVE_DENYLIST

    def test_required_ssh_pub_patterns_in_denylist(self):
        """Verify SSH public key patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        ssh_pub_patterns = {"id_rsa.pub", "id_ed25519.pub", "id_ecdsa.pub", "id_dsa.pub"}
        for pattern in ssh_pub_patterns:
            assert pattern in SENSITIVE_DENYLIST

    def test_required_gnupg_pattern_in_denylist(self):
        """Verify GPG-related patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        assert ".gnupg" in SENSITIVE_DENYLIST

    def test_required_certificate_patterns_in_denylist(self):
        """Verify certificate and key patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        cert_patterns = {"*.pem", "*.key", "*.p12", "*.pfx", "*.crt", "*.keystore"}
        for pattern in cert_patterns:
            assert pattern in SENSITIVE_DENYLIST

    def test_required_env_patterns_in_denylist(self):
        """Verify environment file patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        env_patterns = {".env", ".env.*"}
        for pattern in env_patterns:
            assert pattern in SENSITIVE_DENYLIST

    def test_required_auth_config_patterns_in_denylist(self):
        """Verify authentication config patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        auth_patterns = {".netrc", ".npmrc", ".pypirc"}
        for pattern in auth_patterns:
            assert pattern in SENSITIVE_DENYLIST

    def test_denylist_is_frozenset(self):
        """SENSITIVE_DENYLIST should be immutable."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        assert isinstance(SENSITIVE_DENYLIST, frozenset)

    def test_matches_sensitive_denylist_exact_match(self):
        """Test _matches_sensitive_denylist for exact filename matches."""
        from aci.core.file_scanner import FileScanner

        scanner = FileScanner()
        exact_matches = [
            ".ssh",
            ".gnupg",
            "id_rsa",
            "id_ed25519",
            "id_ecdsa",
            "id_dsa",
            "id_rsa.pub",
            "id_ed25519.pub",
            ".env",
            ".netrc",
            ".npmrc",
            ".pypirc",
        ]

        for filename in exact_matches:
            path = Path(f"/some/path/{filename}")
            assert scanner._matches_sensitive_denylist(path)

    def test_matches_sensitive_denylist_glob_patterns(self):
        """Test _matches_sensitive_denylist for glob pattern matches."""
        from aci.core.file_scanner import FileScanner

        scanner = FileScanner()
        glob_matches = [
            ("server.pem", "*.pem"),
            ("private.key", "*.key"),
            ("cert.p12", "*.p12"),
            ("keystore.pfx", "*.pfx"),
            ("ca.crt", "*.crt"),
            ("app.keystore", "*.keystore"),
            (".env.local", ".env.*"),
            (".env.production", ".env.*"),
            (".env.development", ".env.*"),
        ]

        for filename, _pattern in glob_matches:
            path = Path(f"/some/path/{filename}")
            assert scanner._matches_sensitive_denylist(path)

    def test_matches_sensitive_denylist_non_sensitive_files(self):
        """_matches_sensitive_denylist should ignore normal files."""
        from aci.core.file_scanner import FileScanner

        scanner = FileScanner()
        non_sensitive = [
            "main.py",
            "config.yaml",
            "README.md",
            "package.json",
            "Dockerfile",
            "requirements.txt",
            "test_env.py",
            "ssh_utils.py",
            "key_manager.py",
        ]

        for filename in non_sensitive:
            path = Path(f"/some/path/{filename}")
            assert not scanner._matches_sensitive_denylist(path)

    def test_all_required_patterns_present(self):
        """Comprehensive check of required patterns."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        required_patterns = {
            ".ssh",
            ".gnupg",
            "id_rsa",
            "id_ed25519",
            "id_rsa.pub",
            "id_ed25519.pub",
            "id_ecdsa",
            "id_dsa",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "*.crt",
            "*.keystore",
            ".env",
            ".env.*",
            ".netrc",
            ".npmrc",
            ".pypirc",
        }

        missing_patterns = required_patterns - SENSITIVE_DENYLIST
        assert not missing_patterns
