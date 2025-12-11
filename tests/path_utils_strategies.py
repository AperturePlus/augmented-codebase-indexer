"""Shared strategies for path utilities property tests."""

import os
import tempfile
from pathlib import Path

from hypothesis import strategies as st

from aci.core.path_utils import POSIX_SYSTEM_DIRS, WINDOWS_SYSTEM_DIRS

# Windows reserved device names that can't be used as directory names
WINDOWS_RESERVED = frozenset(
    [
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    ]
)


@st.composite
def non_existent_paths(draw):
    """Generate paths that are very unlikely to exist."""
    components = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
                min_size=1,
                max_size=10,
            ),
            min_size=2,
            max_size=5,
        )
    )
    suffix = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=8,
            max_size=16,
        )
    )
    return os.path.join(tempfile.gettempdir(), *components, f"nonexistent_{suffix}")


@st.composite
def posix_system_paths(draw):
    """Generate paths under POSIX system directories."""
    base = draw(st.sampled_from(list(POSIX_SYSTEM_DIRS)))
    subpath = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P"), min_codepoint=45, max_codepoint=122),
            min_size=0,
            max_size=20,
        ).filter(lambda x: "/" not in x and "\\" not in x)
    )
    if subpath:
        return f"{base}/{subpath}"
    return base


@st.composite
def windows_system_paths(draw):
    """Generate paths under Windows system directories."""
    drive = draw(st.sampled_from(["C:", "D:"]))
    sys_dir = draw(st.sampled_from(list(WINDOWS_SYSTEM_DIRS)))
    sys_dir_proper = sys_dir.title() if sys_dir != "program files (x86)" else "Program Files (x86)"
    subpath = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=0,
            max_size=20,
        )
    )
    if subpath:
        return f"{drive}\\{sys_dir_proper}\\{subpath}"
    return f"{drive}\\{sys_dir_proper}"

