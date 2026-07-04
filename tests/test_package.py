"""Package installation smoke tests."""

import subprocess
import sys
from pathlib import Path

import vdbmat


def test_package_imports_from_installed_source_tree() -> None:
    """The test must exercise uv's editable package installation."""
    package_file = Path(vdbmat.__file__).resolve()

    assert package_file.parts[-3:] == ("src", "vdbmat", "__init__.py")
    assert vdbmat.__version__ == "0.1.0"


def test_core_and_adapter_module_import_without_loading_mitsuba() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vdbmat.core; import vdbmat.exporters.mitsuba; "
            "assert 'mitsuba' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
