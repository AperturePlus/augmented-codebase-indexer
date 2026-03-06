from unittest.mock import patch

from aci.core.qdrant_launcher import ensure_qdrant_running


def test_ensure_qdrant_running_skips_nested_docker_inside_container():
    with patch("aci.core.qdrant_launcher._is_port_open", return_value=False), patch(
        "aci.core.qdrant_launcher.os.environ",
        {"container": "docker"},
    ), patch("aci.core.qdrant_launcher.subprocess.run") as mock_run, patch(
        "aci.core.qdrant_launcher.logger.warning"
    ) as mock_warning:
        ensure_qdrant_running(host="localhost", port=6333)

    mock_run.assert_not_called()
    mock_warning.assert_called_once()
    assert "inside container" in mock_warning.call_args.args[0]
