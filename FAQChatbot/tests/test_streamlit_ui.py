from pathlib import Path


def test_streamlit_app_does_not_display_match_metadata() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app_streamlit.py"
    content = app_path.read_text(encoding="utf-8")

    assert "Matched FAQ" not in content
    assert "Confidence:" not in content
