import inspect
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from streamlit_drawable_canvas import st_canvas
import app


def test_canvas_image_does_not_use_unsupported_arguments():
    """Regression test: st_canvas must not be invoked with obsolete or unsupported kwargs like display_toolbar."""
    sig = inspect.signature(st_canvas)
    assert "display_toolbar" not in sig.parameters, "st_canvas unexpectedly gained display_toolbar."


def test_app_initialization_via_apptest():
    """Verify that app.py initializes cleanly under Streamlit without unhandled exceptions."""
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"app.py raised unhandled exceptions on startup: {[e.message for e in at.exception]}"
    assert at.title, "app.py should render a page title."
    assert at.title[0].value == "Handwritten Character & Word Recognition"


def test_app_modes_and_routing_smoke():
    """Verify Single Character and Handwritten Word modes, blank-input handling, and UI routing."""
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    assert at.header[0].value == "Single Character Recognition"

    # Click Recognize Character with no canvas drawing -> warning for missing/blank input
    at.button(key="recognize_character").click().run()
    assert len(at.exception) == 0
    assert any("Draw a character or upload an image first." in w.value for w in at.warning)

    # Switch sidebar to Handwritten Word mode
    at.sidebar.radio[0].set_value("Handwritten Word").run()
    assert len(at.exception) == 0
    assert at.header[0].value == "Handwritten Word Recognition"

    # Click Recognize Word with no canvas drawing -> warning for missing/blank input
    at.button(key="recognize_word").click().run()
    assert len(at.exception) == 0
    assert any("Draw a word or upload an image first." in w.value for w in at.warning)


def test_canvas_image_fallback_and_exception_handling():
    """Verify _canvas_image gracefully handles headless Streamlit exceptions while letting genuine errors bubble up."""
    from unittest.mock import patch
    import streamlit as st

    # 1. Headless fallback on StreamlitAPIException
    with patch("app.st_canvas", side_effect=st.errors.StreamlitAPIException("Headless component unavailable")):
        res = app._canvas_image(key="test_key", height=100, width=100, stroke_width=5)
        assert res.image_data is None
        assert res.json_data is None

    # 2. Headless fallback on RuntimeError
    with patch("app.st_canvas", side_effect=RuntimeError("Headless runtime missing")):
        res = app._canvas_image(key="test_key", height=100, width=100, stroke_width=5)
        assert res.image_data is None

    # 3. Headless fallback on unregistered component ValueError
    with patch("app.st_canvas", side_effect=ValueError("Component 'streamlit-drawable-canvas.streamlit_drawable_canvas' is not registered")):
        res = app._canvas_image(key="test_key", height=100, width=100, stroke_width=5)
        assert res.image_data is None

    # 4. Ensure genuine defects (e.g., TypeError, arbitrary ValueError) are not silenced
    with patch("app.st_canvas", side_effect=TypeError("Invalid canvas configuration")):
        with pytest.raises(TypeError, match="Invalid canvas configuration"):
            app._canvas_image(key="test_key", height=100, width=100, stroke_width=5)

    with patch("app.st_canvas", side_effect=ValueError("Invalid numeric parameter")):
        with pytest.raises(ValueError, match="Invalid numeric parameter"):
            app._canvas_image(key="test_key", height=100, width=100, stroke_width=5)
