"""Pytest configuration and environment fixtures."""
import pytest


def _ensure_streamlit_components_discovered() -> None:
    """Ensure Streamlit v2 component manifests are discovered when no runtime is active."""
    try:
        import streamlit.components.v2 as ccv2
        import streamlit.components.v2.get_bidi_component_manager as mgr_mod
        from streamlit.components.v2.component_manager import BidiComponentManager
        from streamlit.runtime import Runtime

        local_mgr = BidiComponentManager()
        local_mgr.discover_and_register_components(start_file_watching=False)

        def _get_manager() -> BidiComponentManager:
            if Runtime.exists():
                return Runtime.instance().bidi_component_registry
            return local_mgr

        mgr_mod.get_bidi_component_manager = _get_manager
        ccv2.get_bidi_component_manager = _get_manager
    except Exception:
        pass


_ensure_streamlit_components_discovered()


@pytest.fixture(scope="session", autouse=True)
def setup_streamlit_components():
    """Ensure Streamlit v2 component manifests remain available across test runs."""
    yield
