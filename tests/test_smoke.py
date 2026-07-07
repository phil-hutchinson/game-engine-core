"""Smoke test: proves pytest discovery and configuration work end to end."""


def test_library_packages_import() -> None:
    import game_engine_core
    import game_engine_learning

    assert game_engine_core is not None
    assert game_engine_learning is not None
