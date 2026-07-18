"""Structural checks for the new maverick package."""


def test_version_is_importable():
    import maverick

    assert maverick.__version__ == "1.0.0.dev0"
