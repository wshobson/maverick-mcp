"""Smoke test for maverick.technical's public API surface (`__init__.py`).

Every name Task 7 of the phase 5 plan promises -- the 8 indicators, the 8
analysis functions, `TechnicalService`, the 8 payload types, the settings
accessor, and `configure`/`register` -- must actually be importable from the
package root, not just from the submodule that defines it. Mirrors how
`maverick.portfolio`'s close-out (phase 4) exported its public surface via
`__init__.py`.
"""

from pydantic import BaseModel


def test_import_indicators_from_package():
    from maverick.technical import adx, atr, bollinger, ema, macd, rsi, sma, stochastic

    for fn in (sma, ema, rsi, macd, atr, bollinger, stochastic, adx):
        assert callable(fn)


def test_import_analysis_functions_from_package():
    from maverick.technical import (
        analyze_bollinger,
        analyze_macd,
        analyze_rsi,
        analyze_stochastic,
        analyze_trend,
        analyze_volume,
        generate_outlook,
        support_resistance,
    )

    for fn in (
        analyze_rsi,
        analyze_macd,
        analyze_stochastic,
        analyze_bollinger,
        analyze_volume,
        analyze_trend,
        support_resistance,
        generate_outlook,
    ):
        assert callable(fn)


def test_import_service_and_tool_wiring_from_package():
    from maverick.technical import TechnicalService, configure, register

    assert callable(TechnicalService)
    assert callable(configure)
    assert callable(register)


def test_import_types_from_package():
    from maverick.technical import (
        BollingerAnalysis,
        FullTechnicalAnalysis,
        LevelsResult,
        MACDAnalysis,
        RSIAnalysis,
        StochasticAnalysis,
        TrendAnalysis,
        VolumeAnalysis,
    )

    for model in (
        RSIAnalysis,
        MACDAnalysis,
        StochasticAnalysis,
        BollingerAnalysis,
        VolumeAnalysis,
        TrendAnalysis,
        LevelsResult,
        FullTechnicalAnalysis,
    ):
        assert issubclass(model, BaseModel)


def test_import_settings_accessor_from_package():
    from maverick.technical import get_technical_settings

    assert callable(get_technical_settings)


def test_all_matches_expected_export_set():
    import maverick.technical as technical

    assert set(technical.__all__) == {
        "TechnicalService",
        "RSIAnalysis",
        "MACDAnalysis",
        "StochasticAnalysis",
        "BollingerAnalysis",
        "VolumeAnalysis",
        "TrendAnalysis",
        "LevelsResult",
        "FullTechnicalAnalysis",
        "get_technical_settings",
        "configure",
        "register",
        "sma",
        "ema",
        "rsi",
        "macd",
        "atr",
        "bollinger",
        "stochastic",
        "adx",
        "analyze_rsi",
        "analyze_macd",
        "analyze_stochastic",
        "analyze_bollinger",
        "analyze_volume",
        "analyze_trend",
        "support_resistance",
        "generate_outlook",
    }
    assert len(technical.__all__) == len(set(technical.__all__)), (
        "__all__ has duplicate entries"
    )


def test_every_exported_name_resolves_on_the_package():
    import maverick.technical as technical

    for name in technical.__all__:
        assert hasattr(technical, name), (
            f"{name!r} listed in __all__ but not resolvable on the package"
        )
