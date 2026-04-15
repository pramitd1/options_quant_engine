from pathlib import Path

import research.ml_models.gbt_model as gbt_model_mod
import research.ml_models.logreg_model as logreg_model_mod


def _exercise_loader(module, monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("stub")

    monkeypatch.setattr(module, "_cached_model", None)
    monkeypatch.setattr(module, "_cached_meta", None)
    monkeypatch.setattr(module, "_WARN_ONCE_KEYS", set())

    if module is gbt_model_mod:
        monkeypatch.setattr(module, "GBT_MODEL_PATH", model_path)
        monkeypatch.setattr(module, "GBT_META_PATH", tmp_path / "meta.json")
    else:
        monkeypatch.setattr(module, "LOGREG_MODEL_PATH", model_path)
        monkeypatch.setattr(module, "LOGREG_META_PATH", tmp_path / "meta.json")

    logged = {"warning": [], "exception": []}

    monkeypatch.setattr(
        module.joblib,
        "load",
        lambda path: (_ for _ in ()).throw(
            ValueError("<class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module.")
        ),
    )
    monkeypatch.setattr(module.logger, "warning", lambda msg, *args: logged["warning"].append(msg % args if args else msg))
    monkeypatch.setattr(module.logger, "exception", lambda msg, *args: logged["exception"].append(msg % args if args else msg))

    model, meta = module._load_model()

    assert model is None
    assert meta is None
    assert module._cached_model is module._UNAVAILABLE
    assert not logged["exception"]
    assert logged["warning"]


def test_gbt_loader_suppresses_known_numpy_pickle_compatibility_traceback(monkeypatch, tmp_path):
    _exercise_loader(gbt_model_mod, monkeypatch, tmp_path)


def test_logreg_loader_suppresses_known_numpy_pickle_compatibility_traceback(monkeypatch, tmp_path):
    _exercise_loader(logreg_model_mod, monkeypatch, tmp_path)
