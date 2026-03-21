from __future__ import annotations

import app.streamlit_app as streamlit_app


def test_list_replay_files_chain_uses_source_aware_loader(monkeypatch, tmp_path):
    captured = {}

    def _fake_list(symbol, *, replay_dir, source_label):
        captured["symbol"] = symbol
        captured["replay_dir"] = replay_dir
        captured["source_label"] = source_label
        return (["chain_valid.csv"], [{"path": "chain_bad.csv", "reason": "empty_file"}])

    monkeypatch.setattr(streamlit_app, "list_replay_chain_snapshots", _fake_list)

    chain_files, skipped_files = streamlit_app._list_replay_files(
        str(tmp_path),
        "NIFTY",
        "chain",
        source_label="ICICI",
    )

    assert chain_files == ["chain_valid.csv"]
    assert skipped_files == [{"path": "chain_bad.csv", "reason": "empty_file"}]
    assert captured == {
        "symbol": "NIFTY",
        "replay_dir": str(tmp_path),
        "source_label": "ICICI",
    }
