import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import DAWN, needs_dawn, needs_weights


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app
    return TestClient(app)


def _png(w=96, h=64):
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_health_ready_system(client):
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.get("/api/ready").json()
    assert set(r) >= {"ready", "loading", "error"}
    assert "torch" in client.get("/api/system").json()


def test_variants_and_model_status(client):
    vs = client.get("/api/variants").json()
    assert {v["key"] for v in vs} >= {"b0_no_filter", "b2_erup", "sg_full"}
    st = client.get("/api/model/status").json()
    assert "weights_exist" in st and "checkpoints" in st


@pytest.mark.parametrize("payload,ctype", [(b"", "image/png"), (b"plain text", "text/plain"), (b"\x89PNG\r\n\x1a\n" + b"0" * 40, "image/png")])
def test_detect_rejects_bad_uploads(client, payload, ctype):
    r = client.post("/api/detect", files={"file": ("x.png", payload, ctype)})
    assert r.status_code == 400 and "detail" in r.json()


def test_detect_rejects_bad_params(client):
    png = _png()
    assert client.post("/api/detect", files={"file": ("a.png", png, "image/png")}, data={"variant": "nope"}).status_code == 400
    assert client.post("/api/detect", files={"file": ("a.png", png, "image/png")}, data={"conf": "5"}).status_code == 400
    assert client.post("/api/detect", files={"file": ("a.png", png, "image/png")}, data={"checkpoint": "../../secret.pt"}).status_code == 400


def test_upload_size_limit(client, monkeypatch):
    from backend.app import main
    monkeypatch.setattr(main.settings, "max_upload_mb", 0.001)
    r = client.post("/api/detect", files={"file": ("a.png", _png(200, 200), "image/png")})
    assert r.status_code == 400 and "too large" in r.json()["detail"]


@needs_weights
def test_detect_real_inference_and_download(client):
    src = (DAWN / "Fog" / "foggy-001.jpg").read_bytes() if (DAWN / "Fog").exists() else _png(320, 240)
    r = client.post("/api/detect", files={"file": ("f.jpg", src, "image/jpeg")}, data={"variant": "b0_no_filter", "conf": "0.25"})
    assert r.status_code == 200, r.text
    j = r.json()
    for d in j["detections"]:
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= j["image"]["width"] and 0 <= y1 < y2 <= j["image"]["height"] and 0.25 <= d["score"] <= 1
    if (DAWN / "Fog").exists():
        assert any(d["class_name"] == "car" for d in j["detections"])  # foggy-001 contains a car (GT)
    for k in ("annotated.jpg", "processed.png", "detections.json"):
        assert client.get(j["files"][k]).status_code == 200
    assert client.get("/api/results/doesnotexist/annotated.jpg").status_code == 404


@needs_weights
def test_process_lab_maps(client):
    r = client.post("/api/process", files={"file": ("a.png", _png(160, 120), "image/png")},
                    data={"variant": "sg_full", "bpw_source": "manual", "manual_strength": "0.8", "gate_override": "fixed"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j["maps"]) >= {"original", "bpw", "gate", "blended", "final", "luminance", "dark_channel", "saturation"}
    assert 0 <= j["gate_stats"]["min"] <= j["gate_stats"]["max"] <= 1
    assert "illustrative" in j["provenance"]
    assert client.post("/api/process", files={"file": ("a.png", _png(), "image/png")}, data={"variant": "b0_no_filter"}).status_code == 400


@needs_dawn
def test_dataset_endpoints(client):
    d = client.get("/api/datasets").json()
    names = {x["name"] for x in d["datasets"]}
    assert {"dawn", "voc", "rtts", "exdark", "bdd100k", "acdc", "foggy_cityscapes"} <= names
    s = client.get("/api/datasets/dawn/samples?weather=sand&limit=3").json()
    assert s["total"] == 323 and len(s["items"]) == 3
    sid = s["items"][0]["id"]
    assert client.get(f"/api/datasets/dawn/image?id={sid}").headers["content-type"] == "image/jpeg"
    assert client.get("/api/datasets/dawn/image?id=../../etc/passwd").status_code == 404
    assert client.get("/api/datasets/unknown/samples").status_code == 404
    assert client.get("/api/datasets/rtts/samples").status_code in (200, 404)
    assert client.put("/api/datasets/dawn/path", json={"root": "Z:/definitely/not/here"}).status_code == 400


def test_experiment_and_run_path_traversal(client):
    assert client.get("/api/experiments/..%2F..%2Fbackend").status_code == 404
    assert client.get("/api/training/runs/../../x/seed0").status_code == 404
    assert isinstance(client.get("/api/experiments").json(), list)


def test_training_job_validation(client):
    assert client.post("/api/training/jobs", json={"config": "paper_full", "variant": "sg_full"}).status_code == 400
    assert client.post("/api/training/jobs", json={"config": "smoke", "variant": "nope"}).status_code == 400


def test_ask_fallback_mode(client, monkeypatch):
    from backend.app import main
    monkeypatch.setattr(main.settings, "llm_enabled", "off")
    st = client.get("/api/ask/status").json()
    assert st["mode"] == "fallback"
    r = client.post("/api/ask", json={"messages": [{"role": "user", "content": "What is the saturation-aware loss L_sat?"}]})
    ev = _events(r.text)
    assert ev[0]["type"] == "meta" and ev[0]["mode"] == "fallback" and ev[0]["sources"]
    text = "".join(e["text"] for e in ev if e["type"] == "delta")
    assert "Local fallback mode" in text and ev[-1]["type"] == "done"


def test_ask_validation(client):
    assert client.post("/api/ask", json={"messages": []}).status_code == 422
    assert client.post("/api/ask", json={"messages": [{"role": "assistant", "content": "hi"}]}).status_code == 400
    assert client.post("/api/ask", json={"messages": [{"role": "system", "content": "hi"}]}).status_code == 422


def test_ask_configured_mode_with_mock_llm(client, monkeypatch):
    """Configured mode: the server calls the Anthropic SDK (mocked here, no network, no key in the frontend)."""
    import anthropic
    from backend.app import ask as ask_mod
    from backend.app import main
    calls = {}

    class FakeStream:
        def __init__(self, **kw):
            calls.update(kw)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        @property
        def text_stream(self):
            async def gen():
                for t in ["NSR is the ", "newly-saturated ratio [Paper]."]:
                    yield t
            return gen()

        async def get_final_message(self):
            return type("M", (), {"stop_reason": "end_turn"})()

    class FakeClient:
        def __init__(self, *a, **k):
            self.beta = type("B", (), {"messages": type("Mm", (), {"stream": staticmethod(lambda **kw: FakeStream(**kw))})()})()

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeClient)
    monkeypatch.setattr(main.settings, "llm_enabled", "on")
    monkeypatch.setattr(ask_mod.settings, "llm_enabled", "on")
    assert client.get("/api/ask/status").json()["mode"] == "llm"
    r = client.post("/api/ask", json={"messages": [{"role": "user", "content": "What is NSR?"}]})
    ev = _events(r.text)
    assert ev[0]["mode"] == "llm"
    assert "".join(e["text"] for e in ev if e["type"] == "delta") == "NSR is the newly-saturated ratio [Paper]."
    assert ev[-1] == {"type": "done", "stop_reason": "end_turn"}
    assert calls["model"] == main.settings.llm_model and "<context>" in calls["messages"][-1]["content"]
    assert calls["fallbacks"] == "default" and "system" in calls
