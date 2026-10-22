import json

import numpy as np
import pytest
from PIL import Image

from sgerup.config import make_config, parse_overrides, variant_names
from sgerup.data import datasets as D
from sgerup.data.class_maps import IGNORE, map_label
from sgerup.imaging import ImageError, letterbox, unletterbox_boxes, validate_upload

from conftest import needs_dawn


def _img(path, w=64, h=48):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), 128, np.uint8)).save(path)


def _voc_xml(path, objs, w=64, h=48):
    path.parent.mkdir(parents=True, exist_ok=True)
    o = "".join(f"<object><name>{n}</name><difficult>{int(d)}</difficult><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
                f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>" for n, b, d in objs)
    path.write_text(f"<annotation><size><width>{w}</width><height>{h}</height></size>{o}</annotation>")


def spec(fmt, root, **kw):
    return {"format": fmt, "root": str(root), "class_set": kw.pop("class_set", "protocol5"), **kw}


def collect(fmt, root, split=None, **kw):
    s = spec(fmt, root, **kw)
    return list(D.iter_samples("x", split, s))


# ------------------------------------------------------------------ parsers with synthetic fixtures
def test_voc_flat_rtts_format(tmp_path):
    _img(tmp_path / "JPEGImages" / "a.jpg")
    _voc_xml(tmp_path / "Annotations" / "a.xml", [("car", [1, 2, 30, 40], 0), ("truck", [5, 5, 20, 20], 0), ("dog", [1, 1, 5, 5], 0)])
    _img(tmp_path / "JPEGImages" / "b.jpg")  # missing annotation
    out = collect("voc_flat", tmp_path)
    samples = [s for s, _, _ in out if s]
    issues = [i for _, iss, _ in out for i in iss]
    assert len(samples) == 1 and samples[0].labels == ["car", IGNORE]  # dog dropped, truck ignore
    assert any("missing annotation" in i for i in issues)


def test_voc_difficult_and_splits(tmp_path):
    y = tmp_path / "VOC2007"
    (y / "ImageSets" / "Main").mkdir(parents=True)
    (y / "ImageSets" / "Main" / "trainval.txt").write_text("001\n")
    (y / "ImageSets" / "Main" / "test.txt").write_text("002\n")
    for i in ("001", "002"):
        _img(y / "JPEGImages" / f"{i}.jpg")
        _voc_xml(y / "Annotations" / f"{i}.xml", [("person", [1, 1, 20, 20], 0), ("bus", [2, 2, 30, 30], 1)])
    tv = [s for s, _, _ in collect("voc", tmp_path, "trainval") if s]
    assert [s.id for s in tv] == ["VOC2007/001"] and tv[0].labels == ["person", IGNORE]
    te = [s for s, _, _ in collect("voc", tmp_path, "test") if s]
    assert [s.id for s in te] == ["VOC2007/002"]


def test_invalid_boxes_reported(tmp_path):
    _img(tmp_path / "JPEGImages" / "a.jpg")
    _voc_xml(tmp_path / "Annotations" / "a.xml", [("car", [30, 2, 10, 40], 0), ("car", [0, 0, 500, 40], 0)])
    out = collect("voc_flat", tmp_path)
    issues = [i for _, iss, _ in out for i in iss]
    assert any("invalid box" in i for i in issues) and any("outside image" in i for i in issues)
    s = out[0][0]
    assert len(s.boxes) == 1 and s.boxes[0, 2] <= 64  # clipped


def test_exdark_format(tmp_path):
    _img(tmp_path / "ExDark" / "People" / "x.jpg")
    (tmp_path / "ExDark_Annno" / "People").mkdir(parents=True)
    (tmp_path / "ExDark_Annno" / "People" / "x.jpg.txt").write_text("% bbGt version=3\nPeople 1 2 10 20 0 0 0 0 0 0 0\nCup 3 3 4 4 0 0 0 0 0 0 0\n")
    s = [s for s, _, _ in collect("exdark", tmp_path, class_set="lowlight10") if s][0]
    assert s.labels == ["person", IGNORE] and s.boxes[0].tolist() == [1, 2, 11, 22]


def test_yolo_format(tmp_path):
    _img(tmp_path / "images" / "a.jpg", 100, 100)
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n")
    (tmp_path / "classes.txt").write_text("person\ncar\n")
    s = [s for s, _, _ in collect("yolo", tmp_path) if s][0]
    assert s.labels == ["car"] and np.allclose(s.boxes[0], [40, 30, 60, 70])


def test_yolo_without_names_reports(tmp_path):
    (tmp_path / "images").mkdir()
    out = collect("yolo", tmp_path)
    assert out[0][0] is None and "class-name file" in out[0][1][0]


def test_bdd_format(tmp_path):
    _img(tmp_path / "images" / "100k" / "val" / "a.jpg")
    _img(tmp_path / "images" / "100k" / "val" / "b.jpg")
    (tmp_path / "labels" / "det_20").mkdir(parents=True)
    (tmp_path / "labels" / "det_20" / "det_val.json").write_text(json.dumps([
        {"name": "a.jpg", "attributes": {"weather": "foggy", "timeofday": "daytime"},
         "labels": [{"category": "pedestrian", "box2d": {"x1": 1, "y1": 2, "x2": 10, "y2": 20}}, {"category": "truck", "box2d": {"x1": 1, "y1": 2, "x2": 10, "y2": 20}}]},
        {"name": "b.jpg", "attributes": {"weather": "clear", "timeofday": "daytime"}, "labels": []}]))
    ss = [s for s, _, _ in collect("bdd", tmp_path, split="val", weather=["foggy"], include_night=True) if s]
    assert len(ss) == 1 and ss[0].labels == ["person", IGNORE] and ss[0].weather == "foggy"


def test_coco_json_acdc_format(tmp_path):
    _img(tmp_path / "rgb_anon" / "fog" / "val" / "a.png")
    (tmp_path / "gt_detection").mkdir()
    (tmp_path / "gt_detection" / "instancesonly_val_gt_detection.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "fog/val/a.png", "width": 64, "height": 48}],
        "categories": [{"id": 26, "name": "car"}, {"id": 25, "name": "rider"}],
        "annotations": [{"image_id": 1, "category_id": 26, "bbox": [1, 2, 10, 10]}, {"image_id": 1, "category_id": 25, "bbox": [1, 2, 3, 3]}]}))
    s = [s for s, _, _ in collect("coco_json", tmp_path, split="val") if s][0]
    assert s.labels == ["car", IGNORE] and s.boxes[0].tolist() == [1, 2, 11, 12]


def test_cityscapes_format(tmp_path):
    _img(tmp_path / "leftImg8bit_foggy" / "val" / "city" / "c_000_000_leftImg8bit_foggy_beta_0.02.png")
    p = tmp_path / "gtFine" / "val" / "city"
    p.mkdir(parents=True)
    (p / "c_000_000_gtFine_polygons.json").write_text(json.dumps({"imgWidth": 64, "imgHeight": 48,
        "objects": [{"label": "bicycle", "polygon": [[1, 1], [10, 2], [5, 20]]}]}))
    s = [s for s, _, _ in collect("cityscapes", tmp_path, split="val", beta=0.02) if s][0]
    assert s.labels == ["bicycle"] and s.boxes[0].tolist() == [1, 1, 10, 20]


def test_missing_root_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "registry", lambda: {"ghost": spec("voc_flat", tmp_path / "nope")})
    rep = D.scan_dataset("ghost")
    assert rep["available"] is False and rep["status"] == "missing"


def test_scan_counts_from_files(tmp_path, monkeypatch):
    for i in range(3):
        _img(tmp_path / "JPEGImages" / f"{i}.jpg")
        _voc_xml(tmp_path / "Annotations" / f"{i}.xml", [("car", [1, 1, 20, 20], 0)])
    (tmp_path / "JPEGImages" / "bad.jpg").write_bytes(b"not an image")
    _voc_xml(tmp_path / "Annotations" / "bad.xml", [("car", [1, 1, 20, 20], 0)])
    monkeypatch.setattr(D, "registry", lambda: {"t": spec("voc_flat", tmp_path)})
    rep = D.scan_dataset("t", verify_images=True)
    assert rep["n_images_found"] == 4 and rep["n_valid_samples"] == 3 and rep["n_corrupt_images"] == 1
    assert rep["protocol_class_counts"] == {"car": 3}


def test_split_and_leakage(tmp_path):
    ss = [D.Sample(f"id{i}", str(tmp_path / f"{i}.jpg"), np.zeros((0, 4)), []) for i in range(50)]
    tr, va = D.split_train_val(ss, 0.1, 0)
    assert len(va) == 5 and not ({s.id for s in tr} & {s.id for s in va})
    assert D.split_train_val(ss, 0.1, 0)[1] == va  # deterministic
    assert D.check_disjoint({"a": tr, "b": va})["disjoint"]
    assert not D.check_disjoint({"a": tr, "b": tr[:2]})["disjoint"]


@needs_dawn
def test_real_dawn_counts():
    rep = D.scan_dataset("dawn")
    assert rep["n_valid_samples"] == 1027
    assert rep["weather_counts"] == {"fog": 300, "rain": 200, "sand": 323, "snow": 204}


def test_class_map():
    assert map_label("dawn", "Motorcycle", ["motorbike"]) == "motorbike"
    assert map_label("dawn", "truck", ["car"]) == IGNORE
    assert map_label("voc", "dog", ["car"]) is None


# ------------------------------------------------------------------ config and imaging
def test_config_variants_and_overrides():
    assert {"b0_no_filter", "b2_erup", "sg_full", "b4_global_gate", "b5_fixed_gate"} <= set(variant_names())
    c = make_config("configs/smoke.yaml", "sg_full", parse_overrides(["train.epochs=5", "loss.lambda_s=2.5"]))
    assert c.train.epochs == 5 and c.loss.lambda_s == 2.5 and c.frontend.gate_mode == "learned" and c.aug.p_sha == 0.5
    with pytest.raises(KeyError):
        make_config(variant="nope")
    with pytest.raises(ValueError):
        make_config(overrides={"train": {"bogus": 1}})


def test_letterbox_roundtrip():
    im = np.zeros((300, 600, 3), np.uint8)
    lb, r, pad = letterbox(im, 448)
    assert lb.shape == (448, 448, 3) and r == pytest.approx(448 / 600)
    box = np.array([[60.0, 30.0, 300.0, 200.0]])
    lbox = box * r + np.array([pad[0], pad[1], pad[0], pad[1]])
    assert np.allclose(unletterbox_boxes(lbox, r, pad, 600, 300), box, atol=1e-3)


def test_validate_upload(tmp_path):
    import io
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, "PNG")
    assert validate_upload(buf.getvalue(), 10_000) == "PNG"
    for bad in (b"", b"hello", buf.getvalue()[:30]):
        with pytest.raises(ImageError):
            validate_upload(bad, 10_000)
    with pytest.raises(ImageError):
        validate_upload(buf.getvalue(), 10)
    gif = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(gif, "GIF")
    with pytest.raises(ImageError, match="Unsupported"):
        validate_upload(gif.getvalue(), 10_000)
