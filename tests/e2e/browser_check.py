"""End-to-end browser check (Playwright, Chromium). Requires backend on :8000 and frontend on :5173.

    python tests/e2e/browser_check.py [--base http://127.0.0.1:5173]

Writes logs/e2e_report.json and screenshots to logs/screenshots/.
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "logs" / "screenshots"
PAGES = ["/", "/studio", "/lab", "/analytics", "/datasets", "/experiments", "/docs", "/ask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5173")
    a = ap.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)
    results = []

    def record(name, ok, detail=""):
        results.append({"test": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for vp_name, vp in (("desktop", {"width": 1440, "height": 900}), ("mobile", {"width": 390, "height": 844})):
            ctx = browser.new_context(viewport=vp)
            page = ctx.new_page()
            errors, failed = [], []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 500 else None)
            for path in PAGES:
                errors.clear(); failed.clear()
                page.goto(a.base + path, wait_until="networkidle")
                page.wait_for_timeout(1200)
                overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                page.screenshot(path=str(SHOTS / f"{vp_name}{path.replace('/', '_') or '_home'}.png"), full_page=vp_name == "desktop")
                record(f"[{vp_name}] {path} renders without console errors", not errors, "; ".join(errors[:3]))
                record(f"[{vp_name}] {path} no 5xx responses", not failed, "; ".join(failed[:3]))
                record(f"[{vp_name}] {path} no horizontal overflow", overflow <= 1, f"overflow {overflow}px")
            ctx.close()

        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))

        # Scroll reveal toggles both ways
        page.goto(a.base + "/", wait_until="networkidle")
        page.wait_for_timeout(800)
        page.mouse.wheel(0, 4000); page.wait_for_timeout(900)
        n_in_down = page.locator(".reveal.in").count()
        page.mouse.wheel(0, -8000); page.wait_for_timeout(900)
        n_in_up = page.locator(".reveal.in").count()
        record("Scroll-triggered reveals replay on scroll up/down", n_in_down > 0 and n_in_up > 0, f"in-view after down={n_in_down}, after up={n_in_up}")

        # Studio: real detection on a DAWN sample
        page.goto(a.base + "/studio", wait_until="networkidle")
        page.get_by_role("button", name="Sand").first.click()
        page.wait_for_selector("img[alt='Selected']", timeout=20000)
        t0 = time.time()
        page.get_by_role("button", name="Run Detection").click()
        page.wait_for_selector(".det-item, .det-list p", timeout=120000)
        n = page.locator(".det-item").count()
        page.screenshot(path=str(SHOTS / "flow_studio_result.png"), full_page=True)
        record("Studio: Run Detection returns real detections", n > 0, f"{n} detections in {time.time() - t0:.1f}s")
        with page.expect_download() as dl:
            page.get_by_role("link", name="JSON").click()
        record("Studio: JSON download works", dl.value.suggested_filename.endswith(".json"), dl.value.suggested_filename)
        # error state: unsupported file
        page.set_input_files("input[type=file]", files=[{"name": "x.txt", "mimeType": "text/plain", "buffer": b"hello"}])
        record("Studio: unsupported file shows an error", page.locator(".alert.error").count() > 0)

        # Lab
        page.goto(a.base + "/lab", wait_until="networkidle")
        page.wait_for_selector("img[alt='Selected']", timeout=20000)
        page.get_by_role("button", name="Process image").click()
        page.wait_for_selector(".map-tile img", timeout=120000)
        page.wait_for_timeout(800)
        tiles = page.locator(".map-tile").count()
        page.screenshot(path=str(SHOTS / "flow_lab_result.png"), full_page=True)
        record("Lab: processing returns all intermediate maps", tiles >= 9, f"{tiles} tiles")
        record("Lab: provenance banner shown", page.locator("text=illustrative").count() > 0)

        # Datasets gallery + modal
        page.goto(a.base + "/datasets", wait_until="networkidle")
        page.wait_for_selector(".thumb", timeout=30000)
        page.locator(".thumb").first.click()
        page.wait_for_selector(".modal img", timeout=15000)
        page.wait_for_timeout(600)
        page.screenshot(path=str(SHOTS / "flow_dataset_modal.png"))
        record("Datasets: gallery + annotation modal", page.locator(".modal .tbl tr").count() > 1)
        page.keyboard.press("Escape")

        # Analytics reads saved results
        page.goto(a.base + "/analytics", wait_until="networkidle")
        page.wait_for_timeout(1500)
        record("Analytics: renders measured tables", page.locator("text=All variants").count() > 0 and page.locator(".recharts-surface").count() > 0,
               f"{page.locator('.recharts-surface').count()} charts")

        # Ask AI page
        page.goto(a.base + "/ask", wait_until="networkidle")
        page.get_by_label("Message").fill("What is the newly saturated ratio?")
        page.get_by_role("button", name="Send").click()
        page.wait_for_selector(".msg.assistant .pill", timeout=60000)
        page.wait_for_timeout(1500)
        txt = page.locator(".msg.assistant").last.inner_text()
        page.screenshot(path=str(SHOTS / "flow_ask.png"))
        record("Ask AI: streams an answer with mode label", ("fallback" in txt.lower() or "llm" in txt.lower()) and len(txt) > 80, txt[:90].replace("\n", " "))

        # Reduced motion: page usable with animations disabled
        rctx = browser.new_context(viewport={"width": 1280, "height": 800}, reduced_motion="reduce")
        rp = rctx.new_page()
        rp.goto(a.base + "/", wait_until="networkidle")
        rp.wait_for_timeout(700)
        op = rp.evaluate("getComputedStyle(document.querySelector('.reveal')).opacity")
        record("Reduced motion: content visible without animation", float(op) == 1.0, f"opacity {op}")
        rctx.close()
        record("No uncaught page errors during flows", not errs, "; ".join(errs[:3]))
        browser.close()

    out = ROOT / "logs" / "e2e_report.json"
    out.write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": a.base, "results": results}, indent=2))
    n_fail = sum(r["status"] == "FAIL" for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} passed -> {out}")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
