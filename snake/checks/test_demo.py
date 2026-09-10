"""无桌面显示的事件测试：验证暂停、单步、重置、退出和导出逻辑。"""

import json
import sys

import pytest
from PIL import Image

from snake import demo


def test_desktop_step_pause_reset_and_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    trace = tmp_path / "desktop.jsonl"
    monkeypatch.setattr(sys, "argv", ["demo", "--policy", "human", "--trace", str(trace)])
    pg = demo.pygame
    events = iter([
        [pg.event.Event(pg.KEYDOWN, key=pg.K_UP)],
        [pg.event.Event(pg.KEYDOWN, key=pg.K_LEFT)],
        [pg.event.Event(pg.KEYDOWN, key=pg.K_SPACE)],
        [pg.event.Event(pg.KEYDOWN, key=pg.K_SPACE)],
        [pg.event.Event(pg.KEYDOWN, key=pg.K_n)],
        [pg.event.Event(pg.KEYDOWN, key=pg.K_r)],
        [pg.event.Event(pg.KEYDOWN, key=pg.K_RIGHT)],
        [pg.event.Event(pg.QUIT)],
    ])
    monkeypatch.setattr(pg.event, "get", lambda: next(events))
    # 冻结时间，避免机器调度使本测试意外触发自动前进。
    monkeypatch.setattr(demo.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(demo.time, "sleep", lambda _: None)
    demo.main()
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [r["action"] for r in records if r["event"] == "step"] == [0, 1, 0, 2]
    resets = [r for r in records if r["event"] == "reset"]
    assert len(resets) == 2 and resets[0]["observation"] == resets[1]["observation"]
    assert records[-1]["reason"] == "user_closed"


def test_headless_export_has_trace_and_separate_preview_limit(monkeypatch, tmp_path):
    gif = tmp_path / "preview.gif"
    monkeypatch.setattr(sys, "argv", ["demo", "--headless", "--max-frames", "2", "--gif", str(gif)])
    demo.main()
    with Image.open(gif) as image:
        assert image.n_frames == 3 and image.size == (940, 630)
    records = [json.loads(line) for line in gif.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[-1]["reason"] == "preview_limit"
    assert not records[-2]["terminated"] and not records[-2]["truncated"]


def test_headless_human_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo", "--headless", "--policy", "human"])
    with pytest.raises(SystemExit) as result:
        demo.main()
    assert result.value.code == 2
