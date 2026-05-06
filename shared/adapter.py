from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_module(module_name: str, file_path: Path, patch_page_config: bool = True) -> ModuleType:
    original_set_page_config = st.set_page_config
    if patch_page_config:
        st.set_page_config = lambda *args, **kwargs: None

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        st.set_page_config = original_set_page_config


def _safe_render(loader: Callable[[], None]) -> None:
    try:
        loader()
    except Exception as exc:
        st.error(f"页面加载失败：{exc}")
        st.info("建议先在本地安装依赖后重试，或使用 README 中的备用展示素材。")


def render_a2() -> None:
    def _run() -> None:
        module = _load_module("a2_app", BASE_DIR / "A2" / "app.py")
        module.main()

    _safe_render(_run)


def render_a3() -> None:
    def _run() -> None:
        module = _load_module("a3_core", BASE_DIR / "A3" / "核心代码.py")
        module.main()

    _safe_render(_run)


def render_a4() -> None:
    def _run() -> None:
        module = _load_module("a4_core", BASE_DIR / "A4" / "核心代码.py")
        module.main()

    _safe_render(_run)


def render_a5() -> None:
    def _run() -> None:
        module = _load_module("a5_core", BASE_DIR / "A5" / "核心代码.py")
        module.main()

    _safe_render(_run)


def render_a6() -> None:
    def _run() -> None:
        module = _load_module("a6_core", BASE_DIR / "A6" / "核心代码.py")
        module.main()

    _safe_render(_run)


def render_a7() -> None:
    def _run() -> None:
        module = _load_module("a7_core", BASE_DIR / "A7" / "核心代码.py")
        module.main()

    _safe_render(_run)


def render_a8() -> None:
    def _run() -> None:
        _load_module("a8_core", BASE_DIR / "A8" / "核心代码.py")

    _safe_render(_run)


def render_a9() -> None:
    def _run() -> None:
        module = _load_module("a9_core", BASE_DIR / "A9" / "核心代码.py")
        module.main()

    _safe_render(_run)
