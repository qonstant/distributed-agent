#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tkinter as tk
from pathlib import Path
from typing import Optional

from imgui_bundle import hello_imgui, imgui, immapp
from lightrag.tools.lightrag_visualizer import graph_visualizer
from lightrag.tools.lightrag_visualizer.graph_visualizer import (
    CUSTOM_FONT,
    DEFAULT_FONT_CHI,
    DEFAULT_FONT_ENG,
    GraphViewer,
    draw_text_with_bg,
    show_file_dialog,
)


DEFAULT_GRAPH = Path(__file__).resolve().parents[1] / "out" / "lightrag" / "graph_chunk_entity_relation.graphml"

if not hasattr(imgui, "set_window_font_scale"):
    # Newer imgui_bundle builds removed this legacy helper used by LightRAG's viewer.
    imgui.set_window_font_scale = lambda _scale: None  # type: ignore[attr-defined]


def _load_font() -> None:
    """LightRAG's bundled viewer sets an imgui_bundle attribute that newer builds removed."""
    font_filename = CUSTOM_FONT
    io = imgui.get_io()
    font_size_pixels = 14
    asset_dir = Path(graph_visualizer.__file__).resolve().parent / "assets"

    if not os.path.isfile(font_filename):
        font_filename = str(asset_dir / font_filename)
    if os.path.isfile(font_filename):
        custom_font = io.fonts.add_font_from_file_ttf(
            filename=font_filename,
            size_pixels=font_size_pixels,
            glyph_ranges_as_int_list=io.fonts.get_glyph_ranges_chinese_full(),
        )
        io.font_default = custom_font
        return

    eng_font = asset_dir / DEFAULT_FONT_ENG
    chi_font = asset_dir / DEFAULT_FONT_CHI
    if eng_font.exists():
        io.fonts.add_font_from_file_ttf(filename=str(eng_font), size_pixels=font_size_pixels)
    if chi_font.exists():
        font_config = imgui.ImFontConfig()
        font_config.merge_mode = True
        io.font_default = io.fonts.add_font_from_file_ttf(
            filename=str(chi_font),
            size_pixels=font_size_pixels,
            font_cfg=font_config,
            glyph_ranges_as_int_list=io.fonts.get_glyph_ranges_chinese_full(),
        )


def run_viewer(graph_path: Optional[Path]) -> None:
    viewer = GraphViewer()
    show_fps = True
    text_bg_color = imgui.IM_COL32(0, 0, 0, 100)
    loaded_initial_graph = False

    def gui() -> None:
        nonlocal loaded_initial_graph
        if not viewer.initialized:
            viewer.setup()
            if graph_path and graph_path.exists() and not loaded_initial_graph:
                viewer.load_file(str(graph_path))
                loaded_initial_graph = True

        viewer.window_width = int(imgui.get_window_width())
        viewer.window_height = int(imgui.get_window_height())
        viewer.handle_keyboard_input()
        viewer.handle_mouse_interaction()

        style = imgui.get_style()
        window_bg_color = style.color_(imgui.Col_.window_bg.value)
        window_bg_color.w = 0.8
        style.set_color_(imgui.Col_.window_bg.value, window_bg_color)

        imgui.begin("Graph Controls")
        if graph_path:
            imgui.text(f"Graph: {graph_path.name}")
        if imgui.button("Load GraphML"):
            filepath = show_file_dialog()
            if filepath:
                viewer.load_file(filepath)

        if viewer.show_load_error:
            imgui.push_style_color(imgui.Col_.text, (1.0, 0.0, 0.0, 1.0))
            imgui.text(f"Error loading file: {viewer.error_message}")
            imgui.pop_style_color()

        imgui.separator()
        imgui.text("Camera Controls:")
        imgui.bullet_text("Hold Right Mouse - Look around")
        imgui.bullet_text("W/S - Move forward/backward")
        imgui.bullet_text("A/D - Move left/right")
        imgui.bullet_text("Q/E - Move up/down")
        imgui.bullet_text("Left Mouse - Select node")
        imgui.bullet_text("Wheel - Change movement speed")

        imgui.separator()
        _, viewer.move_speed = imgui.slider_float("Movement Speed", viewer.move_speed, 0.01, 2.0)
        _, viewer.mouse_sensitivity = imgui.slider_float("Mouse Sensitivity", viewer.mouse_sensitivity, 0.01, 0.5)

        imgui.separator()
        imgui.begin_horizontal("buttons")
        if imgui.button("Reset Camera"):
            viewer.reset_view()
        if imgui.button("Update Layout") and viewer.graph:
            viewer.update_layout()
        imgui.end_horizontal()
        imgui.end()

        viewer.render_node_details()
        viewer.render_settings()

        if show_fps:
            imgui.set_window_font_scale(1)
            fps_text = f"FPS: {hello_imgui.frame_rate():.1f}"
            text_size = imgui.calc_text_size(fps_text)
            draw_text_with_bg(fps_text, (10, viewer.window_height - text_size.y - 10), text_size, text_bg_color)

        if viewer.highlighted_node:
            imgui.set_window_font_scale(1)
            node_text = f"Node ID: {viewer.highlighted_node.label}"
            text_size = imgui.calc_text_size(node_text)
            cursor_pos = (viewer.window_width - text_size.x - 10, viewer.window_height - text_size.y - 10)
            draw_text_with_bg(node_text, cursor_pos, text_size, text_bg_color)

        window_bg_color.w = 0
        style.set_color_(imgui.Col_.window_bg.value, window_bg_color)
        viewer.render_labels()

    def custom_background() -> None:
        if viewer.initialized:
            viewer.render()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_geometry.size = (viewer.window_width, viewer.window_height)
    runner_params.app_window_params.window_title = "LightRAG GraphML Viewer"
    runner_params.callbacks.show_gui = gui
    runner_params.callbacks.custom_background = custom_background
    runner_params.callbacks.load_additional_fonts = _load_font

    tk_root = tk.Tk()
    tk_root.withdraw()
    immapp.run(runner_params)
    tk_root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the LightRAG GraphML viewer with a compatibility font patch.")
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH), help="GraphML file to auto-load")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = Path(args.graph).expanduser().resolve() if args.graph else None
    run_viewer(graph)


if __name__ == "__main__":
    main()
