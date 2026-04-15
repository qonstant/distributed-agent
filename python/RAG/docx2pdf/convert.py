#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
COMMON_LIBREOFFICE_PATHS = (
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/Applications/OpenOffice.app/Contents/MacOS/soffice",
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
)
MICROSOFT_WORD_APP = Path("/Applications/Microsoft Word.app")


def discover_docx_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.docx")
        if path.is_file() and not path.name.startswith("~$")
    )


def find_libreoffice_binary() -> str | None:
    for command in ("soffice", "libreoffice", "lowriter"):
        resolved = shutil.which(command)
        if resolved:
            return resolved
    for raw_path in COMMON_LIBREOFFICE_PATHS:
        candidate = Path(raw_path)
        if candidate.exists():
            return str(candidate)
    return None


def find_word_app() -> Path | None:
    return MICROSOFT_WORD_APP if MICROSOFT_WORD_APP.exists() else None


def build_converter(engine: str) -> tuple[str, Callable[[Path, Path], None]]:
    requested = (engine or "auto").strip().lower()
    if requested not in {"auto", "libreoffice", "docx2pdf", "word"}:
        raise RuntimeError(f"unsupported engine: {engine}")

    if requested in {"auto", "word"}:
        word_app = find_word_app()
        if word_app is not None:
            return "word", _word_converter(word_app)
        if requested == "word":
            raise RuntimeError(
                "Microsoft Word.app was not found in /Applications. "
                "Install Microsoft Word or choose a different engine."
            )

    if requested in {"auto", "libreoffice"}:
        soffice = find_libreoffice_binary()
        if soffice:
            return "libreoffice", _libreoffice_converter(soffice)
        if requested == "libreoffice":
            raise RuntimeError(
                "LibreOffice executable not found. Install LibreOffice or choose --engine docx2pdf."
            )

    if requested in {"auto", "docx2pdf"}:
        try:
            from docx2pdf import convert as docx2pdf_convert
        except Exception as exc:
            if requested == "docx2pdf":
                raise RuntimeError(
                    "docx2pdf package is not available in this Python environment."
                ) from exc
        else:
            return "docx2pdf", _docx2pdf_converter(docx2pdf_convert)

    raise RuntimeError(
        "No DOCX->PDF conversion engine is available. "
        "Install Microsoft Word, LibreOffice, or the Python package docx2pdf."
    )


def _libreoffice_converter(soffice_binary: str) -> Callable[[Path, Path], None]:
    def convert_one(src_path: Path, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            soffice_binary,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(dst_path.parent),
            str(src_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            message = stderr or stdout or f"exit code {result.returncode}"
            raise RuntimeError(message)
        produced = dst_path.parent / f"{src_path.stem}.pdf"
        if not produced.exists():
            raise RuntimeError("LibreOffice finished without producing a PDF file")
        if produced != dst_path:
            if dst_path.exists():
                dst_path.unlink()
            produced.replace(dst_path)

    return convert_one


def _docx2pdf_converter(docx2pdf_convert) -> Callable[[Path, Path], None]:
    def convert_one(src_path: Path, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        docx2pdf_convert(str(src_path), str(dst_path))
        if not dst_path.exists():
            raise RuntimeError("docx2pdf finished without producing a PDF file")

    return convert_one


def _word_converter(word_app: Path) -> Callable[[Path, Path], None]:
    script = f"""
on run argv
    set inputPath to POSIX file (item 1 of argv)
    set outputPath to item 2 of argv
    tell application "{word_app.stem}"
        activate
        open inputPath read only true add to recent files false
        save as active document file name outputPath file format format PDF add to recent files false
        close active document saving no
    end tell
end run
""".strip()

    def convert_one(src_path: Path, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "osascript",
            "-e",
            script,
            "--",
            str(src_path),
            str(dst_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            message = stderr or stdout or f"exit code {result.returncode}"
            if "Not authorized" in message or "1743" in message:
                message += (
                    " (macOS may require granting Automation permission for Terminal/Python to control Microsoft Word)"
                )
            raise RuntimeError(message)
        if not dst_path.exists():
            raise RuntimeError("Microsoft Word finished without producing a PDF file")

    return convert_one


def convert_all(
    input_dir: Path,
    output_dir: Path,
    *,
    engine: str,
    overwrite: bool,
) -> int:
    if not input_dir.exists():
        raise RuntimeError(f"input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise RuntimeError(f"input path is not a directory: {input_dir}")

    files = discover_docx_files(input_dir)
    if not files:
        print(f"[docx2pdf] no .docx files found in {input_dir}")
        return 0

    engine_name, convert_one = build_converter(engine)
    print(f"[docx2pdf] engine: {engine_name}")
    print(f"[docx2pdf] input : {input_dir}")
    print(f"[docx2pdf] output: {output_dir}")

    converted = 0
    skipped = 0
    failed = 0

    for src_path in files:
        rel_path = src_path.relative_to(input_dir)
        dst_path = output_dir / rel_path.with_suffix(".pdf")
        if dst_path.exists() and not overwrite:
            print(f"[skip] {rel_path} -> {dst_path.relative_to(output_dir)}")
            skipped += 1
            continue

        try:
            convert_one(src_path, dst_path)
        except Exception as exc:
            print(f"[fail] {rel_path}: {exc}", file=sys.stderr)
            failed += 1
            continue

        print(f"[ok]   {rel_path} -> {dst_path.relative_to(output_dir)}")
        converted += 1

    print(
        f"[docx2pdf] done: converted={converted} skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all DOCX files under an input directory into PDFs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory to scan for .docx files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where PDFs will be written (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "word", "libreoffice", "docx2pdf"),
        help="Conversion backend to use (default: auto)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDFs instead of skipping them",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return convert_all(
        Path(args.input_dir).resolve(),
        Path(args.output_dir).resolve(),
        engine=args.engine,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
