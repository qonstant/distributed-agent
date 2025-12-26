from __future__ import annotations
import argparse
import nbformat
from pathlib import Path
import base64
import os
import sys

def convert(nb_path: Path, out_path: Path | None = None, stdout: bool = False):
    nb = nbformat.read(str(nb_path), as_version=4)
    lines = []
    for cell in nb.cells:
        ctype = cell.get("cell_type", "")
        src = cell.get("source", "")
        if ctype == "markdown":
            # markdown cell -> keep as-is
            lines.append(src.rstrip())
            lines.append("")  # blank line separator
        elif ctype == "code":
            # code cell -> fenced code block
            lines.append("```python")
            # ensure code ends with newline
            if isinstance(src, list):
                src = "".join(src)
            lines.append(src.rstrip())
            lines.append("```")
            lines.append("")  # blank line separator

            # optionally, include text describing outputs (simple)
            outputs = cell.get("outputs", []) or []
            for out in outputs:
                otype = out.get("output_type", "")
                if otype == "stream":
                    text = out.get("text", "")
                    if text:
                        lines.append("```\n" + str(text).rstrip() + "\n```")
                        lines.append("")
                elif otype in ("display_data", "execute_result"):
                    # if there is an image/png in data, save it to an assets dir
                    data = out.get("data", {})
                    if "image/png" in data:
                        # create assets dir next to notebook
                        assets_dir = nb_path.with_suffix("").name + "_assets"
                        os.makedirs(assets_dir, exist_ok=True)
                        img_b64 = data["image/png"]
                        # name images with incremental index
                        idx = len([p for p in Path(assets_dir).glob("*.png")])
                        img_path = Path(assets_dir) / f"output_{idx+1}.png"
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(img_b64))
                        lines.append(f"![output image]({img_path.as_posix()})")
                        lines.append("")
                    else:
                        text = data.get("text/plain")
                        if text:
                            lines.append("```\n" + str(text).rstrip() + "\n```")
                            lines.append("")
                elif otype == "error":
                    tb = out.get("traceback", [])
                    if tb:
                        lines.append("```\n" + "\n".join(tb).rstrip() + "\n```")
                        lines.append("")

        else:
            # other cell types (raw etc.) -> just dump source
            lines.append(str(src).rstrip())
            lines.append("")

    md = "\n".join(lines).rstrip() + "\n"
    if stdout:
        sys.stdout.write(md)
    else:
        if out_path is None:
            out_path = nb_path.with_suffix(".md")
        out_path.write_text(md, encoding="utf-8")
        print("Wrote", out_path)

def main():
    p = argparse.ArgumentParser(description="Simple .ipynb -> .md converter (no nbconvert needed)")
    p.add_argument("notebook", help="input .ipynb file")
    p.add_argument("--out", "-o", help="output .md file (defaults to same basename)")
    p.add_argument("--stdout", action="store_true", help="print markdown to stdout instead of writing file")
    args = p.parse_args()
    nb_path = Path(args.notebook)
    if not nb_path.exists():
        print("Notebook not found:", nb_path, file=sys.stderr)
        raise SystemExit(2)
    out_path = Path(args.out) if args.out else None
    convert(nb_path, out_path=out_path, stdout=args.stdout)

if __name__ == "__main__":
    main()


# For translation from .ipynb to markdown, example:
# python text.py rag.ipynb