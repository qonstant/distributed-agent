#!/usr/bin/env python3
"""
markdown.py

Single-file converter: PDF/DOCX/TXT/MD -> Markdown.
Images are NOT saved; any image encountered is replaced with a simple placeholder:
    [image]

This variant adds an optional `--doc-prefix` (short `-d`) CLI argument that will be
prepended to the `source_file` value written into the markdown frontmatter.

Run:
    python markdown.py input.pdf -o output.md --doc-prefix italy
    python markdown.py input.pdf -o output.md -d italy

Example:
    python markdown.py residence_permit_ru.pdf -o residence_permit_ru.md -d italy
"""
from __future__ import annotations
import sys
import re
import os
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

# ------------------ Utilities ------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _short_hash(data: bytes, n: int = 8) -> str:
    return hashlib.sha1(data).hexdigest()[:n]

def make_readable_name(docname: str, page: Optional[int], idx: int, ext: str = "png") -> str:
    if page is None:
        base = f"{docname}_img{idx}"
    else:
        base = f"{docname}_page{page}_img{idx}"
    return f"{base}.{ext}"

def unique_filename(target_dir: Path, base_name: str, data: bytes) -> str:
    p = target_dir / base_name
    if not p.exists():
        return base_name
    existing = p.read_bytes()
    if existing == data:
        return base_name
    name, ext = os.path.splitext(base_name)
    h = _short_hash(data, 8)
    return f"{name}_{h}{ext}"

# ------------------ Image handling policy (PLACEHOLDERS) ------------------
IMAGE_PLACEHOLDER = "[image]\n\n"

def replace_all_image_markdown_with_placeholder(text: str) -> str:
    return re.sub(r'!\[.*?\]\(.*?\)', IMAGE_PLACEHOLDER, text)

# ------------------ Pandoc handlers ------------------
def try_pandoc_convert(src: str, dst: str, verbose: bool = False) -> bool:
    try:
        import pypandoc
    except Exception:
        if verbose:
            print("[try_pandoc_convert] pypandoc not available")
        return False
    try:
        if verbose:
            print("[try_pandoc_convert] calling pandoc via pypandoc...")
        pypandoc.convert_file(src, "md", outputfile=dst, extra_args=["--wrap=none"])
        if verbose:
            print("[try_pandoc_convert] pandoc conversion done")
        return True
    except Exception as e:
        if verbose:
            print("[try_pandoc_convert] pandoc conversion failed:", e)
        return False

def relocate_pandoc_images_flat(md_path: Path, verbose: bool = False) -> None:
    text = md_path.read_text(encoding="utf-8")
    updated = replace_all_image_markdown_with_placeholder(text)
    if updated != text:
        md_path.write_text(updated, encoding="utf-8")
        if verbose:
            print("[relocate_pandoc_images_flat] replaced image links with placeholders")
    else:
        if verbose:
            print("[relocate_pandoc_images_flat] no image links to replace")

# ------------------ DOCX fallback (mammoth) ------------------
def mammoth_img_cb_factory_placeholder(verbose: bool = False):
    def save_image(image):
        if verbose:
            print("[mammoth_img_cb] image encountered (not saved)")
        return {"src": "image_placeholder"}
    return save_image

def docx_to_md_mammoth_flat(src: Path, dst: Path, verbose: bool = False) -> bool:
    try:
        import mammoth
        from markdownify import markdownify as mdify
    except Exception as e:
        raise RuntimeError("mammoth + markdownify required for DOCX fallback. Install: pip install mammoth markdownify") from e
    ensure_dir(dst.parent)
    cb = mammoth_img_cb_factory_placeholder(verbose=verbose)
    with open(src, "rb") as f:
        result = mammoth.convert_to_html(f, convert_image=cb)
        html = result.value
        warnings = result.messages
    md = mdify(html, heading_style="ATX")
    md = replace_all_image_markdown_with_placeholder(md)
    fm = {"source_file": src.name, "converted_at": datetime.utcnow().isoformat() + "Z", "notes": "mammoth->markdownify (images replaced with placeholder)"}
    content = "---\n" + json.dumps(fm, indent=2) + "\n---\n\n"
    if warnings:
        content += "<!-- mammoth warnings:\n" + "\n".join(map(str, warnings)) + "\n-->\n\n"
    content += md
    dst.write_text(content, encoding="utf-8")
    if verbose:
        print(f"[docx_to_md_mammoth_flat] wrote {dst}")
    return True

# ------------------ PDF helpers ------------------
def build_text_from_chars(chars_line: List[Dict]) -> str:
    if not chars_line:
        return ""
    seq = sorted(chars_line, key=lambda c: float(c.get("x0", 0)))
    widths = [float(c.get("x1",0)) - float(c.get("x0",0)) for c in seq if float(c.get("x1",0)) - float(c.get("x0",0)) > 0]
    median_w = (sorted(widths)[len(widths)//2]) if widths else 3.5
    parts = []
    prev_right = None
    for c in seq:
        ch = c.get("text", "")
        x0 = float(c.get("x0", 0)); x1 = float(c.get("x1", x0 + median_w))
        if prev_right is None:
            parts.append(ch)
        else:
            gap = x0 - prev_right
            if gap > max(0.6 * median_w, 1.2):
                parts.append(" " + ch)
            else:
                parts.append(ch)
        prev_right = x1
    return "".join(parts).replace("\u00A0", " ").strip()

def extract_lines_from_page_chars(page, y_tolerance: float = 3.0, verbose: bool = False) -> List[Dict]:
    chars = getattr(page, "chars", None)
    if not chars:
        words = page.extract_words(extra_attrs=["top","bottom","x0","x1"])
        if not words:
            return []
        words = sorted(words, key=lambda w: (round(w.get("top",0),1), w.get("x0",0)))
        lines = []
        cur_top = None; cur_bottom = None; cur_words = []
        for w in words:
            wtop = float(w.get("top",0)); wbot = float(w.get("bottom", wtop + 1))
            if cur_top is None:
                cur_top = wtop; cur_bottom = wbot; cur_words = [w]
                continue
            if abs(wtop - cur_top) <= y_tolerance:
                cur_words.append(w); cur_bottom = max(cur_bottom, wbot)
            else:
                cur_words_sorted = sorted(cur_words, key=lambda x: x.get("x0",0))
                txt = " ".join([x.get("text","") for x in cur_words_sorted]).strip()
                lines.append({"text": txt, "top": cur_top, "bottom": cur_bottom})
                cur_top = wtop; cur_bottom = wbot; cur_words = [w]
        if cur_words:
            cur_words_sorted = sorted(cur_words, key=lambda x: x.get("x0",0))
            txt = " ".join([x.get("text","") for x in cur_words_sorted]).strip()
            lines.append({"text": txt, "top": cur_top, "bottom": cur_bottom})
        return lines

    chars_sorted = sorted(chars, key=lambda c: (round(float(c.get("top",0)),1), float(c.get("x0",0))))
    lines = []
    cur_top = None; cur_bottom = None; cur_chars = []
    for c in chars_sorted:
        wtop = float(c.get("top", 0)); wbot = float(c.get("bottom", wtop + 1))
        if cur_top is None:
            cur_top = wtop; cur_bottom = wbot; cur_chars = [c]
            continue
        if abs(wtop - cur_top) <= y_tolerance:
            cur_chars.append(c); cur_bottom = max(cur_bottom, wbot)
        else:
            txt = build_text_from_chars(cur_chars)
            lines.append({"text": txt, "top": cur_top, "bottom": cur_bottom})
            cur_top = wtop; cur_bottom = wbot; cur_chars = [c]
    if cur_chars:
        txt = build_text_from_chars(cur_chars)
        lines.append({"text": txt, "top": cur_top, "bottom": cur_bottom})
    if verbose:
        print(f"[extract_lines_from_page_chars] built {len(lines)} lines")
    return lines

def group_lines_to_paragraphs(lines: List[Dict], line_gap_threshold: float = 8.0) -> List[Dict]:
    if not lines:
        return []
    paragraphs = []
    cur_lines = [lines[0]["text"]]
    cur_top = lines[0]["top"]
    cur_bottom = lines[0]["bottom"]
    for ln in lines[1:]:
        gap = ln["top"] - cur_bottom
        if gap <= line_gap_threshold:
            cur_lines.append(ln["text"]); cur_bottom = ln["bottom"]
        else:
            paragraphs.append({"text": "\n".join(cur_lines).strip(), "top": cur_top, "bottom": cur_bottom})
            cur_lines = [ln["text"]]; cur_top = ln["top"]; cur_bottom = ln["bottom"]
    paragraphs.append({"text": "\n".join(cur_lines).strip(), "top": cur_top, "bottom": cur_bottom})
    return paragraphs

# ------------------ PDF image placeholder (no saving) ------------------
def save_pdf_page_images_placeholder(page, page_number: int, verbose: bool = False):
    saved = []
    images = page.images or []
    if verbose:
        print(f"[save_pdf_page_images_placeholder] page {page_number} reported {len(images)} images (will not be saved)")
    for i, img in enumerate(images):
        img_top = img.get("top") or img.get("y0") or 0.0
        assigned_idx = i + 1
        saved.append((None, "[image]", float(img_top), assigned_idx))
    return saved

def pdf_to_md_pdfplumber_flat(src: Path, dst: Path, leading_slash: bool = True, verbose: bool = False, rasterize_pages: bool = False) -> bool:
    try:
        import pdfplumber
    except Exception:
        raise RuntimeError("pdfplumber required. Install: pip install pdfplumber")
    ensure_dir(dst.parent)
    out_lines_all = []
    meta = {"source_file": src.name, "converted_at": datetime.utcnow().isoformat() + "Z", "notes": "pdfplumber fallback (images replaced with placeholders)"}
    with pdfplumber.open(str(src)) as pdf:
        if verbose:
            print(f"[pdf_to_md_pdfplumber_flat] opened PDF, pages: {len(pdf.pages)}")
        for pageno, page in enumerate(pdf.pages, start=1):
            lines = extract_lines_from_page_chars(page, verbose=verbose)
            paragraphs = group_lines_to_paragraphs(lines)
            if verbose:
                print(f"[pdf_to_md_pdfplumber_flat] page {pageno} paragraphs: {len(paragraphs)}")
            saved_imgs = save_pdf_page_images_placeholder(page, pageno, verbose=verbose)
            imgs_by_para = {}
            if paragraphs:
                for (local_path, s3_key, img_top, assigned_idx) in saved_imgs:
                    assigned_idx_para = None
                    for idx, para in enumerate(paragraphs):
                        if para["top"] - 1 <= img_top <= para["bottom"] + 1:
                            assigned_idx_para = idx; break
                    if assigned_idx_para is None:
                        centers = [ (p["top"] + p["bottom"])/2 for p in paragraphs ]
                        best_idx = min(range(len(centers)), key=lambda i: abs(centers[i] - img_top))
                        assigned_idx_para = best_idx
                    imgs_by_para.setdefault(assigned_idx_para, []).append((local_path, s3_key))
            else:
                imgs_by_para.setdefault(-1, []).extend([(lp, sk) for (lp, sk, _, _) in saved_imgs])
            out_lines_all.append(f"\n\n<!-- PAGE {pageno} -->\n\n")
            if paragraphs:
                for idx, para in enumerate(paragraphs):
                    out_lines_all.append(para["text"] + "\n\n")
                    if idx in imgs_by_para:
                        for (local_path, s3_key) in imgs_by_para[idx]:
                            out_lines_all.append(IMAGE_PLACEHOLDER)
            else:
                txt = page.extract_text() or ""
                if txt.strip():
                    out_lines_all.append(txt + "\n\n")
                if -1 in imgs_by_para:
                    for (local_path, s3_key) in imgs_by_para[-1]:
                        out_lines_all.append(IMAGE_PLACEHOLDER)
    dst.write_text("---\n" + json.dumps(meta, indent=2) + "\n---\n\n" + "\n".join(out_lines_all), encoding="utf-8")
    if verbose:
        print(f"[pdf_to_md_pdfplumber_flat] wrote {dst}")
    return True

def txt_or_md_copy(src: Path, dst: Path, verbose: bool = False) -> bool:
    shutil.copyfile(src, dst)
    if verbose:
        print(f"[txt_or_md_copy] copied {src} -> {dst}")
    return True

# ------------------ Frontmatter helpers ------------------
def extract_frontmatter(md_text: str) -> Tuple[Dict[str, Any], str]:
    if md_text.lstrip().startswith("---"):
        parts = md_text.split("---", 2)
        if len(parts) == 3:
            fm_raw = parts[1].strip()
            remainder = parts[2]
            try:
                fm = json.loads(fm_raw)
            except Exception:
                try:
                    import yaml
                    fm = yaml.safe_load(fm_raw) or {}
                except Exception:
                    fm = {}
            return fm, remainder
    return {}, md_text

def write_frontmatter(md_path: Path, fm: Dict[str, Any], body: str) -> None:
    content = "---\n" + json.dumps(fm, ensure_ascii=False, indent=2) + "\n---\n\n" + body.lstrip()
    md_path.write_text(content, encoding="utf-8")

def update_frontmatter_source(md_path: Path, original_src_name: str, doc_prefix: Optional[str], verbose: bool = False) -> None:
    """
    Update (or create) JSON frontmatter 'source_file' with optional doc_prefix.
    If frontmatter exists, it will be modified; if not, a new frontmatter is added.
    original_src_name should be a filename like 'residence_permit.pdf' (used when frontmatter missing).
    """
    text = md_path.read_text(encoding="utf-8")
    fm, body = extract_frontmatter(text)
    if not fm:
        fm = {"source_file": original_src_name, "converted_at": datetime.utcnow().isoformat() + "Z", "notes": "converted (images replaced with placeholder)"}
    # ensure source_file present
    src_name = fm.get("source_file") or original_src_name
    if doc_prefix:
        doc_prefix_clean = str(doc_prefix).rstrip('/')
        if not str(src_name).startswith(f"{doc_prefix_clean}/"):
            src_name = f"{doc_prefix_clean}/{src_name}"
    fm["source_file"] = src_name
    # write back
    write_frontmatter(md_path, fm, body)
    if verbose:
        print(f"[update_frontmatter_source] source_file set to: {fm['source_file']} in {md_path}")

# ------------------ Orchestrator: convert input files to markdown only ------------------
def convert_to_markdown(src: str, dst: Optional[str] = None, leading_slash: bool = True, verbose: bool = False, rasterize_pages: bool = False) -> Path:
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(src)
    if dst is None:
        dst = str(src_path.with_suffix(".md"))
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    ext = src_path.suffix.lower()
    if try_pandoc_convert(str(src_path), str(dst_path), verbose=verbose):
        if verbose:
            print("[convert_to_markdown] pandoc used")
        relocate_pandoc_images_flat(dst_path, verbose=verbose)
        final = dst_path
    else:
        if ext == ".docx":
            if verbose:
                print("[convert_to_markdown] mammoth fallback for docx")
            ok = docx_to_md_mammoth_flat(src_path, dst_path, verbose=verbose)
            if not ok:
                raise RuntimeError("DOCX fallback failed")
            final = dst_path
        elif ext == ".pdf":
            if verbose:
                print("[convert_to_markdown] pdfplumber fallback for pdf")
            ok = pdf_to_md_pdfplumber_flat(src_path, dst_path, leading_slash=leading_slash, verbose=verbose, rasterize_pages=rasterize_pages)
            if not ok:
                raise RuntimeError("PDF fallback failed")
            final = dst_path
        elif ext in (".md", ".markdown", ".txt"):
            txt_or_md_copy(src_path, dst_path, verbose=verbose)
            final = dst_path
        else:
            raise RuntimeError(f"unsupported extension: {ext}. Install pandoc for best conversion.")
    if verbose:
        print(f"[convert_to_markdown] final markdown: {final}")
    return final

# ------------------ CLI ------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Convert PDF/DOCX/TXT -> Markdown; images replaced with placeholder.")
    p.add_argument("input", help="input file (pdf, docx, md, txt)")
    p.add_argument("-o", "--output", required=True, help="output markdown file path")
    p.add_argument("-d", "--doc-prefix", dest="doc_prefix", default=None, help="optional path/prefix to prepend to source_file in frontmatter (e.g. 'italy')")
    p.add_argument("--no-leading-slash", action="store_true", help="ignored in this mode (kept for compatibility)")
    p.add_argument("--verbose", action="store_true", help="print debug logs")
    p.add_argument("--rasterize-pages", action="store_true", help="rasterize pages when no embedded images (not saving images, but may improve text extraction)")
    args = p.parse_args()
    lead = not args.no_leading_slash
    try:
        out_md = convert_to_markdown(args.input, args.output, leading_slash=lead, verbose=args.verbose, rasterize_pages=args.rasterize_pages)
        print("Converted to:", out_md)
        # update frontmatter source_file with doc-prefix (if provided)
        try:
            original_src_name = Path(args.input).name
            update_frontmatter_source(Path(out_md), original_src_name, args.doc_prefix, verbose=args.verbose)
            if args.verbose:
                print("[main] frontmatter updated (if needed).")
        except Exception as e:
            print("[post-convert] frontmatter update failed:", e)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
