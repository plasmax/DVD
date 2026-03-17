#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


SECTION_TITLES = {
    "abstract",
    "introduction",
    "related work",
    "preliminary",
    "methodology",
    "overall framework",
    "experiments",
    "limitations",
    "conclusion",
    "appendix",
    "references",
}


def normalize_text(text: str) -> str:
    replacements = {
        "\r": "",
        "\u000c": "\f",
        "\u0080": "",
        "z Date:": "Date:",
        "§ Github:": "Github:",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def run_pdftotext(pdf_path: Path) -> list[str]:
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return normalize_text(result.stdout).split("\f")


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.fullmatch(r"\d+", stripped):
        return True
    if re.fullmatch(r"\d+\.\d+", stripped):
        return True
    if stripped.startswith("arXiv:"):
        return True
    if stripped in {"†", "‡"}:
        return True
    return False


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.search(r"[=<>↑↓∼]", stripped):
        return False
    if re.fullmatch(r"[1-9]\d*(\.\d+)*\s+[A-Z][A-Za-z0-9 ,:&()/\-]{2,}", stripped):
        return True
    return stripped.lower() in SECTION_TITLES


def looks_like_short_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped.lower() in SECTION_TITLES:
        return True
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9 ,:&()/\-]+", stripped))


def should_join(prev_line: str, next_line: str) -> bool:
    prev = prev_line.strip()
    nxt = next_line.strip()
    if not prev or not nxt:
        return False
    if looks_like_heading(prev) or looks_like_heading(nxt):
        return False
    if nxt.startswith(("Figure ", "Table ", "Date:", "Project:", "Github:")):
        return False
    if prev.startswith(("Figure ", "Table ")):
        return True
    if prev.endswith((".", ":", "?", "!", '"')) and nxt[:1].isupper():
        return False
    if prev.endswith("-"):
        return True
    return True


def merge_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if re.fullmatch(r"\d+(\.\d+)*", line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and looks_like_short_title(lines[j]):
                merged.append(f"{line} {lines[j].strip()}")
                i = j + 1
                continue

        current = line
        j = i + 1
        while j < len(lines):
            candidate = lines[j].strip()
            if not candidate:
                break
            if not should_join(current, candidate):
                break
            if current.endswith("-"):
                current = current[:-1] + candidate
            else:
                current = f"{current} {candidate}"
            j += 1

        merged.append(current)
        i = j + 1 if j > i and j < len(lines) and not lines[j].strip() else j

    return merged


def page_to_blocks(page_text: str) -> list[str]:
    raw_lines = [line.rstrip() for line in page_text.splitlines()]
    cleaned = [line for line in raw_lines if not is_noise_line(line)]
    return merge_lines(cleaned)


def is_caption_label_noise(block: str) -> bool:
    stripped = block.strip()
    if not stripped or len(stripped) > 40:
        return False
    if stripped.startswith(("Figure ", "Table ", "Date:", "Project:", "Github:")):
        return False
    if looks_like_heading(stripped):
        return False
    if re.fullmatch(r"#\d+", stripped):
        return True
    if re.fullmatch(r"Frame\s+#\d+", stripped):
        return True
    if re.fullmatch(r"[A-Z]{2,}", stripped):
        return True
    if re.fullmatch(r"[A-Za-z]+(?:\s+\([A-Za-z]+\))?", stripped):
        return True
    if any(ch in stripped for ch in ".:;!?/"):
        return False
    tokens = stripped.split()
    if not tokens:
        return False
    if all(token.startswith("#") or token.isupper() for token in tokens):
        return True
    return sum(ch.islower() for ch in stripped) <= 2


def is_global_label_noise(block: str) -> bool:
    stripped = block.strip()
    if not stripped or len(stripped) > 60:
        return False
    if stripped.startswith(("Figure ", "Table ", "Date:", "Project:", "Github:")):
        return False
    if looks_like_heading(stripped):
        return False
    if re.fullmatch(r"#\d+", stripped):
        return True
    if re.fullmatch(r"Frame\s+#\d+", stripped):
        return True
    if stripped in {
        "RGB",
        "GT",
        "VDA",
        "DVD (Ours)",
        "DepthCrafter",
        "Latent Manifold Rectification",
    }:
        return True
    if re.fullmatch(r"w/[A-Za-z. ]+|w/o[A-Za-z. ]+", stripped):
        return True
    if stripped.count("τ") + stripped.count("𝜏") >= 2:
        return True
    return False


def looks_like_inline_subhead(block: str) -> bool:
    stripped = block.strip()
    if len(stripped) < 6 or len(stripped) > 80:
        return False
    if not stripped.endswith("."):
        return False
    if stripped.startswith(("Figure ", "Table ")):
        return False
    if looks_like_heading(stripped[:-1]):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9\- ]+\.", stripped))


def looks_like_plain_subhead(block: str) -> bool:
    stripped = block.strip()
    if len(stripped) < 6 or len(stripped) > 80:
        return False
    if stripped.endswith("."):
        return False
    if stripped.startswith(("Figure ", "Table ")):
        return False
    if looks_like_heading(stripped):
        return False
    if re.search(r"[=<>↑↓∼]", stripped):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9\- ]+", stripped))


def looks_like_table_row(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if looks_like_heading(stripped) or stripped.startswith(("Figure ", "Table ")):
        return False
    if "↑" in stripped or "↓" in stripped:
        return True
    if len(re.findall(r"\d", stripped)) >= 4:
        return True
    if re.search(r"\b(?:AbsRel|Recall|Paradigm|Method|KITTI|ScanNet|Bonn|Sintel|NYUv2|DIODE)\b", stripped):
        return True
    return False


def looks_like_prose(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if looks_like_heading(stripped) or looks_like_inline_subhead(stripped) or looks_like_plain_subhead(stripped):
        return False
    if stripped.startswith(("Figure ", "Table ")):
        return False
    if len(re.findall(r"[A-Za-z]", stripped)) < 20:
        return False
    if len(re.findall(r"\d", stripped)) > 8 and "Figure" not in stripped:
        return False
    return "." in stripped or ":" in stripped


def cleanup_blocks(blocks: list[str], title: str) -> list[str]:
    cleaned: list[str] = []
    for block in blocks:
        if not block:
            continue
        if not cleaned and block.startswith(title):
            continue
        if is_global_label_noise(block):
            continue
        if block.startswith(("Figure ", "Table ")):
            while cleaned and is_caption_label_noise(cleaned[-1]):
                cleaned.pop()
        cleaned.append(block)

    while cleaned and (
        "Equal Contribution" in cleaned[0]
        or "HKUST" in cleaned[0]
        or "University" in cleaned[0]
    ):
        cleaned.pop(0)

    first_heading_index = next((i for i, block in enumerate(cleaned) if looks_like_heading(block)), len(cleaned))
    for index in range(first_heading_index):
        if len(cleaned[index]) > 200 and not looks_like_heading(cleaned[index]):
            cleaned.insert(index, "Abstract")
            break

    return cleaned


def blocks_to_markdown(blocks: list[str]) -> str:
    output: list[str] = []
    last_heading: str | None = None
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if looks_like_heading(block):
            heading = f"## {block}"
            if heading != last_heading:
                output.append(heading)
                output.append("")
                last_heading = heading
            i += 1
            continue
        if looks_like_inline_subhead(block):
            heading = f"### {block[:-1]}"
            if heading != last_heading:
                output.append(heading)
                output.append("")
                last_heading = heading
            i += 1
            continue
        if looks_like_plain_subhead(block):
            heading = f"### {block}"
            if heading != last_heading:
                output.append(heading)
                output.append("")
                last_heading = heading
            i += 1
            continue
        if block.startswith("Figure "):
            caption = " ".join(
                part.strip()
                for part in [block, blocks[i + 1] if i + 1 < len(blocks) and not blocks[i + 1].startswith(("Figure ", "Table ")) and not looks_like_heading(blocks[i + 1]) and len(blocks[i + 1]) < 220 else ""]
                if part.strip()
            )
            output.append(f"> {caption}")
            output.append("")
            last_heading = None
            if caption != block:
                i += 1
            i += 1
            continue
        if block.startswith("Table "):
            output.append(f"> {block}")
            output.append("")
            j = i + 1
            while j < len(blocks):
                if looks_like_heading(blocks[j]) or looks_like_inline_subhead(blocks[j]) or looks_like_plain_subhead(blocks[j]):
                    break
                if blocks[j].startswith(("Figure ", "Table ")):
                    break
                if looks_like_prose(blocks[j]):
                    break
                j += 1
            last_heading = None
            i = j
            continue
        output.append(block)
        output.append("")
        last_heading = None
        i += 1
    return "\n".join(output).strip()


def extract_title(first_page: str, fallback: str) -> str:
    lines = [line.strip() for line in first_page.splitlines() if line.strip()]
    if len(lines) >= 2 and len(lines[0]) < 120 and len(lines[1]) < 120:
        return f"{lines[0]} {lines[1]}"
    if lines:
        return lines[0]
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a PDF into LLM-friendly Markdown.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("markdown_path", type=Path)
    args = parser.parse_args()

    pages = run_pdftotext(args.pdf_path)
    title = extract_title(pages[0], args.pdf_path.stem)

    body_blocks: list[str] = []
    for page in pages:
        body_blocks.extend(page_to_blocks(page))
    body_blocks = cleanup_blocks(body_blocks, title)

    markdown = f"# {title}\n\n{blocks_to_markdown(body_blocks)}\n"
    args.markdown_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
