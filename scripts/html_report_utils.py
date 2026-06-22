"""Small self-contained HTML report helper."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


STYLE = """
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f7f8fa; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
h1 { font-size: 30px; margin: 0 0 8px; }
h2 { font-size: 22px; margin-top: 34px; border-bottom: 1px solid #d8dde6; padding-bottom: 8px; }
h3 { font-size: 17px; margin-top: 24px; }
p { line-height: 1.55; }
.section { background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 20px; margin-top: 18px; }
.meta { color: #52606d; font-size: 14px; }
.warning { background: #fff7e6; border: 1px solid #f3c969; border-radius: 6px; padding: 10px 12px; margin: 10px 0; }
.error { background: #ffefef; border: 1px solid #e08a8a; border-radius: 6px; padding: 10px 12px; margin: 10px 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 13px; }
th, td { border: 1px solid #d8dde6; padding: 7px 8px; text-align: left; vertical-align: top; }
th { background: #edf1f5; }
tr:nth-child(even) td { background: #fbfcfd; }
img { max-width: 100%; height: auto; border: 1px solid #d8dde6; border-radius: 6px; background: white; margin: 8px 0 16px; }
code { background: #edf1f5; padding: 1px 4px; border-radius: 4px; }
"""


def _paragraphs(text: str | list[str] | None) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        text = [text]
    return "\n".join(f"<p>{escape(str(item))}</p>" for item in text)


def _render_table(table: Any) -> str:
    if table is None:
        return ""
    if hasattr(table, "to_html"):
        return table.to_html(index=False, border=0, float_format=lambda value: f"{value:.5g}")
    return str(table)


def _render_section(section: dict[str, Any], output_dir: Path) -> str:
    parts = ['<section class="section">']
    heading = section.get("heading")
    if heading:
        parts.append(f"<h2>{escape(str(heading))}</h2>")
    parts.append(_paragraphs(section.get("text")))
    for warning in section.get("warnings", []) or []:
        parts.append(f'<div class="warning"><strong>Warning.</strong> {escape(str(warning))}</div>')
    for error in section.get("errors", []) or []:
        parts.append(f'<div class="error"><strong>Error.</strong> {escape(str(error))}</div>')
    for table in section.get("tables", []) or []:
        if isinstance(table, dict) and "caption" in table:
            parts.append(f"<h3>{escape(str(table['caption']))}</h3>")
            parts.append(_render_table(table.get("data")))
        else:
            parts.append(_render_table(table))
    for image in section.get("images", []) or []:
        caption = None
        path = image
        if isinstance(image, dict):
            path = image.get("path")
            caption = image.get("caption")
        if not path:
            continue
        path = Path(path)
        try:
            src = path.relative_to(output_dir)
        except ValueError:
            src = path
        if caption:
            parts.append(f"<h3>{escape(str(caption))}</h3>")
        parts.append(f'<img src="{escape(str(src))}" alt="{escape(str(caption or path.name))}">')
    parts.append("</section>")
    return "\n".join(parts)


def write_html_report(title: str, sections: list[dict[str, Any]], output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(_render_section(section, output_path.parent) for section in sections)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{STYLE}</style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    {body}
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
