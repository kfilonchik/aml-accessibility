"""Build the AML transit atlas with methodology, data, and GitHub controls.

Place AML_accessibility_methodological_report.pdf in config.DATA/webmap/.
The atlas build publishes the parish CSV beside the GeoJSON layers.
"""

from importlib import import_module
from pathlib import Path
import re
from shutil import copyfile


atlas = import_module("05_webmap_data")
REPORT_NAME = "AML_accessibility_methodological_report.pdf"
CSV_NAME = "AML_accessibility_parishes.csv"

KPI_ACTIONS = f"""
        <a class="metric-btn atlas-action" href="{REPORT_NAME}" target="_blank"
           rel="noopener" aria-label="ABOUT (methodological report, opens in a new tab)">ABOUT</a>
        <button class="metric-btn atlas-action" type="button" id="atlas-data-button"
                aria-haspopup="dialog" aria-controls="atlas-data-dialog">DATA</button>
        <button class="metric-btn atlas-action atlas-github" type="button" disabled>GitHub</button>"""

DATA_DIALOG = f"""
<dialog id="atlas-data-dialog" aria-labelledby="atlas-data-title">
  <h2 id="atlas-data-title">Download parish data</h2>
  <div class="atlas-downloads">
    <a class="metric-btn atlas-action" href="parish.geojson" download>GEOJSON</a>
    <a class="metric-btn atlas-action" href="{CSV_NAME}" download>CSV</a>
  </div>
  <form method="dialog">
    <button class="metric-btn atlas-action" type="submit" autofocus>Close</button>
  </form>
</dialog>
<script>
  document.getElementById("atlas-data-button").addEventListener("click", () => {{
    document.getElementById("atlas-data-dialog").showModal();
  }});
</script>
"""

ACTION_STYLES = """
.atlas-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 32px;
  text-decoration: none;
  font-weight: 600;
  letter-spacing: .04em;
}
.atlas-action:focus-visible { outline: 2px solid var(--accent-ink); outline-offset: 2px; }
.atlas-github { grid-column: 1 / -1; }
.atlas-github:disabled { opacity: .6; cursor: default; border-color: var(--line); color: var(--ink-dim); }
#atlas-data-dialog {
  width: min(320px, calc(100% - 32px));
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--surface);
  color: var(--ink);
  box-shadow: var(--shadow);
}
#atlas-data-dialog::backdrop { background: rgba(20, 30, 40, .5); }
#atlas-data-dialog h2 { font-size: 14.5px; font-weight: 600; }
.atlas-downloads { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }
#atlas-data-dialog form { display: flex; justify-content: flex-end; }
"""


def render_html(template: str) -> str:
    """Place ABOUT/DATA first, GitHub second, and the count cells third."""
    # Reorder the enhanced page only; preserve each tab's attributes and label.
    tab_pattern = r'(?P<indent>[ \t]*)<button\b[^>]*\bdata-tab="(?P<tab>sustain|mode|access|time)"[^>]*>.*?</button>'
    matches = list(re.finditer(tab_pattern, template))
    if len(matches) != 4 or {m.group("tab") for m in matches} != {"sustain", "mode", "access", "time"}:
        raise ValueError("Expected exactly one button for each atlas tab")
    buttons = {m.group("tab"): m.group(0) for m in matches}
    ordered = iter(buttons[tab] for tab in ("sustain", "access", "time", "mode"))
    template = re.sub(tab_pattern, lambda match: next(ordered), template)
    kpi_row = '<div class="kpi-row">'
    population_label = '<span class="l">Population</span></div>'
    for anchor in (kpi_row, population_label, "</style>", "</body>"):
        if template.count(anchor) != 1:
            raise ValueError(f"Expected one atlas template marker: {anchor}")
    return (
        # Separate sibling sections inherit the sidebar's 14px gap. Using
        # .block gives the counts the same border and 12px top padding as
        # the BASE LAYER section below them.
        template.replace(
            kpi_row,
            kpi_row + KPI_ACTIONS + '\n      </div>\n'
            '      <div class="block atlas-kpi-section">\n        ' + kpi_row,
            1,
        )
        .replace(population_label, population_label + '\n        </div>', 1)
        .replace("</style>", ACTION_STYLES + "</style>", 1)
        .replace("</body>", DATA_DIALOG + "</body>", 1)
    )


def write_html() -> Path:
    """Publish the enhanced page and its CSV download."""
    html = render_html(atlas.HTML_TEMPLATE.read_text(encoding="utf-8"))
    copyfile(atlas.STATS_DIR / CSV_NAME, atlas.WEBMAP_DIR / CSV_NAME)
    output = atlas.WEBMAP_DIR / "aml_transit_atlas.html"
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    report = atlas.WEBMAP_DIR / REPORT_NAME
    if not report.is_file():
        raise FileNotFoundError(f"Place the methodological report at {report}")
    atlas.main(publish_html=write_html)


if __name__ == "__main__":
    main()
