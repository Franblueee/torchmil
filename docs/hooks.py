from __future__ import annotations

from pathlib import Path
import re

PLACEHOLDERS = {
    "api/index.md": "<!-- API_AUTOLIST -->",
    "api/data/index.md": "<!-- DATA_AUTOLIST -->",
    "api/datasets/index.md": "<!-- DATASETS_AUTOLIST -->",
    "api/models/index.md": "<!-- MODELS_AUTOLIST -->",
    "api/nn/index.md": "<!-- NN_AUTOLIST -->",
    "api/utils/index.md": "<!-- UTILS_AUTOLIST -->",
    "api/visualize/index.md": "<!-- VISUALIZE_AUTOLIST -->",
}


def _extract_title(markdown_path: Path) -> str:
    content = markdown_path.read_text(encoding="utf-8")
    match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()

    return markdown_path.stem.replace("_", " ").title()


def _get_markdown_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*.md")
        if path.name not in {"index.md", ".pages"} and not path.name.startswith("_")
    )


def _build_local_list(directory: Path) -> str:
    files = _get_markdown_files(directory)
    lines = [f"- [{_extract_title(path)}]({path.name})" for path in files]
    return "\n".join(lines)


def _build_nn_list(nn_dir: Path, path_prefix: str = "") -> str:
    lines: list[str] = []

    nn_groups = [
        ("attention", "Attention"),
        ("gnns", "Graph Neural Networks (GNNs)"),
        ("transformers", "Transformers"),
    ]

    for subdir_name, label in nn_groups:
        subdir = nn_dir / subdir_name
        lines.append(f"- [{label}]({path_prefix}{subdir_name}/index.md)")
        for path in _get_markdown_files(subdir):
            lines.append(
                f"    - [{_extract_title(path)}]({path_prefix}{subdir_name}/{path.name})"
            )

    for path in _get_markdown_files(nn_dir):
        lines.append(f"- [{_extract_title(path)}]({path_prefix}{path.name})")

    return "\n".join(lines)


def _build_api_root_list(api_dir: Path) -> str:
    sections = [
        ("data", "Data", "torchmil.data"),
        ("datasets", "Datasets", "torchmil.datasets"),
        ("nn", "Modules", "torchmil.nn"),
        ("models", "Models", "torchmil.models"),
        ("visualize", "Visualize", "torchmil.visualize"),
        ("utils", "Utils", "torchmil.utils"),
    ]

    chunks: list[str] = []
    for folder, heading, module_name in sections:
        section_dir = api_dir / folder
        chunks.append(f"## {heading}: [{module_name}]({folder}/index.md)")
        chunks.append(f"- [Introduction]({folder}/index.md)")

        if folder == "nn":
            nn_lines = _build_nn_list(section_dir, path_prefix="nn/")
            chunks.extend(nn_lines.splitlines())
        else:
            for path in _get_markdown_files(section_dir):
                chunks.append(f"- [{_extract_title(path)}]({folder}/{path.name})")

        chunks.append("")

    return "\n".join(chunks).rstrip()


def _build_replacement(page_src_path: str, docs_dir: Path) -> str:
    if page_src_path == "api/index.md":
        return _build_api_root_list(docs_dir / "api")
    if page_src_path == "api/nn/index.md":
        return _build_nn_list(docs_dir / "api" / "nn")

    section_name = page_src_path.split("/")[1]
    return _build_local_list(docs_dir / "api" / section_name)


def on_page_markdown(markdown, page, config, files):
    page_src_path = page.file.src_path
    placeholder = PLACEHOLDERS.get(page_src_path)

    if placeholder is None:
        return markdown

    if placeholder not in markdown:
        return markdown

    docs_dir = Path(config["docs_dir"])
    replacement = _build_replacement(page_src_path, docs_dir)
    return markdown.replace(placeholder, replacement)
