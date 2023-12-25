import json
from dataclasses import asdict
from pathlib import Path

import rich
import typer
from rich import progress

from tagclass.common import VOC_FILES
from tagclass.parser import Parser
from tagclass.tag import Vocabulary
from tagclass.tokenizer import Tokenizer
from tagclass.utils import load_label_engines, load_sample_label_engines

app = typer.Typer()


def sample_level_parser(
    vtapiv2: str,
    dataset: str = "unnamed",
    output_dir: str = None,
    sifter_json: str = None,
):
    rich.print(f"[+] Dataset: {dataset}")
    sample_label_engines = load_sample_label_engines(vtapiv2)
    total = len(sample_label_engines)
    rich.print(f"[+] Loaded {total} samples")
    results = {}
    parser = Parser(Tokenizer(), sifter_json=sifter_json, ignore_generic=True)
    voc = Vocabulary(VOC_FILES)
    for md5, label_engines in progress.track(
        sample_label_engines.items(),
        total=total,
        description="Parsing ...",
    ):
        engine_family = {}
        for label, engines in label_engines.items():
            parsed = parser.parse(label, engines[0], voc)
            for r in parsed:
                if r.entity == "family":
                    engine_family[engines[0]] = r.tag
        results[md5] = {"md5": md5, "engine_family": engine_family}

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    save_path = output_dir / f"{dataset}-tagclass-sample_parse.jsonl"
    with open(save_path, "w") as f:
        for _, res in progress.track(
            results.items(),
            total=len(results),
            description="Saving...",
        ):
            f.write(json.dumps(res) + "\n")


def label_level_parser(
    vtapiv2: str,
    dataset: str = "unnamed",
    output_dir: str = None,
):
    rich.print(f"[+] Dataset: {dataset}")
    label_engines = load_label_engines(vtapiv2)
    total = len(label_engines)
    rich.print(f"[+] Loaded {total} labels")
    results = {}
    parser = Parser(Tokenizer())
    voc = Vocabulary(VOC_FILES)
    for label, engines in progress.track(
        label_engines.items(),
        total=len(label_engines),
        description="Parsing ...",
    ):
        parsed = parser.parse(label, engines[0], voc)
        results[label] = {
            "label": label,
            "ner": [asdict(p) for p in parsed],
        }

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    save_path = output_dir / f"{dataset}-tagclass-label_parse.jsonl"
    with open(save_path, "w") as f:
        for _, res in progress.track(
            results.items(),
            total=len(results),
            description="Saving...",
        ):
            f.write(json.dumps(res) + "\n")


@app.callback(invoke_without_command=True)
def main(
    label: str = None,
    engine: str = None,
    vtapiv2: str = None,
    level: str = "label",
    dataset: str = "unnamed",
    output_dir: str = None,
    sifter_json: str = None,
):
    if label is None and vtapiv2 is None:
        rich.print("[!] Need a label or a vtapiv2 file")
        raise typer.Exit(-1)

    # label
    if label:
        parser = Parser(Tokenizer())
        voc = Vocabulary(VOC_FILES)
        engine = engine or "default"
        result = parser.parse(label, engine, voc)
        rich.print(result)
        raise typer.Exit()

    # vtapiv2
    match level:
        case "label":
            label_level_parser(vtapiv2, dataset, output_dir)
        case "sample":
            sample_level_parser(vtapiv2, dataset, output_dir, sifter_json)
        case _:
            rich.print(f"[x] Unknown level {level}")
            raise typer.Exit(-1)
