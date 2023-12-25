import json
from pathlib import Path

import rich
import typer
from rich.table import Table

from instclass.common import DATA_ROOT, TagAlias

app = typer.Typer()


ALIAS_FILE = DATA_ROOT / "alias/default_tagalias.txt"


def accuracy(true: list[str], pred: list[str]) -> float:
    assert len(true) == len(pred)
    return sum([1 if t == p else 0 for t, p in zip(true, pred)]) / len(true)


def load_json(infer: Path, tagalias: TagAlias) -> dict[str, str]:
    with open(infer, "r") as fd:
        return {k: tagalias(v) for k, v in json.load(fd).items()}


def load_tagged_ranked(jsonl: Path) -> dict[str, str]:
    results = {}
    with open(jsonl, "r") as fd:
        for line in fd:
            data = json.loads(line)
            md5 = data["md5"]
            ranked_family = data["ranked_family"]
            ranked_family = " ".join([f"{f}-{r}" for f, r in ranked_family[:3]])
            results[md5] = ranked_family
    return results


def load_avclass_ranked(json_file: Path) -> dict[str, str]:
    with open(json_file, "r") as fd:
        data = json.load(fd)

    return {k: " ".join([f"{f}-{c}" for f, c in rank]) for k, rank in data.items()}


def print_error_table(
    data: list[tuple],
    *,
    name="error",
):
    error_table = Table(title=name)

    column_style = {
        "md5": "cyan",
        "true": "green",
        "pred": "red",
        "tagclass": "cyan",
        "avclass": "cyan",
    }
    for c, s in column_style.items():
        error_table.add_column(c, style=s)

    columns = [k for k, _ in column_style.items()]
    results = []
    for item in data:
        error_table.add_row(*item)
        results.append(dict(zip(columns, item)))

    rich.print(error_table)

    save_path = Path.cwd() / f"logs/{name}.jsonl"
    with open(save_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


@app.callback(invoke_without_command=True)
def main(
    dataset: str = None,
    tagged: str = None,
    # use_selected: bool = False,
    alias_file: str = ALIAS_FILE,
    table: str = None,
):
    dataset_dir = DATA_ROOT / dataset
    if not dataset_dir.exists():
        rich.print(f"Dataset {dataset} does not exist!!!")
        raise typer.Exit()

    if tagged not in ["strongly", "weakly"]:
        rich.print(f"Tagged {tagged} is not supported!!!")
        raise typer.Exit()

    tagged_ranked = load_tagged_ranked(
        dataset_dir / f"{dataset}-tagclass-{tagged}_tagged.jsonl"
    )
    avclass_ranked = load_avclass_ranked(dataset_dir / f"{dataset}-avclass-parsed.json")

    # if use_selected:
    instclass_infer = dataset_dir / f"{dataset}-tagclass-selected-DS-inferred.json"
    # else:
    #     instclass_infer = (
    #         dataset_dir / f"{dataset}-tagclass-{tagged}_tagged-DS-inferred.json"
    #     )
    euphony_infer = dataset_dir / f"{dataset}-euphony-inferred.json"
    avclass_infer = dataset_dir / f"{dataset}-avclass-inferred.json"
    sample_family = dataset_dir / f"{dataset}-sample-family.json"

    for i in [instclass_infer, euphony_infer, avclass_infer, sample_family]:
        if not i.exists():
            rich.print(f"File {i} does not exist!!!")
            raise typer.Exit()

    tagalias = TagAlias(alias_file)
    ground = load_json(sample_family, tagalias)
    tagclass = load_json(instclass_infer, tagalias)
    euphony = load_json(euphony_infer, tagalias)
    avclass = load_json(avclass_infer, tagalias)

    md5_list = [k for k in tagged_ranked]
    true = [ground[k] for k in tagged_ranked]
    instclass_pred = [tagclass[k] for k in tagged_ranked]
    euphony_pred = [euphony[k] for k in tagged_ranked]
    avclass_pred = [avclass[k] for k in tagged_ranked]

    instclass_result = [i == t for i, t in zip(instclass_pred, true)]
    euphony_result = [i == t for i, t in zip(euphony_pred, true)]
    avclass_result = [i == t for i, t in zip(avclass_pred, true)]

    match table:
        case "A1I0":  # print instclass failed but avclass success
            data = [
                (md5, t, p, tagged_ranked[md5], avclass_ranked[md5])
                for md5, t, p, i, a in zip(
                    md5_list, true, instclass_pred, instclass_result, avclass_result
                )
                if i == 0 and a == 1
            ]
            print_error_table(
                data, name=f"Instclass_Failed_Avclass_Success_on_{dataset}_{tagged}"
            )
        case "E1I0":  # print instclass failed but euphony success
            data = [
                (md5, t, p, tagged_ranked[md5], avclass_ranked[md5])
                for md5, t, p, i, e in zip(
                    md5_list, true, instclass_pred, instclass_result, euphony_result
                )
                if i == 0 and e == 1
            ]
            print_error_table(
                data, name=f"Instclass_Failed_Euphony_Success_on_{dataset}_{tagged}"
            )
        case "A0I1":  # print instclass success but avclass failed
            data = [
                (md5, t, p, tagged_ranked[md5], avclass_ranked[md5])
                for md5, t, p, i, a in zip(
                    md5_list, true, instclass_pred, instclass_result, avclass_result
                )
                if i == 1 and a == 0
            ]
            print_error_table(
                data, name=f"Instclass_Success_Avclass_Failed_on_{dataset}_{tagged}"
            )
        case "E0":
            data = [
                (md5, t, p, tagged_ranked[md5], avclass_ranked[md5])
                for md5, t, p, e in zip(md5_list, true, euphony_pred, euphony_result)
                if e == 0
            ]
            print_error_table(data, name=f"Euphony_Failed_on_{dataset}_{tagged}")
        case "A0":
            data = [
                (md5, t, p, tagged_ranked[md5], avclass_ranked[md5])
                for md5, t, p, a in zip(md5_list, true, avclass_pred, avclass_result)
                if a == 0
            ]
            print_error_table(data, name=f"Avclass_Failed_on_{dataset}_{tagged}")
        case "I0":  # print instclass failed
            data = [
                (md5, t, p, tagged_ranked[md5], avclass_ranked[md5])
                for md5, t, p, i in zip(
                    md5_list, true, instclass_pred, instclass_result
                )
                if i == 0
            ]
            print_error_table(data, name=f"Instclass_Failed_on_{dataset}_{tagged}")
    # y_dist = dict(sorted(Counter(true).items(), key=lambda x: x[1], reverse=True))
    num = len(true)
    rich.print()
    rich.print(
        f"""[*] ====================================================
    Inferring Accuracy on {dataset} {tagged}-tagged 
    Total = {num}
    Euphony Acc = {accuracy(true, euphony_pred)}
    AVClass2 Acc = {accuracy(true, avclass_pred)}
    InstClass Acc = {accuracy(true, instclass_pred)}
    ===================================================="""
    )
