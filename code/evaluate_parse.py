import json
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pyrootutils as pru
import rich
import typer
from rich.table import Table

from tagclass.common import (
    ALIAS_FILE,
    INIT_LOCATOR_VOC_FILE,
    INIT_MISC_VOC_FILE,
    LOCATOR_VOC_FILE,
    MISC_VOC_FILE,
)
from tagclass.parser import LOCATORS, Parser, RunMode
from tagclass.tag import TagEntity, TagScore, Vocabulary
from tagclass.tokenizer import Tokenizer
from tagclass.update import locator_incremental_update
from tagclass.utils import TagAlias, load_label_engines, load_sample_label_engines

root = pru.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
)

app = typer.Typer()


SIFTER_FILE = root / "data/tagsifter/tagsifter-malgenome_drebin_motif_bodmas.json"


# namedtuple
ParseResult = namedtuple("ParseResult", ["label", "engine", "family", "score"])


def precision_recall(
    updated: Set[str],
    truth: Set[str],
    exist: Set[str],
    possible: Set[str],
):
    # precision focus on updated
    tp = truth & updated
    if len(updated) == 0:
        return {}
    metrics = {}
    metrics["precision"] = len(tp) / len(updated)
    # recall focus on all possible
    tpos = exist & possible
    metrics["recall"] = len(tpos) / len(possible)
    return metrics


def load_cfs_data(
    nermnl_jsonl: str,
    *,
    threshold_cfs: int,
) -> Tuple[Dict[str, str], Set[str], Dict[str, List[str]]]:
    tag_entity = {}
    possible_locator_set = defaultdict(int)
    remark = defaultdict(set)
    with open(nermnl_jsonl, "r") as f:
        for line in f:
            label, data = json.loads(line)
            for t, e in data.items():
                tag_entity[t] = e
                remark[t].add(label)
            family = [t for t, e in data.items() if e == TagEntity.FAMILY]
            if len(family) > 0:
                for t, e in data.items():
                    if e in LOCATORS:
                        possible_locator_set[t] += 1
    # filter
    possible_locator_set = {
        t for t, c in possible_locator_set.items() if c >= threshold_cfs
    }
    # sort
    remark = {k: sorted(v) for k, v in remark.items()}
    return tag_entity, possible_locator_set, remark


@app.command(short_help="test incremental parsing for locator update")
def update(
    malgenome: bool = False,
    drebin: bool = False,
    threshold_cfs: int = 2,
    max_round: int = 3,
    dump: bool = False,
    lfs_mode: str = RunMode.UPDATE,
    table: int = 1,
    remain_inspect: int = 3,
):
    if malgenome:
        dataset = root / "data/malgenome"
        vtapiv2_file = dataset / "malgenome-vtapiv2.jsonl"
        ground_truth_file = dataset / "malgenome-nermnl-label_parse.jsonl"
    elif drebin:
        dataset = root / "data/drebin"
        vtapiv2_file = dataset / "drebin-vtapiv2.jsonl"
        ground_truth_file = dataset / "drebin-nermnl-label_parse.jsonl"
    else:
        rich.print("[x] Assign a dataset first!")
        raise typer.Exit(-1)

    # load
    tag_entity, possible_locator_set, dataset_remark = load_cfs_data(
        ground_truth_file,
        threshold_cfs=threshold_cfs,
    )
    family_truth_set = {t for t, e in tag_entity.items() if e == TagEntity.FAMILY}
    possible_locator_num = len(possible_locator_set)
    family_truth_num = len(family_truth_set)
    locator_truth_set = {t for t, e in tag_entity.items() if e in LOCATORS}
    # label
    label_engines = load_label_engines(vtapiv2_file)
    rich.print(f"[*] {dataset.name} labels = {len(label_engines)}")
    # init vocabulary
    init_voc = Vocabulary([INIT_LOCATOR_VOC_FILE, INIT_MISC_VOC_FILE])
    # loop
    round_metrics = []
    round = 0
    round_updated_num = 1
    while round_updated_num > 0:
        round += 1
        if round > max_round:
            break
        round_updated_num = 0
        rich.print(f"\n========== Locator update round {round} ==========")
        # start
        start_locator_set = set([t.name for t in init_voc.get_tags(LOCATORS)])
        start_family_set = set([t.name for t in init_voc.get_tags([TagEntity.FAMILY])])
        remain_locator_set = possible_locator_set - start_locator_set
        remain_family_set = family_truth_set - start_family_set
        rich.print("// initial vocabulary")
        rich.print(f"[*] {init_voc}")
        rich.print(f"// {dataset.name} summary")
        rich.print(f"[*] locators = {possible_locator_num}")
        rich.print(f"[*] families = {family_truth_num}")
        if table >= 1:
            inspect_remain_locators = {
                t: dataset_remark[t][:threshold_cfs]
                for i, t in enumerate(remain_locator_set)
                if i < remain_inspect
            }
            inspect_remain_families = {
                t: dataset_remark[t][:threshold_cfs]
                for i, t in enumerate(remain_family_set)
                if i < remain_inspect
            }
            rich.print(
                f"[*] {remain_inspect} of {len(remain_locator_set)} remain locators:"
            )
            rich.print(inspect_remain_locators)
            rich.print(
                f"[*] {remain_inspect} of {len(remain_family_set)} remain families:"
            )
            rich.print(inspect_remain_families)
        # updating
        rich.print("// LFS-CFS loop until no new locator")
        init_voc = locator_incremental_update(
            label_engines,
            voc=init_voc,
            threshold_cfs=threshold_cfs,
            lfs_mode=lfs_mode,
        )
        # end
        end_locator_set = set([t.name for t in init_voc.get_tags(LOCATORS)])
        updated_locator_set = set(
            [t.name for t in init_voc.get_tags(LOCATORS) if t.score == TagScore.UPDATED]
        )
        round_updated_set = end_locator_set - start_locator_set
        round_updated_num = len(round_updated_set)
        # metric
        rich.print("// loop summary")
        rich.print(f"[*] upated locators = {round_updated_num}")
        if table >= 1 and round_updated_num:
            inspect_updated_locators = {
                t: dataset_remark[t][:threshold_cfs] for t in round_updated_set
            }
            rich.print(inspect_updated_locators)

        pre_rec = precision_recall(
            updated_locator_set,
            locator_truth_set,
            end_locator_set,
            possible_locator_set,
        )
        rich.print(pre_rec)
        pre_rec["updated"] = round_updated_num
        round_metrics.append(pre_rec)

        rich.print(init_voc.get("monitor"))

        # simulates manually verifying the update locator
        rich.print("// imitating verification")
        for k, tag in init_voc.value.items():
            # only verify locators
            if tag.entity not in LOCATORS:
                continue
            # ignore old updated
            if tag.name in start_locator_set:
                continue
            # verify tag
            init_voc.update(
                k,
                entity=tag_entity[k],
                score=TagScore.UPDATED,
            )
            # verify remark
            if tag.remark is not None:
                remark = tag.remark.split("->")[0].strip()
                init_voc.update(
                    remark,
                    entity=tag_entity[remark],
                    score=TagScore.UPDATED,
                )
        # clean
        init_voc.value = {
            k: v
            for k, v in init_voc.value.items()
            if v.score in [TagScore.UPDATED, TagScore.CONFIRMED]
        }
        # dump
        if dump:
            init_voc.dump(LOCATORS, INIT_LOCATOR_VOC_FILE)
            init_voc.dump([TagEntity.MISC], INIT_MISC_VOC_FILE)
        # failed updated
        failed_update = possible_locator_set - set(
            [t.name for t in init_voc.get_tags(LOCATORS)]
        )
        if table >= 2 and failed_update:
            rich.print(f"[*] failed updated = {len(failed_update)}")
            rich.print(failed_update)
        rich.print("============================================")
    # report
    failed_update = possible_locator_set - set(
        [t.name for t in init_voc.get_tags(LOCATORS)]
    )
    rich.print("// report")
    if round <= max_round:
        rich.print(f"[-] LIU finishes at round {round}")

    else:
        rich.print(f"[-] LIU exceeds max round {max_round}")
    rich.print(f"[-] threshold_cfs = {threshold_cfs} | lfs_mode = {lfs_mode}")
    rich.print(
        f"[-] {dataset.name}: labels = {len(label_engines)} | locators = {len(possible_locator_set)}"
    )
    rich.print(f"[*] failed updated = {len(failed_update)}")
    rich.print("[*] metricss of each round: ")
    rich.print(round_metrics)


@app.command(short_help="test CFS threshold")
def threshold(
    dataset: str = "motif",
    max_threshold: int = 16,
    max_round: int = 3,
    lfs_mode: str = "updating",
):
    ground_truth_file = root / f"data/{dataset}/{dataset}-nermnl-label_parse.jsonl"
    vtapiv2_file = root / f"data/{dataset}/{dataset}-vtapiv2.jsonl"
    # malware labels
    label_engines = load_label_engines(vtapiv2_file)
    rich.print(f"[*] {dataset} labels = {len(label_engines)}")

    record = {}
    for threshold in range(2, max_threshold + 1):
        rich.print(f"========== threshold_cfs = {threshold} ==========")
        # init vocabulary
        init_voc = Vocabulary([INIT_LOCATOR_VOC_FILE, INIT_MISC_VOC_FILE])
        # truth
        tag_entity, possible_locator_set, _ = load_cfs_data(
            ground_truth_file,
            threshold_cfs=threshold,
        )
        locator_truth_set = {t for t, e in tag_entity.items() if e in LOCATORS}
        # loop
        round_metrics = []
        round = 0
        round_updated_num = 1
        while round_updated_num > 0:
            round += 1
            if round >= max_round:
                break
            round_updated_num = 0
            # start
            start_locator_set = set([t.name for t in init_voc.get_tags(LOCATORS)])
            # updating
            init_voc = locator_incremental_update(
                label_engines,
                voc=init_voc,
                threshold_cfs=threshold,
                lfs_mode=lfs_mode,
            )
            # end
            end_locator_set = set([t.name for t in init_voc.get_tags(LOCATORS)])
            updated_locator_set = set(
                [
                    t.name
                    for t in init_voc.get_tags(LOCATORS)
                    if t.score == TagScore.UPDATED
                ]
            )
            round_updated_set = end_locator_set - start_locator_set
            round_updated_num = len(round_updated_set)
            # pr
            pre_rec = precision_recall(
                updated_locator_set,
                locator_truth_set,
                end_locator_set,
                possible_locator_set,
            )
            pre_rec["updated"] = round_updated_num
            round_metrics.append(pre_rec)

            # simulates manually verifying the update locator
            rich.print("// imitating verification")
            for k, tag in init_voc.value.items():
                # ignore old updated
                if tag.name in start_locator_set:
                    continue
                # verify tag
                init_voc.update(
                    k,
                    entity=tag_entity[k],
                    score=TagScore.UPDATED,
                )
                # verify remark
                if tag.remark is not None:
                    remark = tag.remark.split("->")[0].strip()
                    init_voc.update(
                        remark,
                        entity=tag_entity[remark],
                        score=TagScore.UPDATED,
                    )
            # clean
            init_voc.value = {
                k: v
                for k, v in init_voc.value.items()
                if v.score in [TagScore.UPDATED, TagScore.CONFIRMED]
            }
        # report
        rich.print("// report")
        if round < max_round:
            rich.print(f"[-] LIU finishes at round {round}")

        else:
            rich.print(f"[-] LIU exceeds max round {round}")
        rich.print(f"[-] threshold_cfs = {threshold} | lfs_mode = {lfs_mode}")
        rich.print(
            f"[-] {dataset}: labels = {len(label_engines)} | locators = {len(possible_locator_set)}"
        )
        rich.print("[*] round_metricss of each round: ")
        rich.print(round_metrics)
        record[threshold] = {"init": round_metrics[0], "final": round_metrics[-1]}
    # records
    rich.print("[*] metrics of each threshold: ")
    rich.print(record)
    with open(Path.cwd() / "threshold-metirc.json", "w") as f:
        json.dump(record, f)


def load_lfs_data(
    nermnl_jsonl: str,
    *,
    threshold_cfs: int,
) -> Tuple[Vocabulary, Dict[str, str]]:
    count = {k: defaultdict(int) for k in LOCATORS}
    label_family = {}
    with open(nermnl_jsonl, "r") as f:
        for line in f:
            label, data = json.loads(line)
            label = label.strip().lower()
            family = [t for t, e in data.items() if e == TagEntity.FAMILY]
            if len(family) == 0:
                label_family[label] = None
            else:
                label_family[label] = family[-1]
                for t, e in data.items():
                    if e in count:
                        count[e][t] += 1

    voc = Vocabulary([INIT_LOCATOR_VOC_FILE, INIT_MISC_VOC_FILE])
    # Imitate CFS
    # tags with threshold_cfs >= 6 will be confirmed by the CFS
    for e, data in count.items():
        for t, c in data.items():
            if c >= threshold_cfs:
                voc.update(t, entity=e, score=TagScore.CONFIRMED)

    return voc, label_family


def print_error_table(
    data: List[Tuple],
    *,
    name="error",
    level="label",
):
    error_table = Table(title=name)

    if level == "label":
        column_style = {
            "engine": "cyan",
            "label": "cyan",
            "true": "cyan",
            "euphony": "green",
            "tagclass": "red",
        }
    elif level == "sample":
        column_style = {
            "md5": "cyan",
            "true": "cyan",
            "avclass": "green",
            "tagclass": "red",
        }
    else:
        raise ValueError(f"level {level} not support!")

    for c, s in column_style.items():
        error_table.add_column(c, style=s)

    columns = list(column_style.keys())
    resutls = []
    for item in data:
        error_table.add_row(*item)
        resutls.append({k: v for k, v in zip(columns, item)})

    save_path = Path.cwd() / f"logs/{name}.jsonl"
    rich.print(error_table)
    with open(save_path, "w") as f:
        for item in resutls:
            f.write(json.dumps(item) + "\n")


def evaluate_label_level_parse(
    vtapiv2_file: Path,
    dataset: str,
    ground_truth_file: Path,
    euphony_parsed_file: Path,
    table: str = "T0",
):
    if any(
        not i.exists() for i in [vtapiv2_file, ground_truth_file, euphony_parsed_file]
    ):
        rich.print("[x] Finish dataset first!")
        raise typer.Exit(-1)

    # voc & groundtruth & parser
    _, groundtruth = load_lfs_data(ground_truth_file, threshold_cfs=6)
    voc = Vocabulary([LOCATOR_VOC_FILE, MISC_VOC_FILE])
    parser = Parser(
        Tokenizer(), mode=RunMode.PARSE, sifter_json=SIFTER_FILE, ignore_generic=False
    )

    # load
    label_engines = load_label_engines(vtapiv2_file)
    with open(euphony_parsed_file, "r") as f:
        euphony_result: Dict[str, str] = json.load(f)
    # tagclass
    results: dict[str, ParseResult] = {}
    for label, engines in label_engines.items():
        data = parser.parse(
            label,
            engine=engines[0],
            voc=voc,
        )
        label_lower = label.strip().lower()
        family = None
        score = TagScore.UNKNOWN
        for r in data:
            if r.entity == TagEntity.FAMILY:
                family = r.tag
                score = r.score
                break

        if label_lower not in results:
            results[label_lower] = ParseResult(
                label,
                engines[0],
                family,
                score,
            )
        elif score > results[label_lower].score:
            results[label_lower] = ParseResult(
                label,
                engines[0],
                family,
                score,
            )

    # euphony scope acc
    scope_acc = defaultdict(list)
    eup1_tag0 = []
    scope_tag0 = []
    tag1_eup0 = []
    eup0 = []
    for label, euphony_family in euphony_result.items():
        # gt
        label = label.strip().lower()
        if label not in groundtruth:
            continue

        gt_fam = groundtruth[label]
        # tagclass
        parsed = results[label]
        label_origin = parsed.label
        engine = parsed.engine
        tagclass_family = parsed.family
        # acc
        eup = gt_fam == euphony_family
        tag = gt_fam == tagclass_family
        scope_acc["euphony"].append(eup)
        scope_acc["tagclass"].append(tag)
        record = (engine, label_origin, gt_fam, euphony_family, tagclass_family)
        # eup0
        if not eup:
            eup0.append(record)
        # tag0
        if not tag:
            scope_tag0.append(record)
            if eup:
                eup1_tag0.append(record)
        if tag and not eup:
            tag1_eup0.append(record)
    # all acc
    tagclass_acc = []
    tag0 = []
    for label, parsed in results.items():
        if label not in groundtruth:
            continue

        euphony_family = euphony_result.get(label, "OOV")
        tagclass_family = parsed.family

        gt_fam = groundtruth[label]
        acc = gt_fam == tagclass_family
        tagclass_acc.append(acc)
        label_origin = parsed.label
        engine = parsed.engine
        record = (engine, label_origin, gt_fam, euphony_family, tagclass_family)
        if not acc:
            tag0.append(record)

    match table:
        case "T0E":
            print_error_table(
                scope_tag0,
                name=f"TagClass-Parse-Error-EScope-on-{dataset}",
            )
        case "E1T0":
            print_error_table(
                eup1_tag0,
                name=f"Euphony-Success-TagClass-Error-on-{dataset}",
            )
        case "E0T1":
            print_error_table(
                tag1_eup0,
                name=f"Euphony-Error-TagClass-Success-on-{dataset}",
            )
        case "T0":
            print_error_table(tag0, name=f"TagClass-Parse-Error-on-{dataset}")

    # summary
    # scope acc
    rich.print(
        f"""[*] ==================================================
    Label-level Accuracy on {dataset} 
    Labels = {len(groundtruth)}
    Euphony scope = {len(euphony_result)}
    Euphony success but Tagclass failed = {len(eup1_tag0)}
    Tagclass failed in the Scope of Euphony = {len(scope_tag0)}
    Euphony Acc = {sum(scope_acc['euphony']) / len(scope_acc['euphony'])}
    Tagclass Acc = {sum(scope_acc['tagclass']) / len(scope_acc['tagclass'])}
    =================================================="""
    )
    # # all acc
    # rich.print(
    #     f"""[*] ============= Acc of all labels =============
    # Labels = {len(groundtruth)}
    # Tagclass failed  = {len(tag0)}
    # Tagclass Acc = {sum(tagclass_acc) / len(tagclass_acc)}
    # ============================================="""
    # )


def rank2string(rank: list[tuple[str, int]]) -> str:
    data = [f"{k}-{v}" for k, v in rank]
    return " ".join(data)


def evaluate_sample_level_parse(
    vtapiv2_file: Path,
    dataset: str,
    ground_truth_file: Path,
    avclass_parsed_file: Path,
    topk: int = 3,
    table: str = None,
):
    if any(
        not i.exists()
        for i in [
            vtapiv2_file,
            ground_truth_file,
            avclass_parsed_file,
        ]
    ):
        rich.print("[x] Finish dataset first!")
        raise typer.Exit(-1)

    dataset = vtapiv2_file.parent.name
    # tagalias
    tagalias = TagAlias(ALIAS_FILE)

    # groundtruth
    groundtruth: dict[str, str] = json.loads(ground_truth_file.read_text())
    groundtruth = {k: tagalias(v) for k, v in groundtruth.items()}

    voc = Vocabulary([LOCATOR_VOC_FILE, MISC_VOC_FILE])
    parser = Parser(
        Tokenizer(), mode=RunMode.PARSE, sifter_json=SIFTER_FILE, ignore_generic=True
    )

    # avclass
    avclass_result: dict[str, dict[str, int]] = {}
    with open(avclass_parsed_file, "r") as f:
        data = json.load(f)
        for md5, tags in data.items():
            avclass_result[md5] = [(tagalias(k), v) for k, v in tags]

    # tagclass
    tagclass_result: dict[str, list[tuple[str, int]]] = {}
    sample_label_engines = load_sample_label_engines(vtapiv2_file)
    for md5, label_engines in sample_label_engines.items():
        family_tags = defaultdict(int)
        for label, engines in label_engines.items():
            data = parser.parse(
                label,
                engine=engines[0],
                voc=voc,
            )
            for r in data:
                if r.entity == TagEntity.FAMILY:
                    family_tags[tagalias(r.tag)] += 1
        family_tags = sorted(
            family_tags.items(), key=lambda x: (x[1], x[0]), reverse=True
        )
        tagclass_result[md5] = family_tags

    # all acc
    tagclass_acc = []
    avclass_acc = []
    av1_tag0 = []
    tag1_av0 = []
    tag0 = []
    av0 = []
    for md5, gt_fam in groundtruth.items():
        if md5 not in tagclass_result:
            continue
        if md5 not in avclass_result:
            continue

        ranked_family = tagclass_result[md5]

        if dataset in ["motif", "bodmas"]:
            ranked_family = [(f, c) for f, c in ranked_family if c >= 2]

        # Rule 1: ignore null-tagged samples
        if len(ranked_family) == 0:
            continue

        # Rule 2: ignore groundtruth family not in parsed
        parsed_families = [v for v, _ in ranked_family]
        if gt_fam not in parsed_families:
            continue

        # check topk
        tagclass_current = ranked_family
        avclass_current = avclass_result[md5]
        tagclass_topk = {k for k, _ in tagclass_current[:topk]}
        avclass_topk = {k for k, _ in avclass_current[:topk]}
        av_acc = gt_fam in avclass_topk
        tag_acc = gt_fam in tagclass_topk
        tagclass_acc.append(tag_acc)
        avclass_acc.append(av_acc)

        verbose = (
            md5,
            gt_fam,
            rank2string(avclass_current),
            rank2string(tagclass_current),
        )
        if not tag_acc:
            tag0.append(verbose)

        if not av_acc:
            av0.append(verbose)

        if av_acc and not tag_acc:
            av1_tag0.append(verbose)

        if tag_acc and not av_acc:
            tag1_av0.append(verbose)

    # sort
    av0 = sorted(av0, key=lambda x: x[1])
    av1_tag0 = sorted(av1_tag0, key=lambda x: x[1])
    tag0 = sorted(tag0, key=lambda x: x[1])

    # error
    match table:
        case "A0":
            print_error_table(
                av0,
                name=f"AVClass-Parse-Error-on-{dataset}",
                level="sample",
            )
        case "A1T0":
            print_error_table(
                av1_tag0,
                name=f"AVClass-Success-TagClass-Error-on-{dataset}",
                level="sample",
            )
        case "A0T1":
            print_error_table(
                tag1_av0,
                name=f"AVClass-Error-TagClass-Success-on-{dataset}",
                level="sample",
            )
        case "T0":
            print_error_table(
                tag0,
                name=f"TagClass-Parse-Error-on-{dataset}",
                level="sample",
            )

    # summary
    rich.print(
        f"""[*] ===========================================
    Sample-level Accuracy on {dataset} 
    Samples = {len(tagclass_acc)}
    AVClass failed = {len(av0)}
    AVClass success but Tagclass failed = {len(av1_tag0)}
    TagClass failed = {len(tag0)}
    AVClass Acc = {sum(avclass_acc) / len(avclass_acc)}
    Tagclass Acc = {sum(tagclass_acc) / len(tagclass_acc)}
    ==========================================="""
    )


@app.command(
    short_help="test location first search for sample-level or label-level parsing"
)
def parse(
    dataset: str = "malgenome",
    level: str = "label",
    table: str = None,
):
    if dataset not in ["malgenome", "drebin", "motif", "bodmas"]:
        rich.print("[x] Dataset not support!")
        raise typer.Exit(-1)

    data_path = root / f"data/{dataset}"
    vtapiv2_file = data_path / f"{dataset}-vtapiv2.jsonl"
    nermnl_file = data_path / f"{dataset}-nermnl-label_parse.jsonl"
    sample_family_file = data_path / f"{dataset}-sample-family.json"
    euphony_parsed_file = data_path / f"{dataset}-euphony-parsed.json"
    avclass_parsed_file = data_path / f"{dataset}-avclass-parsed.json"

    if level == "label":
        evaluate_label_level_parse(
            vtapiv2_file=vtapiv2_file,
            dataset=dataset,
            ground_truth_file=nermnl_file,
            euphony_parsed_file=euphony_parsed_file,
            table=table,
        )
    elif level == "sample":
        evaluate_sample_level_parse(
            vtapiv2_file=vtapiv2_file,
            dataset=dataset,
            ground_truth_file=sample_family_file,
            avclass_parsed_file=avclass_parsed_file,
            table=table,
        )
    else:
        rich.print(f"level {level} not support!")
        raise typer.Exit(-1)


if __name__ == "__main__":
    update(
        malgenome=True,
        drebin=False,
        threshold_cfs=2,
        max_round=5,
        dump=False,
        lfs_mode=RunMode.UPDATE,
        table=1,
    )
