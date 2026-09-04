"""统一数据集加载。

把 offline_exp/data 下的各个数据集读成同一种记录：
    {idx: str, question: str, gold: str|None, meta: dict}

question 是直接发给模型的完整 prompt（含指令），gold 是用于打分的标准答案。
代码类（humaneval / livecodebench / mbpp）无法用字符串比对，gold 为 None，
判分所需的测试用例放在 meta 里。
"""
import json
import os

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

NUM_SUFFIX = "\n\n请逐步推理。最后一行只输出最终答案，格式为：Answer: <答案>"
MCQ_SUFFIX = "\n\n只输出正确选项的字母，不要解释。"


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mcq(stem, labels, texts):
    opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return f"{stem}\n\n{opts}{MCQ_SUFFIX}"


# --------------------------------------------------------------- 评测集


def gsm8k():
    df = pd.read_parquet(f"{DATA}/eval/gsm8k/main/test-00000-of-00001.parquet")
    return [
        dict(idx=f"gsm8k-{i}", question=r.question + NUM_SUFFIX,
             gold=r.answer.split("####")[-1].strip().replace(",", ""), meta={})
        for i, r in enumerate(df.itertuples())
    ]


def math500():
    rows = _jsonl(f"{DATA}/eval/math500/test.jsonl")
    return [
        dict(idx=f"math500-{r['unique_id']}", question=r["problem"] + NUM_SUFFIX,
             gold=str(r["answer"]), meta={"subject": r["subject"], "level": r["level"]})
        for r in rows
    ]


def arc_challenge():
    df = pd.read_parquet(f"{DATA}/eval/arc_challenge/ARC-Challenge/test-00000-of-00001.parquet")
    out = []
    for r in df.itertuples():
        ch = r.choices
        out.append(dict(idx=f"arc-{r.id}", question=_mcq(r.question, ch["label"], ch["text"]),
                        gold=str(r.answerKey), meta={"labels": list(ch["label"])}))
    return out


def financebench():
    rows = _jsonl(f"{DATA}/eval/financebench/financebench_merged.jsonl")
    out = []
    for r in rows:
        ev = "\n\n".join(e["evidence_text"] for e in (r.get("evidence") or []) if e.get("evidence_text"))
        q = (f"以下是 {r['company']} 的财报节选（{r['doc_name']}）：\n\n{ev}\n\n"
             f"问题：{r['question']}\n\n请基于上述材料，只输出一行：Answer: <答案>。不要解释。")
        out.append(dict(idx=f"financebench-{r['financebench_id']}", question=q,
                        gold=str(r["answer"]), meta={"question_type": r["question_type"]}))
    return out


def humaneval():
    df = pd.read_parquet(f"{DATA}/eval/humaneval/openai_humaneval/test-00000-of-00001.parquet")
    out = []
    for r in df.itertuples():
        q = ("补全下面的 Python 函数，把完整实现放在一个 ```python 代码块里，不要写测试代码。\n\n"
             f"```python\n{r.prompt}```")
        out.append(dict(idx=f"humaneval-{r.task_id.replace('/', '_')}", question=q, gold=None,
                        meta={"prompt": r.prompt, "test": r.test, "entry_point": r.entry_point}))
    return out


# LCB 的 release_vN = test.jsonl .. testN.jsonl 累加，见仓库内 code_generation_lite.py
LCB_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]


def livecodebench(release=6):
    out = []
    for name in LCB_FILES[:release]:
        for r in _jsonl(f"{DATA}/eval/livecodebench/{name}"):
            fn = (json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]).get("func_name")
            mode = "调用函数" if r.get("starter_code") else "从 stdin 读入、往 stdout 输出"
            q = (f"{r['question_content']}\n\n用 Python 解决上述问题，{mode}。"
                 f"把完整代码放在一个 ```python 代码块里。")
            if r.get("starter_code"):
                q += f"\n\n从这个模板开始：\n```python\n{r['starter_code']}\n```"
            out.append(dict(idx=f"lcb-{r['question_id']}", question=q, gold=None,
                            meta={"platform": r["platform"], "difficulty": r["difficulty"],
                                  "contest_date": r["contest_date"], "func_name": fn,
                                  "public_test_cases": r["public_test_cases"]}))
    return out


# --------------------------------------------------------------- 校准集
# 校准只需要"输入 + 模型自生成回答"，gold 仅留作记录，离线分析时不使用。


def asdiv():
    # ASDiv release bundled in this experiment only publishes validation.
    # Keep it as the sole available calibration split rather than inventing
    # a train split from it.
    df = pd.read_parquet(f"{DATA}/calib/asdiv/asdiv/validation-00000-of-00001.parquet")
    return [
        dict(idx=f"asdiv-{i}", question=f"{r.body} {r.question}" + NUM_SUFFIX,
             gold=str(r.answer), meta={"solution_type": r.solution_type})
        for i, r in enumerate(df.itertuples())
    ]


def aime():
    # Keep the legacy AIME calibration composition unchanged:
    # 2022-2024 train plus the separately released AIME 2025 test.
    # The 2025 test is retained here for backward compatibility with
    # previous runs; it is not part of the new four-dataset queue.
    out = []
    answer_only = "\n\n只输出一行：Answer: <答案>。不要解释。"
    df = pd.read_parquet(f"{DATA}/calib/aime/aime2022_2024/data/train-00000-of-00001.parquet")
    for r in df.itertuples():
        out.append(dict(idx=f"aime-{r.id}", question=r.problem + answer_only,
                        gold=str(r.answer), meta={"url": r.url}))
    for r in _jsonl(f"{DATA}/calib/aime/aime2025/test.jsonl"):
        out.append(dict(idx=f"aime25-{r['id']}", question=r["problem"] + answer_only,
                        gold=str(r["answer"]), meta={}))
    return out


def alime():
    """Backward-compatible spelling used by the experiment commands."""
    return aime()


def mbpp():
    df = pd.read_parquet(f"{DATA}/calib/mbpp/full/train-00000-of-00001.parquet")
    out = []
    for r in df.itertuples():
        tests = "\n".join(r.test_list)
        q = (f"你是 Python 专家，请完成这个任务：{r.text}\n\n"
             f"你的代码需要通过这些测试：\n```python\n{tests}\n```\n\n"
             "把完整代码放在一个 ```python 代码块里。")
        out.append(dict(idx=f"mbpp-{r.task_id}", question=q, gold=None,
                        meta={"test_list": list(r.test_list),
                              "test_setup_code": r.test_setup_code or ""}))
    return out


def openbookqa():
    df = pd.read_parquet(f"{DATA}/calib/openbookqa/main/train-00000-of-00001.parquet")
    return [
        dict(idx=f"obqa-{r.id}", question=_mcq(r.question_stem, r.choices["label"], r.choices["text"]),
             gold=str(r.answerKey), meta={})
        for r in df.itertuples()
    ]


def finqa():
    """FinQA dev split used for financial calibration.

    dev rather than train: same dataset, same annotation pass, so swapping the
    split costs nothing in similarity to FinanceBench, while 883 rows instead of
    6,251 keeps the 32k-context generation budget reasonable.
    """
    with open(f"{DATA}/calib/finqa/official/dev.json", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        table = "\n".join(" | ".join(str(c) for c in row) for row in r["table"])
        ctx = "\n".join(r["pre_text"]) + f"\n\n{table}\n\n" + "\n".join(r["post_text"])
        q = (f"以下是财报节选：\n\n{ctx}\n\n问题：{r['qa']['question']}\n\n"
             "请基于上述材料回答，最后一行输出：Answer: <答案>")
        out.append(dict(idx=f"finqa-{r['id']}", question=q, gold=str(r["qa"]["exe_ans"]),
                        meta={"answer_str": r["qa"]["answer"], "program": r["qa"].get("program")}))
    return out


def medmcqa():
    """MedMCQA validation split (NEET PG) used for medical calibration.

    Replaces PubMedQA: yes/no/maybe answers produced reasoning chains far shorter
    than MedQA-USMLE's, so they never activated the routing pattern being measured.
    Four options like MedQA, and disjoint by exam system (Indian AIIMS/NEET PG vs
    US USMLE).  validation rather than train because train is 182,822 rows, which
    is not a similarity question but a generation-budget one.
    """
    df = pd.read_parquet(f"{DATA}/calib/medmcqa/data/validation-00000-of-00001.parquet")
    labels = ["A", "B", "C", "D"]
    out = []
    for r in df.itertuples():
        q = _mcq(r.question, labels, [r.opa, r.opb, r.opc, r.opd])
        # cop is a 0-based index into opa..opd, so map it onto the letter labels.
        out.append(dict(idx=f"medmcqa-{r.id}", question=q, gold=labels[int(r.cop)],
                        meta={"source_split": "validation", "subject": r.subject_name,
                              "choice_type": r.choice_type}))
    return out


def medqa_usmle():
    """USMLE multiple-choice test set used for medical evaluation."""
    rows = _jsonl(f"{DATA}/eval/medqa_usmle/phrases_no_exclude_test.jsonl")
    out = []
    for i, r in enumerate(rows):
        labels = list(r["options"])
        q = _mcq(r["question"], labels, [r["options"][label] for label in labels])
        out.append(dict(idx=f"medqa-usmle-{i}", question=q,
                        gold=str(r["answer_idx"]), meta={"source_split": "test"}))
    return out


LEXAM_MCQ_CONFIGS = ["mcq_4_choices", "mcq_8_choices", "mcq_16_choices", "mcq_32_choices"]


def lexam_law():
    """LEXam law-exam MCQ, English subset, used for legal calibration.

    Replaces legalqaeval (isaacus/LegalQAEval), which is SQuAD 2.0 / MS MARCO /
    HotpotQA / Natural Questions validation examples filtered for legal topicality
    -- general web extractive QA, not legal reasoning.

    LEXam groups the *same* questions into four configs by how many option
    combinations they support, so the configs overlap heavily: mcq_8 and mcq_16
    are strict subsets of mcq_4.  mcq_32 is not -- it holds 93 English questions
    mcq_4 lacks.  Reading all four and de-duplicating on id therefore gives the
    full English pool (712) rather than mcq_4 alone (619), and never counts a
    question twice.  Config order is fixed so a question keeps the smallest
    option count it appears with, which makes the prompt deterministic.
    """
    seen, out = set(), []
    for config in LEXAM_MCQ_CONFIGS:
        df = pd.read_parquet(f"{DATA}/calib/lexam_law/{config}/test-00000-of-00001.parquet")
        for r in df[df["language"] == "en"].itertuples():
            if r.id in seen:
                continue
            seen.add(r.id)
            choices = list(r.choices)
            labels = [chr(ord("A") + i) for i in range(len(choices))]
            q = _mcq(r.question, labels, choices)
            # gold is a 0-based index into choices.
            out.append(dict(idx=f"lexam-{r.id}", question=q, gold=labels[int(r.gold)],
                            meta={"source_split": "test", "config": config,
                                  "course": r.course, "area": r.area,
                                  "jurisdiction": r.jurisdiction, "year": int(r.year),
                                  "n_statements": int(r.n_statements)}))
    return out


def mmlu_pro_law():
    """MMLU-Pro law test subset used for legal evaluation."""
    df = pd.read_parquet(f"{DATA}/eval/mmlu_pro/data/test-00000-of-00001.parquet")
    out = []
    for r in df.itertuples():
        if r.category != "law":
            continue
        labels = [chr(ord("A") + i) for i in range(len(r.options))]
        q = _mcq(r.question, labels, list(r.options))
        out.append(dict(idx=f"mmlu-pro-law-{r.question_id}", question=q,
                        gold=str(r.answer), meta={"labels": labels,
                                                   "source_split": "test"}))
    return out


LOADERS = {
    "gsm8k": gsm8k, "math500": math500, "arc_challenge": arc_challenge,
    "financebench": financebench, "humaneval": humaneval, "livecodebench": livecodebench,
    "asdiv": asdiv, "aime": aime, "alime": alime, "mbpp": mbpp, "openbookqa": openbookqa,
    "finqa": finqa, "medmcqa": medmcqa, "medqa_usmle": medqa_usmle,
    "lexam_law": lexam_law, "mmlu_pro_law": mmlu_pro_law,
}
# The five domains actually in the experiment.  arc_challenge / openbookqa (the
# dropped science domain) and aime / alime stay in LOADERS so they can still be
# run by name, but they are out of the groups below so that `--dataset eval`,
# `--dataset calib` and `--dataset all` match the configuration in data/README.md.
EVAL_SETS = [
    "gsm8k", "math500",            # 数学
    "humaneval", "livecodebench",  # 代码
    "financebench",                # 金融
    "medqa_usmle",                 # 医学
    "mmlu_pro_law",                # 法律
]
CALIB_SETS = [
    "asdiv",        # 数学
    "mbpp",         # 代码
    "finqa",        # 金融  (dev)
    "medmcqa",      # 医学  (validation)
    "lexam_law",    # 法律  (English MCQ)
]


def load(name, **kw):
    return LOADERS[name](**kw)
