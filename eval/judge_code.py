"""代码类数据集的判分：真跑测试用例，不用 LLM 看。

    HumanEval      自己跑：抽出代码 + 官方 test + check(entry_point)，子进程执行
    MBPP           自己跑：抽出代码 + 每题 test_list，子进程执行
    LiveCodeBench  用官方 lcb_runner/evaluation（已 vendor，blob SHA 与上游一致），
                   保证 stdin/functional 两种模式的判定口径和 leaderboard 相同

用法：
    python eval/judge_code.py --tag baseline --dataset humaneval,livecodebench,mbpp

注意：执行的是模型生成的代码。LCB 官方的 run_test 会先调 reliability_guard()
屏蔽一批破坏性调用并限制 4GB 内存，HumanEval 这边我用独立子进程 + 超时 + 临时空目录。
两者都不是真沙箱，容器隔离才是——当前环境本身就在容器里，按需自行评估。
"""
import argparse
import base64
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loaders as D  # noqa: E402
from judge import write_summary  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strip_think(text):
    if not text:
        return ""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return "" if "<think>" in text else text


def extract_code(text):
    """取最后一个 ```python 代码块；没有就退而取任意 ``` 块。"""
    body = strip_think(text)
    if not body:
        return ""
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", body, flags=re.S)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", body, flags=re.S)
    return blocks[-1].strip() if blocks else ""


# ------------------------------------------------------------ 代码产物

def code_path(tag, name):
    return os.path.join(ROOT, "runs", tag, "code", name + ".code.jsonl")


def build_code_artifact(tag, name):
    """从推理结果里抽出代码，单独存一份 <ds>.code.jsonl。

    三个阶段各自落盘、互不重复劳动：
        output/<ds>.jsonl        模型原始回答（run_infer 产出，最贵，绝不重跑）
        code/<ds>.code.jsonl     抽出来的代码（本函数产出，可反复重建）
        score/<ds>.scored.jsonl  执行判分结果
    """
    rows = [{"idx": r["idx"], "code": extract_code(r.get("answer")),
              "finish_reason": r.get("finish_reason", "stop")} for r in load_run(tag, name)]
    os.makedirs(os.path.dirname(code_path(tag, name)), exist_ok=True)
    with open(code_path(tag, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_empty = sum(1 for r in rows if not r["code"])
    print(f"  [{name}] 抽出代码 {len(rows)} 条，其中 {n_empty} 条没找到代码块 "
          f"-> {os.path.basename(code_path(tag, name))}", flush=True)
    return rows


def load_code_artifact(tag, name, rebuild=False):
    p = code_path(tag, name)
    if rebuild or not os.path.exists(p):
        return build_code_artifact(tag, name)
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"  [{name}] 复用已有代码产物 {len(rows)} 条（--rebuild-code 可重建）", flush=True)
    return rows


# ------------------------------------------------------------ HumanEval

HE_TIMEOUT = 15
MBPP_TIMEOUT = 15


def run_humaneval_one(job):
    """在独立子进程里跑 官方 test + check(entry_point)。返回 (idx, 是否通过, 原因)。"""
    idx, code, prompt, test, entry_point = job
    if not code:
        return idx, False, "回答里没有代码块"
    # 抽出来的代码里若没定义目标函数，就退回 prompt + 补全的拼法
    solution = code if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code) else prompt + "\n" + code
    script = solution + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, "-c", script], cwd=d, timeout=HE_TIMEOUT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            return idx, False, f"超时 >{HE_TIMEOUT}s"
    if p.returncode == 0:
        return idx, True, "pass"
    tail = (p.stderr or b"").decode("utf-8", "replace").strip().splitlines()
    return idx, False, (tail[-1] if tail else f"退出码 {p.returncode}")[:150]


def judge_humaneval(tag, workers, code_rows):
    recs = code_rows
    meta = {r["idx"]: r["meta"] for r in D.load("humaneval")}
    jobs = []
    for r in recs:
        m = meta[r["idx"]]
        jobs.append((r["idx"], r["code"], m["prompt"], m["test"], m["entry_point"]))
    trunc = {r["idx"] for r in recs if r.get("finish_reason") == "length"}
    jobs = [j for j in jobs if j[0] not in trunc]
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_humaneval_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            idx, ok, why = f.result()
            out[idx] = (ok, why)
            if i % 50 == 0:
                print(f"  [humaneval] {i}/{len(jobs)}", flush=True)
    res = []
    for r in recs:
        if r["idx"] in trunc:
            # 被 max_tokens 截断：没答完，不计入分母
            res.append({"idx": r["idx"], "correct": None, "method": "truncated",
                        "note": "被 max_tokens 截断，未计分"})
        else:
            ok, why = out[r["idx"]]
            res.append({"idx": r["idx"], "correct": ok,
                        "method": "no_code" if not r["code"] else "exec", "note": why})
    return res


# ------------------------------------------------------------ MBPP


def run_mbpp_one(job):
    """Run an MBPP completion against all tests in its test_list."""
    idx, code, setup_code, tests = job
    if not code:
        return idx, False, "回答里没有代码块"
    # Some MBPP setup snippets instantiate classes defined by the completion.
    # Imports in setup still run before the assertions invoke generated functions.
    script = "\n".join(x for x in (code, setup_code, *tests) if x)
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, "-c", script], cwd=d,
                               timeout=MBPP_TIMEOUT, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            return idx, False, f"超时 >{MBPP_TIMEOUT}s"
    if p.returncode == 0:
        return idx, True, f"{len(tests)}/{len(tests)} 用例通过"
    tail = (p.stderr or b"").decode("utf-8", "replace").strip().splitlines()
    why = tail[-1] if tail else f"退出码 {p.returncode}"
    return idx, False, f"0/{len(tests)} 用例通过: {why[:120]}"


def judge_mbpp(tag, workers, code_rows):
    meta = {r["idx"]: r["meta"] for r in D.load("mbpp")}
    trunc = {r["idx"] for r in code_rows if r.get("finish_reason") == "length"}
    jobs = []
    for r in code_rows:
        if r["idx"] in trunc:
            continue
        m = meta.get(r["idx"], {})
        jobs.append((r["idx"], r["code"], m.get("test_setup_code", ""),
                     m.get("test_list", [])))
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_mbpp_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            idx, ok, why = f.result()
            out[idx] = (ok, why)
            if i % 50 == 0:
                print(f"  [mbpp] {i}/{len(jobs)}", flush=True)
    res = []
    for r in code_rows:
        if r["idx"] in trunc:
            res.append({"idx": r["idx"], "correct": None, "method": "truncated",
                        "note": "被 max_tokens 截断，未计分"})
        else:
            ok, why = out.get(r["idx"], (False, "没有对应测试元数据"))
            res.append({"idx": r["idx"], "correct": ok,
                        "method": "no_code" if not r["code"] else "exec", "note": why})
    return res


# ------------------------------------------------------------ LiveCodeBench


def decode_private(blob):
    """private_test_cases 是四层套娃：base64 -> zlib -> pickle -> json 字符串。
    实测 400 条全部如此，pickle 解出来的是 str 而不是 list，还要再 json.loads 一次。"""
    dec = zlib.decompress(base64.b64decode(blob))
    try:
        obj = pickle.loads(dec)
    except Exception:
        obj = dec.decode()
    if isinstance(obj, (str, bytes)):
        obj = json.loads(obj)
    return obj


def build_lcb_sample(row):
    """拼成官方 run_test 要的 {"input_output": json_str} 形式。"""
    cases = json.loads(row["public_test_cases"]) + decode_private(row["private_test_cases"])
    md = row["metadata"]
    md = json.loads(md) if isinstance(md, str) else (md or {})
    io = {"inputs": [c["input"] for c in cases], "outputs": [c["output"] for c in cases]}
    if md.get("func_name"):
        io["fn_name"] = md["func_name"]
    return {"input_output": json.dumps(io)}


def judge_livecodebench(tag, workers, release, timeout, code_rows):
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    recs = code_rows
    rows = {}
    for name in D.LCB_FILES[:release]:
        with open(os.path.join(D.DATA, "eval/livecodebench", name), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[f"lcb-{r['question_id']}"] = r
    recs = [r for r in recs if r["idx"] in rows]
    trunc = [r for r in recs if r.get("finish_reason") == "length"]
    live = [r for r in recs if r.get("finish_reason") != "length"]
    if trunc:
        print(f"  [livecodebench] {len(trunc)} 条被截断，不计入分母", flush=True)
    samples = [build_lcb_sample(rows[r["idx"]]) for r in live]
    gens = [[r["code"]] for r in live]
    print(f"  [livecodebench] {len(live)} 题送入官方判分器，单用例超时 {timeout}s", flush=True)
    metrics, results, _ = codegen_metrics(samples, gens, k_list=[1],
                                          num_process_evaluate=workers, timeout=timeout)
    out = []
    for i, r in enumerate(live):
        # results[i] 是 [[每个用例的结果]]，全部为 True/1 才算通过
        res = results.get(i) or results.get(str(i)) or [[]]
        flat = res[0] if res else []
        ok = bool(flat) and all(x is True or x == 1 for x in flat)
        n_pass = sum(1 for x in flat if x is True or x == 1)
        out.append({"idx": r["idx"], "correct": ok,
                    "method": "no_code" if not r["code"] else "exec",
                    "note": f"{n_pass}/{len(flat)} 用例通过"})
    for r in trunc:
        out.append({"idx": r["idx"], "correct": None, "method": "truncated",
                    "note": "被 max_tokens 截断，未计分"})
    print(f"  [livecodebench] 官方 metrics: {metrics}", flush=True)
    return out


# ------------------------------------------------------------ 驱动


def load_run(tag, name):
    """读推理结果，按 idx 去重只留正常结束的那条。"""
    path = os.path.join(ROOT, "runs", tag, "output", name + ".jsonl")
    if not os.path.exists(path):
        return []
    best = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rank = (r.get("error") is None, r.get("finish_reason", "stop") == "stop")
            if r["idx"] not in best or rank >= best[r["idx"]][0]:
                best[r["idx"]] = (rank, r)
    # best[idx] = (rank, rec)，所以要取 v[1][1]
    return [v[1][1] for v in sorted(best.items())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="baseline")
    p.add_argument("--dataset", default="humaneval,livecodebench")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--lcb-release", type=int, default=1)
    p.add_argument("--lcb-timeout", type=int, default=6, help="LCB 单个用例的超时，官方默认 6s")
    p.add_argument("--rebuild-code", action="store_true", help="重新从回答里抽代码，覆盖已有产物")
    p.add_argument("--extract-only", action="store_true", help="只抽代码存产物，不执行")
    args = p.parse_args()

    rows = []
    for name in [s.strip() for s in args.dataset.split(",")]:
        recs = load_run(args.tag, name)
        if not recs:
            print(f"[{name}] 没有推理结果，跳过")
            continue
        print(f"[{name}] {len(recs)} 条待判")
        code_rows = load_code_artifact(args.tag, name, args.rebuild_code)
        if args.extract_only:
            continue
        if name == "humaneval":
            scored = judge_humaneval(args.tag, args.workers, code_rows)
        elif name == "mbpp":
            scored = judge_mbpp(args.tag, args.workers, code_rows)
        else:
            scored = judge_livecodebench(args.tag, args.workers, args.lcb_release,
                                         args.lcb_timeout, code_rows)
        out = os.path.join(ROOT, "runs", args.tag, "score", name + ".scored.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in scored:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        judged = [r for r in scored if r["correct"] is not None]
        n_ok = sum(1 for r in judged if r["correct"])
        methods, unjudged = {}, {}
        for r in scored:
            methods[r["method"]] = methods.get(r["method"], 0) + 1
            if r["correct"] is None:
                unjudged[r["method"]] = unjudged.get(r["method"], 0) + 1
        rows.append(dict(dataset=name, total=len(scored), judged=len(judged), correct=n_ok,
                         acc=(n_ok / len(judged) if judged else None),
                         # truncated 不计入分母；no_code 是正常收尾但抽不到代码，计失败
                         unjudged_excluded_from_denominator=unjudged, methods=methods))

    print(f"\n{'数据集':<16}{'判分/总数':>13}{'通过':>7}{'pass@1':>9}   说明")
    for r in rows:
        nc = r["methods"].get("no_code", 0)
        note = f"截断未计 {r['total'] - r['judged']} 条"
        if nc:
            note += f"，答完但抽不到代码计失败 {nc} 条"
        print(f"{r['dataset']:<16}{r['judged']:>6}/{r['total']:<6}{r['correct']:>7}"
              f"{(r['acc'] or 0) * 100:>8.1f}%   {note}")
    if rows:
        print(f"\n汇总写入 {write_summary(args.tag, rows)}")


if __name__ == "__main__":
    main()
