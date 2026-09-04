"""对 run_infer.py 的产出打分。

判分策略按数据集分工，不是一律交给 LLM：
    数值题（gsm8k / asdiv / finqa / aime）  规则抽取 + 数值归一化比对，规则判不了才降级到 LLM
    选择题（arc_challenge / openbookqa）    抽取选项字母，纯规则
    LaTeX 答案（math500）                   LLM 判等价（3 与 6/2、\\frac 与 / 这类规则挡不住）
    自由文本（financebench）                LLM 判，金额/口径表述差异太多
    代码题（humaneval / lcb / mbpp）        必须真跑测试用例，本文件不判，见 --help

用法：
    python eval/judge.py --tag baseline --dataset all --fake
    python eval/judge.py --tag baseline --dataset math500 --judge-model glm-4.5-air
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_infer import chat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RULE_NUMERIC = {"gsm8k", "asdiv", "finqa", "aime"}
# Every single-best-answer MCQ set belongs here.  medqa_usmle and mmlu_pro_law
# used to fall through to the paid LLM judge even though their gold is a bare
# option letter.
RULE_MCQ = {"arc_challenge", "openbookqa", "medmcqa", "medqa_usmle",
            "lexam_law", "mmlu_pro_law"}
LLM_ONLY = {"math500", "financebench"}
NEED_EXEC = {"humaneval", "livecodebench", "mbpp"}

JUDGE_PROMPT = """你是严格的判分员。判断"待判答案"是否正确回答了题目。

题目：
{question}

标准答案：
{gold}

待判答案：
{answer}

两边都可能带有解释或推导，只比较最终结论是否一致：
结论相同即为正确，不要求表述、格式、单位写法、LaTeX 写法、精度一致；
标准答案里的解释性内容不要求待判答案也复述出来。
只输出一个 JSON：{{"correct": true}} 或 {{"correct": false}}"""


class RateLimiter:
    """按 RPM 和 TPM 两个维度限流，取更紧的那个。滑动窗口，够用且不引依赖。"""

    def __init__(self, rpm, tpm):
        self.rpm, self.tpm = rpm, tpm
        self.reqs, self.toks = [], []
        self.lock = threading.Lock()

    def acquire(self, est_tokens):
        while True:
            with self.lock:
                now = time.time()
                self.reqs = [t for t in self.reqs if now - t < 60]
                self.toks = [(t, n) for t, n in self.toks if now - t < 60]
                used = sum(n for _, n in self.toks)
                if len(self.reqs) < self.rpm and used + est_tokens <= self.tpm:
                    self.reqs.append(now)
                    self.toks.append((now, est_tokens))
                    return
                wait = 60 - (now - min(self.reqs[0], self.toks[0][0])) if (self.reqs or self.toks) else 1
            time.sleep(max(0.05, min(wait, 5)))


LIMITER = None


def chat_retry(args, prompt, est_tokens, tries=5):
    """判分调用：限流 + 指数退避重试。429 和 5xx 都重试。"""
    last = None
    for i in range(tries):
        if LIMITER:
            LIMITER.acquire(est_tokens)
        try:
            return chat(args.judge_base_url, args.judge_model, prompt, args.judge_max_tokens,
                        0.0, args.timeout, args.api_key)
        except Exception as e:
            last = e
            code = getattr(e, "code", None)
            if code is not None and code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(min(2 ** i, 30))
    raise last


def strip_think(text):
    """去掉 GLM 的思考段。<think> 没闭合说明推理还没写完就被 max_tokens 截断，
    这种情况正文是空的，返回空串让上层判成未答。"""
    if not text:
        return text
    if "</think>" in text:
        return text.split("</think>")[-1]
    if "<think>" in text:
        return ""
    return text


def extract_tagged(text):
    """取最后一个 'Answer:' 后面的内容，没有就退回最后一行非空文本。"""
    text = strip_think(text)
    if not text:
        return None
    hits = re.findall(r"(?:Answer|答案)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if hits:
        return hits[-1].strip().rstrip(".。").strip("*$ ")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None


def to_number(s):
    """把 '$1,577.00'、'9 (apples)'、'12%' 之类归一成 float，失败返回 None。"""
    if s is None:
        return None
    s = re.sub(r"\([^)]*\)", "", str(s))          # 去掉括号里的单位说明
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def judge_numeric(rec):
    got, want = to_number(extract_tagged(rec["answer"])), to_number(rec["gold"])
    if got is None or want is None:
        return None, "数值抽取失败"
    tol = max(abs(want) * 1e-4, 1e-6)
    return abs(got - want) <= tol, f"{got} vs {want}"


def judge_mcq(rec):
    labels = rec.get("meta", {}).get("labels") or ["A", "B", "C", "D", "E"]
    tagged = extract_tagged(rec["answer"]) or ""
    m = re.search(r"\b([" + "".join(labels) + r"])\b", tagged.upper())
    if not m:
        return None, "选项抽取失败"
    return m.group(1) == str(rec["gold"]).strip().upper(), f"{m.group(1)} vs {rec['gold']}"


def judge_llm(rec, args):
    if getattr(args, "no_llm", False):
        return None, "禁用了 LLM 判分（--no-llm）"
    # 送题目 + gold + 去掉 <think> 后的完整正文。
    # 题目必须给：像 FinanceBench 这类题，gold 是"结论 + 论据"的整句而模型只答 Yes，
    # 不给题目判分模型会判成不等价（实测掉 24 个点）。正文不做抽取，因为 gold 和
    # 模型回答两边都可能带解释，交给判分模型比正则可靠。思考过程仍然不送。
    # 每条都是一次独立请求，不拼批。
    prompt = JUDGE_PROMPT.format(question=rec["question"], gold=rec["gold"],
                                 answer=strip_think(rec["answer"] or "").strip())
    est = (len(prompt) + args.judge_max_tokens * 3) // 3  # 粗估 token，中文约 1.5 字/token
    if args.fake:
        # 桩判分：只做归一化字符串/数值比对，用来验证链路而不是用来出数
        a, b = extract_tagged(rec["answer"]), str(rec["gold"])
        na, nb = to_number(a), to_number(b)
        ok = (na == nb) if (na is not None and nb is not None) else \
             (re.sub(r"\s+", "", str(a or "")) == re.sub(r"\s+", "", b))
        return ok, "fake-judge"
    # 判分模型偶尔不吐 JSON（它自己也会思考，可能把 JSON 写丢）。
    # 这属于判分调用失败而不是模型答错，所以重试到能解析为止。
    last = ""
    for _ in range(args.parse_retries):
        out = chat_retry(args, prompt, est)
        m = re.search(r'"correct"\s*:\s*(true|false)', out, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true", "llm"
        last = out
    return None, f"判分输出连续 {args.parse_retries} 次无法解析: {last[:80]}"


def score_one(rec, name, args):
    """返回 (correct: bool|None, method: str, note: str)。None 表示未判（跳过/失败）。"""
    if rec.get("error") or rec.get("answer") is None:
        return None, "skip", "推理阶段就失败了: " + str(rec.get("error"))[:60]
    if extract_tagged(rec["answer"]) is None:
        why = "思考段未闭合就被截断" if rec.get("finish_reason") == "length" else "回答里抽不到内容"
        return None, "no_answer", why
    if name in NEED_EXEC:
        return None, "need_exec", "代码题需真跑测试用例，未实现"
    if name in RULE_NUMERIC:
        ok, note = judge_numeric(rec)
        if ok is not None:
            return ok, "rule_numeric", note
        ok, note = judge_llm(rec, args)          # 规则抽不出来才花钱问 LLM
        return ok, "rule_failed" if args.no_llm else "llm_fallback", note
    if name in RULE_MCQ:
        ok, note = judge_mcq(rec)
        if ok is not None:
            return ok, "rule_mcq", note
        ok, note = judge_llm(rec, args)
        return ok, "rule_failed" if args.no_llm else "llm_fallback", note
    ok, note = judge_llm(rec, args)
    return ok, "llm", note


def ans_hash(rec):
    """回答内容的指纹。回答变了（比如截断样本重跑过）就让缓存失效。"""
    return hashlib.sha1((rec.get("answer") or "").encode("utf-8")).hexdigest()[:12]


def load_judge_cache(path):
    """读上一次的判分结果，只有 LLM 判出来的才值得缓存（规则判分不花钱）。"""
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("correct") is not None and r.get("method") in ("llm", "llm_fallback") \
                    and r.get("ans_hash"):
                cache[(r["idx"], r["ans_hash"])] = r
    return cache


def write_summary(tag, rows):
    """合并写入 runs/<tag>/summary.json。只更新本次判的数据集，不覆盖其他记录。"""
    path = os.path.join(ROOT, "runs", tag, "summary.json")
    old = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            old = {r["dataset"]: r for r in data} if isinstance(data, list) else dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            old = {}
    for r in rows:
        old[r["dataset"]] = r
    rows = [old[k] for k in sorted(old)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    write_summary_md(tag, rows)
    return path


def write_summary_md(tag, rows):
    """同时出一份人看的 summary.md，内容与 summary.json 一致。"""
    path = os.path.join(ROOT, "runs", tag, "summary.md")
    lines = [f"# 评测汇总 tag={tag}", "",
             f"生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "准确率的分母是**判出的条数**，不是总数。未计入分母的条目及原因列在最后一列。", "", "```"]
    for r in rows:
        acc = "  -   " if r["acc"] is None else f"{r['acc'] * 100:5.1f}%"
        un = r.get("unjudged_excluded_from_denominator") or {}
        un_s = " ".join(f"{k}={v}" for k, v in sorted(un.items())) or "无"
        lines.append(f"{r['dataset']:<15} 判 {r['judged']:>5}/{r['total']:<5} "
                     f"正确 {r['correct']:>5}   {acc}   未计入分母 {un_s}")
    lines += ["```", "", "## 判分方式明细", "", "```"]
    for r in rows:
        m = " ".join(f"{k}={v}" for k, v in sorted(r["methods"].items()))
        lines.append(f"{r['dataset']:<15} {m}")
    lines += ["```", "",
              "各 method 的含义见 `eval/EVAL.md`。", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def score_dataset(name, args):
    src = os.path.join(ROOT, "runs", args.tag, "output", name + ".jsonl")
    if not os.path.exists(src):
        print(f"[{name}] 没有推理结果，跳过（{src} 不存在）")
        return None
    # 同一 idx 可能有多行：失败后重跑、或截断后重跑都会追加新记录。
    # 保留最优的那条——先要没报错的，再要没被截断的，同等条件下取最后写入的。
    best = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rank = (r.get("error") is None, r.get("finish_reason") != "length")
            cur = best.get(r["idx"])
            if cur is None or rank >= cur[0]:
                best[r["idx"]] = (rank, r)
    recs = [v[1] for v in best.values()]
    out_path = os.path.join(ROOT, "runs", args.tag, "score", name + ".scored.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cache = {} if args.rejudge else load_judge_cache(out_path)
    lock = threading.Lock()
    scored = []
    reused = [0]

    def work(rec):
        h = ans_hash(rec)
        hit = cache.get((rec["idx"], h))
        if hit is not None:
            with lock:
                scored.append(hit)
                reused[0] += 1
            return
        try:
            ok, method, note = score_one(rec, name, args)
        except Exception as e:
            ok, method, note = None, "judge_error", f"{type(e).__name__}: {e}"
        row = {k: rec[k] for k in ("idx", "gold", "finish_reason", "completion_tokens")
               if k in rec}
        row.update(answer=rec.get("answer"), correct=ok, method=method, note=note, ans_hash=h)
        with lock:
            scored.append(row)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, recs))

    if reused[0]:
        print(f"  [{name}] 复用缓存判分 {reused[0]} 条，省下这些 LLM 调用（--rejudge 可强制重判）")
    out = out_path
    with open(out, "w", encoding="utf-8") as f:
        for r in sorted(scored, key=lambda x: str(x["idx"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    judged = [r for r in scored if r["correct"] is not None]
    acc = sum(r["correct"] for r in judged) / len(judged) if judged else None
    methods, unjudged = {}, {}
    for r in scored:
        methods[r["method"]] = methods.get(r["method"], 0) + 1
        if r["correct"] is None:
            unjudged[r["method"]] = unjudged.get(r["method"], 0) + 1
    # unjudged 里各项都不计入分母，含义：
    #   no_answer  思考段未闭合就被 max_tokens 截断，没答完
    #   rule_failed 规则抽不出答案且禁用了 LLM 判分
    #   skip / judge_error  推理阶段失败 / 判分调用失败
    return dict(dataset=name, total=len(scored), judged=len(judged),
                correct=sum(r["correct"] for r in judged), acc=acc,
                unjudged_excluded_from_denominator=unjudged, methods=methods)


def main():
    p = argparse.ArgumentParser(
        description="判分。代码类数据集（humaneval/livecodebench/mbpp）不在此判——"
                    "它们要在沙箱里真跑测试用例，混进 LLM 判分会得到虚高的数字。")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--dataset", default="all")
    p.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", ""),
                   help="判分模型的地址，默认读环境变量 JUDGE_BASE_URL")
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""),
                   help="判分模型名，默认读环境变量 JUDGE_MODEL")
    p.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    p.add_argument("--judge-max-tokens", type=int, default=512,
                   help="判分模型也会思考，给太小会拿不到 JSON")
    p.add_argument("--rpm", type=int, default=15000, help="判分 API 的 RPM 上限")
    p.add_argument("--tpm", type=int, default=1200000, help="判分 API 的 TPM 上限")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--parse-retries", type=int, default=5,
                   help="判分模型没吐出可解析 JSON 时重试几次")
    p.add_argument("--rejudge", action="store_true",
                   help="忽略已有判分结果，全部重判（改了判分 prompt 时用）")
    p.add_argument("--no-llm", action="store_true",
                   help="完全不调 LLM 判分。规则判不了的记为未判，不计入分母")
    p.add_argument("--fake", action="store_true", help="不联网，用桩判分跑通流程")
    args = p.parse_args()

    global LIMITER
    if not args.no_llm:
        if not args.judge_base_url or not args.judge_model:
            p.error("没有判分模型配置。source ~/.bashrc 让 JUDGE_BASE_URL / JUDGE_MODEL 生效，"
                    "或者显式传 --judge-base-url / --judge-model，或者加 --no-llm 只用规则判")
        LIMITER = RateLimiter(args.rpm, args.tpm)
        print(f"判分模型 {args.judge_model} @ {args.judge_base_url} "
              f"限流 RPM={args.rpm} TPM={args.tpm} 并发={args.workers}")

    import data_loaders as D
    groups = {"all": D.EVAL_SETS + D.CALIB_SETS, "eval": D.EVAL_SETS, "calib": D.CALIB_SETS}
    names = groups.get(args.dataset) or [s.strip() for s in args.dataset.split(",")]

    rows = [r for r in (score_dataset(n, args) for n in names) if r]
    print(f"\n{'数据集':<16}{'判分/总数':>12}{'正确':>8}{'准确率':>10}   判分方式")
    for r in rows:
        acc = "-" if r["acc"] is None else f"{r['acc'] * 100:.1f}%"
        m = " ".join(f"{k}={v}" for k, v in sorted(r["methods"].items()))
        print(f"{r['dataset']:<16}{r['judged']:>6}/{r['total']:<5}{r['correct']:>8}{acc:>10}   {m}")
    print(f"\n汇总写入 {write_summary(args.tag, rows)}")


if __name__ == "__main__":
    main()
