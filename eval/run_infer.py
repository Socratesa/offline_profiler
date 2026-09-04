"""对数据集批量打请求，把模型回答落盘成 jsonl。

用法：
    python eval/run_infer.py --dataset gsm8k --limit 20 --no-think
    python eval/run_infer.py --dataset all --fake        # 不联网，跑通流程

输出 runs/<tag>/<dataset>.jsonl，每行含 idx / question / answer / gold / meta /
finish_reason / prompt_tokens / completion_tokens / latency / error。
同名文件已存在时按 idx 跳过已完成的样本，可以直接重跑续传。
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loaders as D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chat_full(base_url, model, prompt, max_tokens, temperature, timeout,
              api_key="", no_think=False):
    """调 OpenAI 兼容的 /v1/chat/completions，返回 text / finish_reason / token 数。"""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    # max_tokens 为 0/None 就不传，让服务用满剩余窗口。这样"截断"只可能是
    # 真的顶到了模型上下文上限，而不是我们在客户端设小了。
    if max_tokens:
        body["max_tokens"] = max_tokens
    if no_think:
        # GLM-4.5 关思考的开关。实测 FastDeploy 只认 chat_template_kwargs 这一种写法，
        # 顶层传 enable_thinking 会被静默忽略（照样输出 <think>）。
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    ch = data["choices"][0]
    u = data.get("usage") or {}
    return {"text": ch["message"]["content"], "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens")}


def chat(*a, **kw):
    return chat_full(*a, **kw)["text"]


def fake_chat(rec, wrong_rate=0.3):
    """离线桩：按 gold 造回答，并按 wrong_rate 故意答错，用来验证判分链路。"""
    if rec["gold"] is None:  # 代码题
        return "先分析题意，再给出实现。\n\n```python\ndef solve():\n    return 42\n```"
    gold = rec["gold"]
    if random.random() < wrong_rate:
        gold = "B" if len(gold) == 1 and gold.isalpha() else "-999"
    return "我们一步步来看。\n先整理已知条件，再计算。\n\nAnswer: " + gold


def compact(path):
    """就地整理 jsonl：按 idx 去重，只保留正常推理结束的记录，返回这些 idx。

    报错的和被 max_tokens 截断的会被丢掉，于是下一轮会自动重跑它们——
    这样文件里永远只有 finish_reason=stop 的干净结果。
    """
    if not os.path.exists(path):
        return set()
    keep, dropped = {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            # fake 模式没有 finish_reason，视为正常
            good = r.get("error") is None and r.get("finish_reason", "stop") == "stop"
            if good:
                keep[r["idx"]] = r
            else:
                dropped += 1
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in keep.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    if dropped:
        print(f"  整理 {os.path.basename(path)}: 剔除 {dropped} 条（报错/截断/重复），"
              f"保留 {len(keep)} 条", flush=True)
    return set(keep)


def run_dataset(name, args):
    kw = {"release": args.lcb_release} if name == "livecodebench" else {}
    recs = D.load(name, **kw)
    max_tokens = args.max_tokens  # 0 = 不限，交给服务按剩余窗口决定
    if args.limit:
        recs = recs[: args.limit]
    out_path = os.path.join(ROOT, "runs", args.tag, "output", name + ".jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    skip = compact(out_path)
    todo = [r for r in recs if r["idx"] not in skip]
    print(f"[{name}] 共 {len(recs)} 条，已完成 {len(skip)}，本次 {len(todo)} 条，"
          f"max_tokens={max_tokens or '不限'}", flush=True)
    if not todo:
        return
    lock = threading.Lock()
    stats = {"ok": 0, "err": 0, "trunc": 0}
    fout = open(out_path, "a", encoding="utf-8")

    def work(rec):
        t0 = time.time()
        err, extra = None, {}
        try:
            if args.fake:
                time.sleep(random.uniform(0.001, 0.01))
                ans = fake_chat(rec)
            else:
                # 截断就再试几次。temperature=0 下结果通常一样，但服务端 batch 组成不同
                # 会带来微小的数值差异，偶尔能跑出不同结果。试完还截断就认了。
                for attempt in range(args.truncate_retries + 1):
                    res = chat_full(args.base_url, args.model, rec["question"], max_tokens,
                                    args.temperature, args.timeout, args.api_key, args.no_think)
                    if res.get("finish_reason") != "length":
                        break
                ans = res.pop("text")
                extra = res
        except Exception as e:  # 单条失败不影响整体，重跑时会自动补
            ans, err = None, f"{type(e).__name__}: {e}"
        row = dict(rec, answer=ans, latency=round(time.time() - t0, 3), error=err, **extra)
        with lock:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            stats["err" if err else "ok"] += 1
            if extra.get("finish_reason") == "length":
                stats["trunc"] += 1
            n = stats["ok"] + stats["err"]
            if n % 50 == 0 or n == len(todo):
                print(f"  [{name}] {n}/{len(todo)}  ok={stats['ok']} err={stats['err']} "
                      f"截断={stats['trunc']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fout.close()
    if stats["trunc"]:
        print(f"  [{name}] 注意：{stats['trunc']} 条因 max_tokens 被截断，这些样本抽不到答案",
              flush=True)
    print(f"[{name}] 完成 ok={stats['ok']} err={stats['err']} -> {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all",
                   help="数据集名，或 all / eval / calib，逗号分隔多个")
    p.add_argument("--tag", default="baseline", help="本次运行的目录名，放在 runs/ 下")
    # 被测模型写死在这里，需要时用命令行覆盖
    p.add_argument("--base-url", default="http://127.0.0.1:1910/v1")
    p.add_argument("--model", default="/workspace/models/GLM-4.5-Air")
    p.add_argument("--api-key", default="")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="0（默认）= 不传 max_tokens，让服务用满剩余上下文窗口")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--limit", type=int, default=0, help="每个集只跑前 N 条，0 为全量")
    p.add_argument("--lcb-release", type=int, default=6, help="LiveCodeBench release_vN")
    p.add_argument("--truncate-retries", type=int, default=0,
                   help="生成被截断时重试几次。每次都会烧满整个窗口，调大很贵")
    p.add_argument("--no-think", action="store_true",
                   help="关掉 GLM 的思考模式。默认开启")
    p.add_argument("--fake", action="store_true", help="不联网，用桩回答跑通流程")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    groups = {"all": D.EVAL_SETS + D.CALIB_SETS, "eval": D.EVAL_SETS, "calib": D.CALIB_SETS}
    names = groups.get(args.dataset) or [s.strip() for s in args.dataset.split(",")]
    unknown = [n for n in names if n not in D.LOADERS]
    if unknown:
        p.error(f"未知数据集 {unknown}，可选：{sorted(D.LOADERS)}")

    print(f"tag={args.tag} fake={args.fake} no_think={args.no_think} model={args.model} "
          f"max_tokens={args.max_tokens} base_url={'-' if args.fake else args.base_url}", flush=True)
    for n in names:
        run_dataset(n, args)


if __name__ == "__main__":
    main()
