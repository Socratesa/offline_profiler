"""Self-contained LiveCodeBench code-generation judge.

Reimplements the official LiveCodeBench evaluation semantics locally, because
the upstream ``lcb_runner`` package is not installable in this environment and
github is unreachable.  Follows the official CODE_TYPE split:

  * functional / call-based  (metadata has ``func_name``):
        each test ``input`` is split on newlines, every non-empty line is
        ``json.loads``-ed into one positional argument; the candidate code is
        exec'd, ``Solution().<fn>`` (or a top-level ``<fn>``) is called and the
        return value is compared to ``json.loads(output)``.
  * standard input  (no ``func_name``):
        the candidate runs as a script, the test ``input`` is fed on stdin and
        stdout is compared to the expected ``output`` line-by-line, with
        whitespace trimmed and per-token float tolerance (1e-6).

A problem passes only if every one of its test cases passes.  Each problem is
executed in a child subprocess with a timeout so infinite loops and crashes are
contained; this is not a hardened sandbox (we are already inside a container).

Interface kept identical to upstream so ``judge_code.py`` imports it unchanged:
    metrics, results, metadata = codegen_metrics(samples, generations,
        k_list=[1], num_process_evaluate=N, timeout=T)
where samples[i] = {"input_output": json({"inputs":[...],"outputs":[...],
"fn_name"?:str})}, generations[i] = [code_str], and results[i] = [[per-case
bool, ...]].
"""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

_CALL_DRIVER = r'''
import json, sys
code = open(sys.argv[1], encoding="utf-8").read()
data = json.load(open(sys.argv[2], encoding="utf-8"))
fn, cases = data["fn_name"], data["cases"]
ns = {"__name__": "__lcb_sol__"}
try:
    exec("from typing import *", ns)
    import collections, math, heapq, bisect, itertools, functools, re, string
    for _m in (collections, math, heapq, bisect, itertools, functools, re, string):
        ns[_m.__name__] = _m
except Exception:
    pass
def norm(x):
    if isinstance(x, tuple): x = list(x)
    if isinstance(x, list): return [norm(i) for i in x]
    return x
def eq(a, b):
    if norm(a) == norm(b): return True
    try:
        if abs(float(a) - float(b)) <= 1e-6: return True
    except Exception:
        pass
    return False
res = []
try:
    exec(code, ns)
    for args, exp in cases:
        try:
            if "Solution" in ns and isinstance(ns["Solution"], type):
                f = getattr(ns["Solution"](), fn)
            else:
                f = ns[fn]
            res.append(bool(eq(f(*args), exp)))
        except Exception:
            res.append(False)
except Exception:
    res = [False] * len(cases)
print("LCBRESULT:" + json.dumps(res))
'''


def _num(s):
    try:
        return float(s)
    except Exception:
        return None


def _tok_eq(a, b):
    if a == b:
        return True
    na, nb = _num(a), _num(b)
    return na is not None and nb is not None and abs(na - nb) <= 1e-6


def _stdout_match(got, exp):
    g = [l.rstrip() for l in got.strip("\n").split("\n")]
    e = [l.rstrip() for l in exp.strip("\n").split("\n")]
    if g == e:
        return True
    gt, et = got.split(), exp.split()
    if len(gt) == len(et) and gt and all(_tok_eq(a, b) for a, b in zip(gt, et)):
        return True
    if len(g) == len(e):
        for a, b in zip(g, e):
            at, bt = a.split(), b.split()
            if len(at) != len(bt) or not all(_tok_eq(x, y) for x, y in zip(at, bt)):
                return False
        return True
    return False


def _run_call_based(code, inputs, outputs, fn_name, timeout):
    cases = []
    for inp, out in zip(inputs, outputs):
        try:
            args = [json.loads(l) for l in str(inp).split("\n") if l.strip() != ""]
        except Exception:
            args = [inp]
        try:
            exp = json.loads(out)
        except Exception:
            exp = out
        cases.append([args, exp])
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "sol.py"); open(sp, "w", encoding="utf-8").write(code or "")
        cp = os.path.join(d, "cases.json")
        json.dump({"fn_name": fn_name, "cases": cases}, open(cp, "w", encoding="utf-8"))
        dp = os.path.join(d, "driver.py"); open(dp, "w", encoding="utf-8").write(_CALL_DRIVER)
        try:
            p = subprocess.run([sys.executable, dp, sp, cp], capture_output=True,
                               text=True, timeout=max(6, min(timeout * len(cases) + 5, 120)), cwd=d)
            for line in p.stdout.splitlines():
                if line.startswith("LCBRESULT:"):
                    return json.loads(line[len("LCBRESULT:"):])
        except Exception:
            pass
        return [False] * len(cases)


def _run_stdin(code, inputs, outputs, timeout):
    res = []
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "sol.py"); open(sp, "w", encoding="utf-8").write(code or "")
        for inp, out in zip(inputs, outputs):
            try:
                p = subprocess.run([sys.executable, sp], input=str(inp), capture_output=True,
                                   text=True, timeout=timeout, cwd=d)
                res.append(bool(_stdout_match(p.stdout, str(out))))
            except Exception:
                res.append(False)
    return res


def _evaluate(task):
    sample, gen, timeout = task
    io = json.loads(sample["input_output"])
    inputs, outputs = io.get("inputs", []), io.get("outputs", [])
    fn = io.get("fn_name")
    code = gen[0] if gen else ""
    if not code:
        return [False] * len(inputs)
    if fn:
        return _run_call_based(code, inputs, outputs, fn, timeout)
    return _run_stdin(code, inputs, outputs, timeout)


def codegen_metrics(samples, generations, k_list=(1,), num_process_evaluate=16,
                    timeout=6, **_kw):
    tasks = [(samples[i], generations[i], timeout) for i in range(len(samples))]
    results = {}
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=max(1, num_process_evaluate)) as ex:
        futs = {ex.submit(_evaluate, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                flags = fut.result()
            except Exception:
                flags = [False]
            results[i] = [flags]
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [lcb] judged {done}/{total}", flush=True)
    npass = sum(1 for i in results if results[i][0] and all(results[i][0]))
    metrics = {"pass@1": (100.0 * npass / total) if total else 0.0}
    return metrics, results, {}
