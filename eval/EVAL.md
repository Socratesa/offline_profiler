# 评测流程说明

这套脚本做两件事：让模型对数据集作答（生成），再判断答得对不对（判分）。
两件事**严格分开**，各自落盘，互不重复劳动——生成最贵，绝不因为改判分逻辑而重跑。

```
data/{eval,calib}/<ds>/          数据集原始文件（只读）
        │  run_infer.py
        ▼
runs/<tag>/<ds>.jsonl            模型原始回答         ← 最贵，改判分不动它
        │  judge_code.py（仅代码集）
        ▼
runs/<tag>/<ds>.code.jsonl       从回答里抽出的代码    ← 可反复重建，不花钱
        │  judge.py / judge_code.py
        ▼
runs/<tag>/<ds>.scored.jsonl     逐条判分结果 + 缓存指纹
runs/<tag>/summary.json / .md    汇总
```

`tag` 是一次实验的名字，比如 `baseline`、`prune_top8_skip2`。不同 tag 完全隔离，
对比实验就是对比两个 tag 的 summary。

## 一、生成阶段 run_infer.py

```bash
python eval/run_infer.py --dataset eval --tag baseline --workers 96
```

- `--dataset` 可写 `all` / `eval` / `calib`，或逗号分隔的集名
- 被测模型默认 `http://127.0.0.1:1910/v1` + `/workspace/models/GLM-4.5-Air`，
  用 `--base-url` / `--model` 覆盖
- 默认**开思考**。GLM-4.5-Air 会先输出 `<think>…</think>` 再给正文。
  `--no-think` 可关（会在请求里带 `chat_template_kwargs={"enable_thinking": false}`；
  实测 FastDeploy 只认这一种写法，顶层传 `enable_thinking` 会被静默忽略）
- `--temperature` 默认 0
- 生成预算统一 `32768`（见 `data_loaders.MAX_TOKENS`），等于服务端 `max_model_len`
  的整个窗口。服务会自动按剩余窗口收窄，不会因为 prompt 加起来超了而报错。
  **统一成一个数是硬要求**：剪枝前后两组必须用相同预算，否则截断率不同，
  准确率的差值就分不清是剪枝造成的还是预算造成的

### jsonl 的自清理与续传

每次运行前会先 `compact()` 整理目标 jsonl：按 `idx` 去重，**只保留
`error=None` 且 `finish_reason="stop"` 的记录**，报错的、被截断的、重复的一律物理删除。
被删掉的 `idx` 自动变成"未完成"，同一次运行里补跑。

所以：

- 文件里永远只有正常推理结束的结果，判分时不需要再兜底
- 中断了直接原样重跑，已完成的会跳过
- 想重跑被截断的样本，也是原样重跑即可，不需要额外开关

### 什么时候该怀疑推理服务出问题

生成阶段每条失败都记在 `error` 字段里，跑的时候也会打印 `err=` 计数。按现象区分：

| 现象 | 判断 | 处理 |
| --- | --- | --- |
| `URLError: Connection refused` | 服务没起 | 起服务，原样重跑 |
| 零星 `HTTPError: 500` | 服务端瞬时故障 | 重跑即可。实测 1172 条里出过 58 条，重试全部成功 |
| 成片 `500` 且重跑仍旧 | 服务端真有问题 | 看 FastDeploy 日志，别在客户端硬扛 |
| `TimeoutError` | 排队太久，不是模型不行 | 调大 `--timeout`（默认 3600）或降并发 |
| `/v1/models` 十几秒不响应 | 服务假死 | 见下 |

**服务假死的判据**（实际遇到过一次）：`/v1/models` 超时无响应；`nvidia-smi` 显示部分卡
100% 利用率但功耗只有 120W/500W 左右——高利用率配低功耗是 NCCL 集合通信自旋等待的特征，
不是真在算；worker 进程还活着、显存还占着；生成端一条结果都不出。
这种情况客户端无解，只能重启服务，日志在 `/workspace/fd_afd_dev/*_log/` 下。

**吞吐是否正常**：服务端日志里的 `#running-req` 和 `#queue-req` 最直接。
`#queue-req=0` 且 `#running-req` 远小于 `--workers`，说明瓶颈在客户端任务数不够，
不是服务扛不住。服务端 `max_num_seqs` 是并发天花板，实测它从 8 提到 256，
聚合吞吐从 310 tok/s 涨到 1106 tok/s。

## 二、判分阶段

### 谁在判

- **非代码集** `judge.py`，判分模型是**外部 API**（`deepseek-v4-flash-0731`），
  配置从 `~/.bashrc` 的 `JUDGE_BASE_URL` / `JUDGE_MODEL` / `JUDGE_API_KEY` 读
- **代码集** `judge_code.py`，不用任何 LLM，真跑测试用例

判分模型**必须与被测模型解耦**。之前默认指向本地 1910，等于让被测模型判自己：
一是自评偏好，二是更致命——1910 换成剪枝后的模型时，判分器也跟着变了，
量尺随被测物一起变形，测出来的差值没有意义。

### 四种判分策略

| 数据集 | 策略 | 怎么判 |
| --- | --- | --- |
| gsm8k / asdiv / aime / finqa | `rule_numeric` | 抽 `Answer:` 后的数字，归一化后比数值 |
| arc_challenge / openbookqa | `rule_mcq` | 抽选项字母，与 `answerKey` 比 |
| math500 / financebench | `llm` | 送外部判分模型 |
| humaneval / livecodebench / mbpp | `exec` | 执行测试用例 |

能用规则的就不用 LLM——规则确定、可复现、零成本，而 LLM 判分本身带噪声。
规则抽不出答案时会降级问 LLM（记为 `llm_fallback`）；加 `--no-llm` 则记为
`rule_failed` 且不计入分母。

### gold 和模型回答怎么算一致

**规则数值** `judge_numeric`：先 `strip_think()` 去掉 `<think>…</think>`，取最后一个
`Answer:` 后面的内容（没有就退回最后一行非空文本），再归一化——去掉括号里的单位说明
（`9 (apples)` → `9`）、千分位逗号、`$`、`%`，取第一个数字。与 gold 同样处理后按
**相对误差 1e-4** 比较。抽不出数字则判不了，降级或记未判。

**规则选择题** `judge_mcq`：从抽出的内容里找单个选项字母（用该题的 `labels` 限定范围），
与 `answerKey` 比。抽不到字母则判不了。

**LLM 判分** `judge_llm`：送三段——**题目原文、gold、去掉 `<think>` 的完整正文**，
要求只输出 `{"correct": true/false}`。三个设计都是踩过坑定下来的：

- **题目必须给**。只送 gold 和答案时，FinanceBench 从 86.6% 掉到 60.7%。
  原因是它的 gold 是"结论 + 论据"整句（`Yes. Paypal has a positive working capital of
  $1.6Bn...`），而模型按指令只答 `Yes`，判分模型看到长句对两个字符就判不等价。
  yes/no 类题目没有题目根本判不了
- **正文不做正则抽取**，因为 gold 和模型回答两边都可能带解释，交给判分模型比正则可靠
- **思考过程不送**。判等价只需要最终答案，送进去既费额度又容易把判分模型带偏

判分指令里明确写了"两边都可能带解释，只比最终结论；gold 里的解释性内容不要求
待判答案复述"。判分模型没吐出可解析 JSON 时重试（`--parse-retries`，默认 5 次）——
这属于判分调用失败，不是模型答错，不能算错。

**执行判分** `exec`：
- HumanEval 自己实现：抽 ```python 代码块 → 若其中没定义目标函数则退回
  `prompt + 补全` 拼法 → 接上官方 `test` 与 `check(entry_point)` → 子进程执行，
  15s 超时、cwd 为临时空目录。**已用正负对照验证**：喂数据集自带
  `canonical_solution` 判 164/164 通过，喂全返回 `None` 的假实现判 20/20 不通过
- LiveCodeBench 用**官方判分器**，`lcb_runner/evaluation/` 三个文件从上游 vendor，
  并用 `api.github.com` 的 blob SHA 逐个校验一致。stdin/functional 两种模式的判定口径、
  浮点容差、超时算法全是原版，与 leaderboard 可比。测试用例来自
  `public_test_cases`（JSON 字符串）加解开**四层套娃**（base64 → zlib → pickle → JSON 字符串）
  的 `private_test_cases`——pickle 解出来的是 str 不是 list，还要再 `json.loads` 一次，
  **全部用例通过才算这题通过**

代码执行没有容器隔离。LCB 官方 `run_test` 会先 `reliability_guard()` 屏蔽一批
破坏性调用并限 4GB 内存，HumanEval 侧是独立子进程加超时。都不是真沙箱。

## 三、积分策略

**准确率的分母是"判出的条数"，不是总数。** `correct` 为 `None` 的条目一律排除，
并在 `summary.json` 的 `unjudged_excluded_from_denominator` 里逐项列明。

| method | correct | 计不计分母 | 含义 |
| --- | --- | --- | --- |
| `rule_numeric` / `rule_mcq` / `llm` / `exec` | true/false | 计 | 正常判出 |
| `llm_fallback` | true/false | 计 | 规则抽不出，改用 LLM 判出 |
| `no_code` | false | **计，且算失败** | 正常收尾但抽不到代码块 |
| `no_answer` | None | 不计 | 思考段未闭合就被截断，没答完 |
| `truncated` | None | 不计 | 同上，代码集里的叫法 |
| `rule_failed` | None | 不计 | 规则抽不出且 `--no-llm` |
| `skip` | None | 不计 | 推理阶段就失败了 |
| `judge_error` | None | 不计 | 判分调用异常 |

被截断和答完但抽不出结果，是**两件不同的事**，口径也不同：

- 截断 = 模型没答完，不该算它错，排除出分母
- 答完了但抽不到代码 = 模型确实没给出可运行程序，pass@1 口径下就是失败

## 四、产物文件的含义

| 文件 | 内容 | 什么时候会被改写 |
| --- | --- | --- |
| `<ds>.jsonl` | 模型原始回答，每行 `idx / question / answer / gold / meta / finish_reason / prompt_tokens / completion_tokens / latency / error` | 只有 `run_infer.py` 会写。每次运行前 `compact()` 会剔掉报错和截断的行 |
| `<ds>.code.jsonl` | 从回答里抽出的代码，`idx / code / finish_reason` | `judge_code.py` 写。不存在时自动建，`--rebuild-code` 强制重建 |
| `<ds>.scored.jsonl` | 逐条判分，`idx / gold / answer / correct / method / note / ans_hash` | 每次判分覆盖写 |
| `summary.json` | 各集汇总，**合并写入**，只更新本次判的集 | 每次判分 |
| `summary.md` | 同上，人看的版本 | 每次判分 |

`ans_hash` 是模型回答的 sha1 前 12 位，用来做**判分缓存的键**。默认复用上次
LLM 判分的结果，回答变了（比如截断样本重跑过）指纹不匹配则自动重判。
规则判分不缓存，本来不花钱。

`note` 字段是判分依据的人类可读说明，比如 `18.0 vs 18.0`、`3/5 用例通过`、
`被 max_tokens 截断，未计分`。判分结果有疑问时先看这个字段。

## 五、常用命令

```bash
source ~/.bashrc            # 让 JUDGE_* 生效，判分前必做

# 生成
python eval/run_infer.py --dataset eval  --tag baseline --workers 96
python eval/run_infer.py --dataset calib --tag baseline --workers 96 --limit 256
python eval/run_infer.py --dataset livecodebench --lcb-release 1 --tag baseline
python eval/run_infer.py --dataset gsm8k --tag baseline --limit 20        # 小样试跑
python eval/run_infer.py --dataset all --fake --limit 30 --tag smoke      # 不联网跑通流程

# 中断/失败/截断后：原样重跑即可，compact 会自动决定跑哪些
python eval/run_infer.py --dataset gsm8k --tag baseline --workers 96

# 判分（非代码集）
python eval/judge.py --tag baseline --dataset eval                 # 默认复用判分缓存
python eval/judge.py --tag baseline --dataset math500 --rejudge    # 改了判分 prompt 后强制重判
python eval/judge.py --tag baseline --dataset gsm8k,arc_challenge --no-llm   # 只用规则，一分钱不花
python eval/judge.py --tag baseline --dataset eval --fake          # 判分链路自检，不联网

# 判分（代码集，纯 CPU）
python eval/judge_code.py --tag baseline --dataset humaneval,livecodebench --workers 16
python eval/judge_code.py --tag baseline --dataset humaneval --extract-only   # 只抽代码，先人工看
python eval/judge_code.py --tag baseline --dataset humaneval --rebuild-code   # 改了抽取规则后重建

# 换被测模型（剪枝后的实验）
python eval/run_infer.py --dataset eval --tag prune_v1 \
    --base-url http://127.0.0.1:1910/v1 --model /path/to/pruned
python eval/judge.py --tag prune_v1 --dataset eval
```

对比实验就是 `diff` 两个 tag 的 `summary.md`。注意两组必须用**相同的生成预算、
相同的判分器、相同的 prompt**，否则差值不可解释。

## 六、已知的噪声与坑

- **判分模型不确定**。同一批 150 条、同一 prompt、`temperature=0`，连判三次得到
  88.0% / 88.6% / 89.2%，约 1 个点的噪声（判分模型自己会思考，temp=0 不保证复现）。
  **所以差值小于 1-2 个点时不要下结论**。判分缓存的另一个作用就是把 baseline 的
  判定固定下来，剪枝后只判新数据、baseline 不重判
- **截断的样本是有偏的**——难题才会思考到收不住。把它们排除出分母会让准确率虚高。
  math500 曾出现 32/500 被截断，按判出的算 98.7%、把未判的全算错则是 92.4%，
  真值靠下界更近。报数时必须同时给出未计入分母的数量
- **模型会退化重复**。HumanEval 有 11 条烧满预算，其中包括最简单的
  `HumanEval_0`；正常样本 p50 只有 1145 token、max 14513。这种加预算没用
- **LLM 判分不适合代码**。看代码"像不像对的"会给出虚高且不可复现的数字，必须真跑用例
- `datasets` 这个模块名与 HuggingFace 的包撞名，本目录里叫 `data_loaders.py`
- LiveCodeBench 靠 loading script 分发，`datasets>=3.0` 无法 `load_dataset`，
  只能自己读 jsonl；论文里必须写明用的哪个 release（`release_vN` = `test.jsonl`
  到 `testN.jsonl` 累加）
