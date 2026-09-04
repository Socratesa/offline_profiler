# MoE Expert Dashboard 使用说明

`moe_expert_dashboard.py` 可以把 `runs` 目录中的 MoE 专家统计 NPZ 文件作为 HTML dashboard 提供出去。浏览器直接访问服务器的 `IP:端口/tag/数据集` 即可查看原有的柱状图、专家排名和距离热力图，不需要预先生成或下载 HTML 文件。

## 数据目录

服务按照下面的结构发现数据：

```text
runs/
└── <tag>/
    └── stats/
        ├── <dataset>.npz
        └── <group>/
            └── <dataset>.npz
```

例如：

```text
runs/glm_think_32k_c512_20260811/stats/math500.npz
```

对应页面路径为：

```text
/glm_think_32k_c512_20260811/math500
```

嵌套数据也可以访问。例如：

```text
runs/glm_think_32k_c512_20260811/stats/cumulative_main/01_math500.npz
```

对应：

```text
/glm_think_32k_c512_20260811/cumulative_main/01_math500
```

## 启动服务

在当前项目目录运行：

```bash
/workspace/miniconda3/envs/fd_final/bin/python \
  vis/moe_expert_dashboard.py \
  --serve \
  --root runs \
  --host 0.0.0.0 \
  --port 8765
```

`--root runs` 会扫描 `runs` 下面所有包含 `stats` 目录的 tag。也可以只提供一个 run：

```bash
/workspace/miniconda3/envs/fd_final/bin/python \\
  vis/moe_expert_dashboard.py \\
  --serve \\
  --root runs/glm_think_32k_c512_20260811 \\
  --host 0.0.0.0 \\
  --port 8765
```

参数说明：

- `--host 0.0.0.0`：监听所有网卡，允许其他机器访问。
- `--port 8765`：HTTP 服务端口，可换成未被占用的端口。
- `--root`：可以是总的 `runs` 目录，也可以是某一个具体的 run 目录。

## 浏览器访问

数据集目录页：

```text
http://<服务器IP>:8765/
```

直接打开完整 dashboard：

```text
http://<服务器IP>:8765/glm_think_32k_c512_20260811/math500
```

当前机器上的示例地址：

```text
http://10.63.64.227:8765/glm_think_32k_c512_20260811/math500
```

正常 dashboard 地址不包含 `/api`。`/api/data/...` 返回的是 JSON 数据，不是 HTML 页面。

如果其他机器无法连接，请确认：

- 服务使用 `--host 0.0.0.0` 启动。
- 服务器防火墙或安全组已经放行对应端口。
- 客户端可以访问服务器所使用的 IP 网段。

## 后台运行

需要退出终端后继续运行时，可以使用：

```bash
nohup /workspace/miniconda3/envs/fd_final/bin/python \\
  vis/moe_expert_dashboard.py \\
  --serve \\
  --root runs \\
  --host 0.0.0.0 \\
  --port 8765 \\
  > vis/moe_dashboard.log 2>&1 &
```

查看进程和日志：

```bash
pgrep -af 'moe_expert_dashboard.py --serve'
tail -f vis/moe_dashboard.log
```

## 可选接口

服务还提供以下检查和数据接口：

```text
GET /healthz
GET /api/catalog
GET /api/data/<tag>/<dataset>
```

其中 `/healthz` 用于健康检查，`/api/catalog` 返回数据集目录，`/api/data/...` 返回 dashboard 使用的 JSON 数据。

## 生成单个静态 HTML

原来的静态生成方式仍然保留：

```bash
/workspace/miniconda3/envs/fd_final/bin/python \\
  vis/moe_expert_dashboard.py \\
  --input runs/glm_think_32k_c512_20260811/stats/math500.npz \\
  --output vis/res/math500.html
```

服务模式下不需要执行这一步。
