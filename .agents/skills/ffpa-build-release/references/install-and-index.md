# 安装与索引参考

## 1. 两种构建

| 构建 | 内容 | 来源 |
|---|---|---|
| `ffpa-attn` | 纯 Python 源码，Triton + CuTe-DSL 后端 | [PyPI](https://pypi.org/project/ffpa-attn/) |
| `ffpa-attn`（CUDA） | 同一份源码 **编译进 CUDA 扩展**（D>256，forward only） | 自建索引 `https://vipshop.github.io/ffpa-attn/whl/cu13/` |

两者**同名同包**（`ffpa_attn/`），靠 wheel 的 local 版本区分：CUDA 包是 `0.2.4+cu13`，纯包是 `0.2.4`。

## 2. 安装命令与语义

```bash
pip install -U ffpa-attn                    # 默认：PyPI 纯包，最通用（无 arch/CPython/torch/toolkit 约束）

python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 期望 ...+cu130 13.0
pip install ffpa-attn --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/ \
  --extra-index-url https://pypi.org/simple                              # CUDA 扩展包
```

- **必须带 `--extra-index-url https://pypi.org/simple`**：我们的索引只放 `ffpa-attn` 一个项目、不带依赖，缺第二个索引时 pip 会在第一个未安装的依赖上失败。
- **先装 torch**（`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`）。wheel 只针对 CUDA 13.x 编译（验证基线 `torch==2.11.0+cu130`）；若让 pip 去 PyPI 拉 torch，会拿到非 cu130 构建，装完在 `import` 时报 `libcudart.so.13` 找不到。
- 运行前提：Linux x86_64、glibc ≥ 2.38、Python 3.10–3.14（以索引页列为准）、torch cu130、GPU 属于已编译架构（sm_89 或 sm_120f）。
- pip 行为（实测）：local 版本排序高于同 base 版本，所以两个索引同时可用时我们的包胜出，且已装的纯包不再满足需求；装过 CUDA 包后 `pip install -U ffpa-attn`（不带我们的索引）**不会**降级回纯包；要退回纯包是「卸载再装」：

```bash
pip uninstall -y ffpa-attn && pip install -U ffpa-attn
```

## 3. 资产与索引布局

| 位置 | URL |
|---|---|
| PEP 503 通道根 | `https://vipshop.github.io/ffpa-attn/whl/cu13/` → 列出 `ffpa-attn` |
| 项目页（pip 真正读的） | `https://vipshop.github.io/ffpa-attn/whl/cu13/ffpa-attn/` |
| 人类 landing | `https://vipshop.github.io/ffpa-attn/` |
| wheel 资产 | `https://github.com/vipshop/ffpa-attn/releases/download/v<版本>+cu13/<wheel 文件名>#sha256=<hex>` |
| 镜像（个人站点，锚点仍指向 vipshop 资产） | `https://deftruth.github.io/ffpa-attn/whl/cu13/ffpa-attn/` |

- 索引是纯静态 HTML，放在项目 `gh-pages` 分支根（Pages: Deploy from a branch / `gh-pages` / `(root)`），靠 `.nojekyll` 关掉 Jekyll。
- 每个锚点带 `#sha256=`，pip 会强制校验：资产被替换或上传不完整会显式失败，而不是装出坏包。
- 一个 CUDA toolkit 一个目录（`whl/cu13/`）；换 toolkit 只改目录与 local 版本后缀，发行名不变。
- 索引生成是 **merge 语义**（以文件名为键）：发新版本会保留历史版本的锚点，`pip install ffpa-attn==<旧版本>` 仍可解析；同一个文件名重发则覆盖自己的锚点，不会留下两个 sha256。

## 4. 验收配方

```bash
mkdir -p .tmp/ffpa-build-release
python3 -m http.server -d .tmp/gh-pages 8765 &     # 服务本地索引工作树
pip download "ffpa-attn==${V}+cu13" --no-deps -d .tmp/ffpa-build-release \
  --index-url http://127.0.0.1:8765/whl/cu13/
sha256sum "dist/ffpa_attn-${V}+cu13-"*.whl .tmp/ffpa-build-release/*.whl
```

本地 HTTP 索引的下载会走锚点里的 sha256 校验，字节与 `dist/` 一致即等价于线上可装。
本机 `github.com:443` 被阻断（资产从 `github.com/.../releases/download/...` 下载），所以线上闭环用「本地索引 + 页面核对」：

```bash
curl -s https://vipshop.github.io/ffpa-attn/whl/cu13/ffpa-attn/ | grep -o 'ffpa_attn[^"<]*whl'
```

其余三项自检：

```bash
# 扩展能力标志（在装了 torch 的环境里 import 安装后的包；纯包三项全 False）
python -c "import ffpa_attn; from ffpa_attn.cuda import \
CUDA_FWD_AVAILABLE, CUDA_TMA_AVAILABLE, CUDA_CUTE_TMA_AVAILABLE; \
print(ffpa_attn.__version__, CUDA_FWD_AVAILABLE, CUDA_TMA_AVAILABLE, CUDA_CUTE_TMA_AVAILABLE)"

# bench smoke：ffpa-attn 默认 7 号卡；D 必须取已编译 headdim；精度列要 --verbose
CUDA_VISIBLE_DEVICES=7 python -m ffpa_attn.bench --backend cuda --no-bwd --cuda-impl fp16 --D 512 --verbose

# 量化家族（sm_120）：窗口内可跑，越窗会 raise 提示 ENABLE_FFPA_QUANT_SPLIT_D=1
CUDA_VISIBLE_DEVICES=7 python -m ffpa_attn.bench --backend cuda --no-bwd --cuda-impl fp8 --D 128 --verbose
```

## 5. 已发布事实（2026-10-14 核对，会随时间变化）

- Release `v0.2.4+cu13`：5 个资产（cp310–cp314，`manylinux_2_38_x86_64`）；Release `v0.2.4`：带 `ffpa_attn-0.2.4-py3-none-any.whl` 资产与 Summary 型 notes。
- PyPI `ffpa-attn` 最新版 `0.2.4`，只有 `ffpa_attn-0.2.4-py3-none-any.whl`（纯包）。
- 仓库 **Latest 指针**：`gh api repos/vipshop/ffpa-attn/releases/latest` 返回 `v0.2.4+cu13`，尽管脚本以 `--latest=false` 创建它。若希望普通 release（`v<版本>`）占据 Latest，发布后需在 GitHub Releases 页面手动 `Set as the latest release`。
- 两个 tag 并非同一修订：`v0.2.4+cu13` → `61f2ff3`、`v0.2.4` → `c545235`（CUDA 通道是补发形态）。后续发布建议两者同源。
- 本机索引工作树在 `.tmp/gh-pages`（`.tmp` 已被 `.gitignore` 忽略）；`.tmp/release_build/py310..py314` 是 full 档的增量 workspace。

## 6. 为何 CUDA 包不上 PyPI（决策记录）

- PyPI 拒收带 local 段（`+cu13`）的版本号，服务端直接 400。
- 若去掉后缀以干净版本上传：实测 pip 在「同名同版本、纯包与平台包并存」时优先选平台 wheel，`pip install -U ffpa-attn` 默认就会装 80 MiB 的 CUDA 包，违背「默认最通用」。
- 单文件 100 MB 上限（现在 80 MiB，加 arch/headdim 就危险）、PyPI 不可变（只能 yank）、manylinux 标签表达不了「需要 torch cu13x」——都指向独立索引。
- 历史先例：0.1.x 时代 PyPI 上本来就是同名 + 平台 wheel 与纯 wheel 共存（92 个平台 / 13 个纯）；0.2.x 起把 CUDA 线整体移到自建索引。
