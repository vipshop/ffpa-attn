---
name: ffpa-build-release
description: "ffpa-attn 发布全流程（给定一个版本号即可执行完）：pure profile 构建 PyPI 纯 Python wheel、full profile 构建带 CUDA 扩展的 whl（cp310–cp314，sm_89 + sm_120f）、RPATH 与 manylinux 平台标签归一、GitHub Release 资产上传、gh-pages 上的 PEP 503 静态索引发布与镜像、PyPI 上传，以及发布后验收与安装语义。Use when cutting an ffpa-attn release, building or publishing the CUDA-extension wheels under whl/cu13, regenerating the wheel index on gh-pages, uploading the pure-Python wheel to PyPI, or verifying an install from the index."
user-invocable: true
---

# ffpa-attn 发布（build & release）

一次发布 = 同一个版本号下产出两件东西：**PyPI 的纯 Python 包**与**自建索引上的 CUDA 扩展包**。
除版本号 `V`（如 `0.2.5`）外无需其它输入；通道 `CHANNEL` 默认 `cu13`，除新增 CUDA toolkit 外不要改。
下文所有路径相对 **ffpa-attn 仓库根目录**。

## 产物清单（以 `${V}` 为例）

| 产物 | 名称 | 去向 |
|---|---|---|
| CUDA wheel ×5 | `ffpa_attn-${V}+cu13-cp3{10..14}-cp3{10..14}-manylinux_2_38_x86_64.whl`（各 ~80 MB） | Release `v${V}+cu13` 资产 + gh-pages 索引 `whl/cu13/` |
| 纯 Python wheel ×1 | `ffpa_attn-${V}-py3-none-any.whl`（~0.5 MB） | PyPI + Release `v${V}` 资产 |
| 索引页 | `whl/cu13/ffpa-attn/index.html`（PEP 503，每个锚点带 sha256） | 项目 `gh-pages` 分支根 → `vipshop.github.io/ffpa-attn/` |

安装口径（发布后用户侧）：

```bash
pip install -U ffpa-attn                                              # 默认：PyPI 纯 Python 包
pip install ffpa-attn --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/ \
  --extra-index-url https://pypi.org/simple                           # CUDA 扩展包，如 0.2.4+cu13
```

## 执行顺序（勿跳步）

### 0. 预检

```bash
git status --porcelain -- src csrc setup.py env.py pyproject.toml setup.cfg MANIFEST.in README.md LICENSE
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version | tail -2     # 需 CUDA 13.x
for e in py310 py311 py312 py313 py314; do \
  conda run -n "$e" python -c \
  "import sys, packaging, setuptools_scm, torch, wheel; print(sys.version.split()[0], torch.__version__)"; done
command -v ccache patchelf objdump gh twine
gh auth status
gh release view "v${V}+${CHANNEL}" --repo vipshop/ffpa-attn      # 期望 not found
```

- 第一条输出必须为空：full 档只从已提交源构建。确有需要可 `FFPA_RELEASE_ALLOW_UNCOMMITTED=1` 绕过（会告警，产物不对应任何 commit）。
- PyPI 版本不可覆盖：确认 `ffpa-attn` 上没有 V（`curl -s https://pypi.org/pypi/ffpa-attn/json`）。
- py310–py314 提供发布用的 CPython 集合，缺哪个就少一个 wheel；发布集合以实际 env 为准。

### 1. dry-run（可选，首次发布或改过发布脚本时）

```bash
FFPA_RELEASE_PROFILE=full FFPA_RELEASE_VERSION=$V FFPA_RELEASE_DRY_RUN=1 bash tools/build_releases.sh
```

### 2. 构建 CUDA 扩展 wheel（full 档）

```bash
FFPA_RELEASE_PROFILE=full FFPA_RELEASE_VERSION=$V FFPA_RELEASE_CHANNEL=cu13 \
FFPA_RELEASE_ENVS="py310 py311 py312 py313 py314" \
bash tools/build_releases.sh
```

- 发布编译集由脚本默认固定为 `--arch 89,120f --ext all --headdim default`（= 64,128,192,256,320,512），不要改：用户定档「发包只含 sm_89 与 sm_120f」。
- 只改版本/元数据的重发：追加 `FFPA_RELEASE_INCREMENTAL=1`（复用 `.tmp/release_build/<env>/workspace` 的已编译对象，分钟级降到秒级）。
- 只做回归验证、不碰已发布件：追加 `FFPA_RELEASE_DIST=.tmp/<task>/dist`。
- 编译耗时较长（ccache 无命中时更久）：发出命令后等它结束，不要轮询。

### 3. 构建纯 Python wheel（pure 档）

```bash
FFPA_RELEASE_PROFILE=pure FFPA_RELEASE_VERSION=$V bash tools/build_releases.sh
```

- 不编译任何扩展（`--ext none`）、版本不带 `+cu13`、只用第一个 env（py3-none-any 与 Python 版本无关）、不校验 nvcc/ccache。
- 脚本会解包断言产物里没有 `.so`，并把 wheel 拷进 `dist/`。

### 4. 发布 CUDA 通道（Release 资产 + gh-pages 索引）

```bash
git fetch origin gh-pages
git worktree add .tmp/gh-pages gh-pages     # 已存在则跳过；存在时先 git -C .tmp/gh-pages pull --rebase
bash tools/publish_index.sh --tag "v${V}+${CHANNEL}" \
  --wheels "dist/ffpa_attn-${V}+${CHANNEL}-"*.whl --site-repo .tmp/gh-pages
```

一次调用做完四件事：`gh release create v${V}+${CHANNEL}`（`--latest=false`、不传 `--title`、notes 写安装命令）→ 上传全部 wheel 资产 → 按 merge 语义重生成索引（历史版本保留）→ commit 并 push `gh-pages`。

- tag 规范 `v<版本>+<通道>`，与 wheel 的 local 版本同构；tag 与 wheel 版本不一致会直接报错。
- 同名资产已存在且内容不同：脚本拒绝，需 `FFPA_FORCE_UPLOAD=1` 显式替换（或换版本号）。
- push 被拒时按打印的提示处理：`git -C .tmp/gh-pages pull --rebase && git -C .tmp/gh-pages push`。

### 5. 上传纯包（PyPI + 普通 Release）

```bash
python -m twine upload "dist/ffpa_attn-${V}-py3-none-any.whl"
gh release create "v${V}" --repo vipshop/ffpa-attn --target <发布提交> \
  --notes-file <notes.md> "dist/ffpa_attn-${V}-py3-none-any.whl"
```

- release notes 用 `release-notes-generator` skill 从 git 历史生成，与历史 release 的 Summary 形态一致。
- `--target` 不传时 gh 用默认分支（main）打 tag；发布提交不在 main 上时必须显式给。
- twine 403 表示 token 没有该项目权限（历史上由其它账号上传，需要 owner 加协作者）。
- 纯包与 CUDA 通道最好来自**同一个发布提交**。历史现状是补发形态：`v0.2.4+cu13` 指向 `61f2ff3`、`v0.2.4` 指向 `c545235`，两者不是同一修订。

### 6. 发布后验收（必做）

```bash
mkdir -p .tmp/ffpa-build-release
python3 -m http.server -d .tmp/gh-pages 8765 &          # 本地服务索引工作树
pip download "ffpa-attn==${V}+${CHANNEL}" --no-deps \
  -d .tmp/ffpa-build-release --index-url http://127.0.0.1:8765/whl/${CHANNEL}/
sha256sum "dist/ffpa_attn-${V}+${CHANNEL}-"*.whl .tmp/ffpa-build-release/*.whl   # 应逐一对齐
```

- 本机 `github.com:443` 被阻断（wheel 资产从 `github.com/.../releases/download/...` 下载），`pip install --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/` 会在下载资产时卡住；因此线上验收用「本地索引跑通（sha256 与 dist 一致）+ 核对线上页面」两步代替：

```bash
curl -s "https://vipshop.github.io/ffpa-attn/whl/${CHANNEL}/ffpa-attn/" | grep -o 'ffpa_attn[^"<]*whl'
```

- 扩展能力自检（在装了 torch 的环境里跑；纯包三个 flag 全 False，CUDA 包应为 True）：

```bash
python -c "import ffpa_attn; from ffpa_attn.cuda import \
CUDA_FWD_AVAILABLE, CUDA_TMA_AVAILABLE, CUDA_CUTE_TMA_AVAILABLE; \
print(ffpa_attn.__version__, CUDA_FWD_AVAILABLE, CUDA_TMA_AVAILABLE, CUDA_CUTE_TMA_AVAILABLE)"
```

- bench smoke（ffpa-attn 默认 7 号卡；`--D` 必须取已编译 headdim，精度列要 `--verbose`）：

```bash
CUDA_VISIBLE_DEVICES=7 python -m ffpa_attn.bench --backend cuda --no-bwd --cuda-impl fp16 --D 512 --verbose
```

### 7. 收尾（按需）

镜像站点（个人站点下的同一套索引，锚点仍指向 vipshop 资产）：

```bash
FFPA_SITE_URL=https://deftruth.github.io FFPA_INDEX_PREFIX=ffpa-attn \
  bash tools/publish_index.sh --index-only --tag "v${V}+${CHANNEL}" \
  --wheels "dist/ffpa_attn-${V}+${CHANNEL}-"*.whl \
  --release-url "https://github.com/vipshop/ffpa-attn/releases/download/v${V}+${CHANNEL}" \
  --site-repo <DefTruth.github.io checkout>
```

`--index-only` 必须显式给 `--release-url`（`docs/user_guide/install.md` 的镜像示例漏了它，照抄会直接 fail）。

- README / `docs/index.md` 的安装版本口令同步到 V（`ffpa-attn==V`、`ffpa-attn==V+cu13`）；`docs/user_guide/install.md` 的「What the CUDA wheel covers」只在编译集变化时才改。
- 验收完可 `git worktree remove .tmp/gh-pages`。

## 硬约束（脚本已内置守卫，不要绕过）

| 约束 | 原因 |
|---|---|
| 发布 arch 固定 `89,120f`，不能用 `120` | CuTe fp8/fp4 需要 family 目标，`sm_120` 上 ptxas 报 `.kind::f8f6f4` not supported；family SASS 在 sm_120 设备上可运行 |
| CUDA wheel 必须带 `+cu13` local 段，纯包必须不带 | PyPI 拒收 local 版本；local 段是 pip 区分两种构建的唯一依据 |
| full 档只从已提交源构建 | 产物必须对得上某个 commit |
| mini 档（`.devN`、只编 headdim 128）永不发布 | 缺大部分 headdim，`publish_index.sh` 会拒收 |
| 文档不写 `cu14` | cu14 尚不存在（用户口径，仓库已清干净） |

## 故障速查

| 症状 | 处置 |
|---|---|
| `profile full builds from committed sources; commit these first` | 提交列出的 shipped inputs，或 `FFPA_RELEASE_ALLOW_UNCOMMITTED=1` |
| `needs a release version, got '...devN'` | 传 `FFPA_RELEASE_VERSION=$V` |
| `nvcc at ... is not CUDA 13+` | 把 `CUDA_HOME` 指到 CUDA 13.x |
| `refusing to publish a pre-release wheel` | 误用了 mini 档产物，改用 full 档重编 |
| `tag '...' does not name version '...'` | tag 必须写成 `v${V}+${CHANNEL}` |
| `already published with different contents` | `FFPA_FORCE_UPLOAD=1` 替换资产，或换版本号 |
| 线上索引看不到新版本 | GitHub Pages 部署有约 1 分钟延迟，别用 `?cb=` 误判成没发上去 |
| 误发了不该发的更高版本 | 删掉该 Release（资产）**并**删掉索引页里对应锚点（`make_index.py` 是 merge 语义，只删 release 不够），再用正确 wheel 重新生成索引 |

## 参考文件

- `references/release-pipeline.md`：发布工具链（`build_releases.sh` / `build_fast.sh` / `release_workspace.py` / `fix_release_wheel.py` / `make_index.py` / `publish_index.sh`）的职责与关键行为、`FFPA_RELEASE_*` 与 `--ext/--arch/--headdim` 环境变量全表、守卫清单、发布 wheel 的编译矩阵与体积。
- `references/install-and-index.md`：两种构建的安装语义与 pip 行为、索引与资产布局、验收配方、已发布事实与历史决策（为何不上 PyPI）。
