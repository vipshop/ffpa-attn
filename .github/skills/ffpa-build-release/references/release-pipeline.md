# 发布链路内部实现参考

代码路径均相对 ffpa-attn 仓库根目录。行为以当前工作树（`dev` 分支）为准。

## 1. 工具链总览

| 工具 | 角色 | 关键行为 |
|---|---|---|
| `tools/build_releases.sh` | 发布入口 | 解析 profile 与版本 → 逐 env 准备 workspace → 调 `build_fast.sh` 出 wheel → 归一产物（CUDA 走 `fix_release_wheel.py`，pure 直接拷进 dist） |
| `tools/build_fast.sh` | 编译开关解析 + 加速 | flag → `FFPA_*`/`ENABLE_FFPA_*` 环境变量；ccache（影子 `CUDA_HOME` 只替换 `bin/nvcc`）；`MAX_JOBS` 默认 `min(nproc,32)`；`FFPA_NVCC_THREADS` 默认 4；首个透传参数是 `wheel` 时走 `python -m pip wheel`（非 inplace，扩展落 `build/lib.*`，源码树不被污染） |
| `tools/release_workspace.py` | 出树 workspace | 改写 `pyproject.toml` 的 `name` 与 extras 自引用；拷贝 `setup.py env.py src/ csrc/ LICENSE MANIFEST.in README.md setup.cfg`；软链 `.git third_party tools/build_fast.sh tools/nvcc`；排除 `*.so`、`csrc/cuffpa/generated/`、`_version.py`、`__pycache__`、`*.egg-info`；可选写 `_version.py`；`--incremental` 保留 `build/` |
| `tools/fix_release_wheel.py` | 产物归一 | `patchelf --set-rpath '$ORIGIN'`；平台标签按 `objdump -T` 扫到的最高 `GLIBC_x.y` 定 `manylinux_2_<y>_x86_64`；重算 RECORD；重打包用固定 1980-01-01 时间戳（同一输入可复现，索引里的 sha256 才有意义）；拒绝无 `.so` 的 wheel |
| `tools/make_index.py` | 索引生成 | PEP 503 三层页面（通道根 → 项目页 → landing）+ `.nojekyll`；merge 语义（以文件名为键覆盖，历史版本保留）；锚点 = Release 资产 URL + `#sha256=`；`data-requires-python` 取自 wheel METADATA |
| `tools/publish_index.sh` | 发布编排 | `gh release create`（`--latest=false`，无 `--title`）→ 资产上传（幂等：size + 索引 sha256 都一致才跳过）→ `make_index.py` → `git add/commit/push` 站点仓库 |

`build.sh`（仓库根）只是 `tools/build_fast.sh` 的转发壳，源码安装用它；发布走 `build_releases.sh`。

## 2. `FFPA_RELEASE_*` 环境变量（build_releases.sh）

| 变量 | 默认 | 说明 |
|---|---|---|
| `FFPA_RELEASE_PROFILE` | `mini` | `mini` / `full` / `pure` |
| `FFPA_RELEASE_ENVS` | `py312` | 空格分隔的 conda env；`pure` 只取第一个 |
| `FFPA_RELEASE_VERSION` | 空（scm 推导） | full / pure 必给；mini 禁止给 |
| `FFPA_RELEASE_DIST` | `dist/` | 产物落点 |
| `FFPA_RELEASE_WORK_ROOT` | `.tmp/release_build` | workspace 根，逐 env 一个 |
| `FFPA_RELEASE_ARCH` | `89,120f` | → `FFPA_BUILD_ARCH` |
| `FFPA_RELEASE_EXT` | `all`（pure 固定 `none`） | → `--ext` |
| `FFPA_RELEASE_HEADDIM` | mini `128` / full `default` | → `--headdim` |
| `FFPA_RELEASE_JOBS` | build_fast 默认 | → `MAX_JOBS` |
| `FFPA_RELEASE_NAME` | `ffpa-attn` | 发行名 |
| `FFPA_RELEASE_CHANNEL` | `cu13` | 索引目录 + local 版本后缀 |
| `FFPA_RELEASE_DRY_RUN` | `0` | `1` 只解析并打印解析后的 flags |
| `FFPA_RELEASE_INCREMENTAL` | `0` | `1` 复用已有 workspace 的 `build/`，只重链接重打包 |
| `FFPA_RELEASE_ALLOW_UNCOMMITTED` | `0` | `1` 允许脏树构建（告警） |
| `FFPA_RELEASE_PLAT_TAG` | `linux_x86_64` | 构建期 `FFPA_PLAT_TAG`；最终标签由 `fix_release_wheel.py` 重写 |

## 3. `build_fast.sh` 的 flag → 环境变量

| flag | 环境变量 | 说明 |
|---|---|---|
| `--arch <spec>` | `FFPA_BUILD_ARCH` | sm 列表，可重复；`89,120f`，别名 `ada`/`blackwell_geforce` |
| `--ext <all\|none\|csv>` | `ENABLE_FFPA_CUDA_IMPL` / `ENABLE_FFPA_CUTE_EXT` / `ENABLE_FFPA_TMA_EXT` | `all` = cuda+cute+tma；`cute`/`tma` 隐含 `cuda`；所有目标 arch<90 时 TMA 自动关并改开 `ENABLE_FFPA_FP8_SM89_EXT` |
| `--headdim <list\|all\|default>` | `FFPA_DEV_HEADDIMS` | `default` = 64,128,192,256,320,512；`all` 多一个 768；其它值（如 832/1024/32）必须显式列 |
| `--stages <csv\|all>` | `FFPA_BUILD_STAGES` | 默认 `2,3` |
| `--debug-ext <csv\|all\|none>` | `ENABLE_FFPA_BUILD_DEBUG` | 默认 `none`，剥离 debug getenv 分支与 FORCE_KERNEL 双实例化 |
| `--input-dtypes <csv>` | `ENABLE_FFPA_CUDA_INPUT_FP16` | bf16 恒编；需要 fp16 输入 TU 才加 |
| `--mask-dtypes <csv>` | `ENABLE_FFPA_CUDA_MASK_FP32` | fp16/bf16 mask 恒编；fp32 mask 才加 |
| `-j/--jobs N` | `MAX_JOBS` | 默认 `min(nproc,32)` |
| `--shm` | `FFPA_BUILD_IN_SHM` | `build/` 放 tmpfs |
| `--clean` | `FFPA_CLEAN` | 先清 `build/ dist/ *.so` 与生成的 TU |
| `--ptxas-verbose` | `FFPA_PTXAS_VERBOSE` | 打印寄存器/smem 用量 |
| `--editable` / `--no-editable` | `FFPA_EDITABLE` | 发布用 `--no-editable` |
| `--nvcc-threads N` | `FFPA_NVCC_THREADS` | TU 内并行，默认 4 |

## 4. 守卫清单（照抄自脚本，改动前先看这里）

`build_releases.sh`：

- 每个 env：Python 版本与 `py<M><m>` 名匹配；能 `import packaging, setuptools_scm, torch, wheel`。
- 非 pure：`$CUDA_HOME/bin/nvcc` 存在且 major ≥ 13。
- full / pure：shipped inputs（`src csrc setup.py env.py pyproject.toml setup.cfg MANIFEST.in README.md LICENSE`）无未提交改动；版本不含 `.dev`；版本形如 `x.y.z`。
- pure：版本不得带 local 段；`FFPA_RELEASE_EXT` 只能是 `none`。
- mini：拒绝 `FFPA_RELEASE_VERSION`（版本必须由 scm 推导，天然排在任何正式版本之下）。
- 每个 env 断言 raw 目录里「恰好一个」该 CPython tag 的 wheel，且文件名含目标版本；发布目录同样断言唯一。
- 重发同 (版本, CPython) 前会先删掉 dist 里的旧 wheel，避免旧产物顶替断言。

`publish_index.sh`：

- 版本从 wheel 文件名读出，`--tag` 必须等于 `v<版本>+<通道>`；一次发布只允许一个版本。
- 拒收 `.dev` 版本与缺 `+<通道>` 后缀的 wheel。
- 已存在的同名资产：size 与「索引页记录的 sha256」都一致才跳过；否则要求 `FFPA_FORCE_UPLOAD=1` 才 `--clobber` 替换。
- `--site-repo` 必须是 git work tree 的根（`--index-only` 例外，可对任意目录生成静态树）。
- `--index-only` 必须显式给 `--release-url`（不给就直接 fail）。

## 5. 三个 profile 对照

| profile | 编译 flags | 版本 | 产物 | 用途 |
|---|---|---|---|---|
| `mini` | `--arch 89,120f --ext all --headdim 128` | scm 推导 `0.2.x.devN` | 默认单 env，约 21 MiB | 流水线验证；禁止发布 |
| `full` | `--arch 89,120f --ext all --headdim default` | 必给 V，自动补 `+<通道>` | 5 个 cp310–cp314 wheel，各约 80 MiB | 发布 CUDA 通道 |
| `pure` | `--ext none` | 必给 V，且不带 local 段 | 1 个 `py3-none-any`，约 0.5 MiB | 发布到 PyPI |

## 6. 发布 CUDA wheel 的内容矩阵

编译集 = `--arch 89,120f --ext all --headdim default`：

| family | sm_89（Ada） | sm_120f（Blackwell） | 编译 headdim |
|---|---|---|---|
| fp16 / bf16 | cp.async | TMA | 64,128,192,256,320,512 |
| FP8 persist-D | cp.async | TMA | 64,128,192 |
| NVFP4 persist-D | —（运行时拒绝） | TMA | 64,128,192,256 |
| native（cp.async / TMA） | ✓ | ✓ | 全 6 档，另 `D % 8 == 0` 的 [8,1024] 走 pad |

- 量化家族在默认构建里是 **persist-D only**：越出窗口会报 `rebuild with ENABLE_FFPA_QUANT_SPLIT_D=1`（`ENABLE_FFPA_QUANT_SPLIT_D` 默认 0，发布包未开）。
- Ada 上 fp8 可用（反直觉但实测）：路由放行条件是 `ENABLE_FFPA_TMA_EXT || ENABLE_FFPA_FP8_SM89_EXT`，发布包带 TMA_EXT，`major==8 && minor>=9` 即放行，dispatch 选 cp.async 的 sm_89 kernel。
- 体积：CUDA wheel 约 80 MiB（内含未压缩 `.so` 约 389 MB，540 cubin = sm_89×270 + sm_120×270）；mini 约 21 MiB；pure 约 0.5 MiB。
- 精度告警：`0.2.4` 及更早的 wheel 里 FP8 `D=192` 有 shared-memory reuse race（影响 `D>128` 的整个窗口），`0.2.5` 起修复。

## 7. 版本推导

- 不传 `FFPA_RELEASE_VERSION` 时用 setuptools-scm：`version_scheme='python-simplified-semver'`、`local_scheme='no-local-version'`（scm 推导出的 local/dirty 段会被剥掉，所以「用版本字符串判脏」不可靠，脚本改为直接查 `git status --porcelain`）。
- 构建时以通用 `SETUPTOOLS_SCM_PRETEND_VERSION` 把版本钉进 workspace 的 `_version.py`；脚本另设的 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFPA_ATTN_CU13` 是早前发行名（`ffpa-attn-cu13`）的遗留，对现在的同名发行不生效。
- CUDA 通道的 tag 由 `publish_index.sh` 的 `gh release create` 自动创建，形如 `v0.2.5+cu13`；普通 release `v0.2.5` 由第 5 步单独创建。
