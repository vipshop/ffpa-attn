#!/usr/bin/env python3
"""Normalize a release wheel: RPATH, platform tag, RECORD.

Two things a wheel built in a conda environment cannot ship as-is:

* The extension carries an RPATH pointing at the build environment's lib
  directory, which is meaningless on a user's machine and is a wrong library if
  that path happens to exist there. Every ``.so`` is rewritten to ``$ORIGIN``.
  The torch libraries the extension needs are loaded by ``import torch`` before
  the extension is imported, so their SONAMES are already resolved and no RPATH
  is required to find them.
* ``setup.py`` forces a platform tag, which saw a ``manylinux_2_34`` string that
  its own ELF payload does not satisfy. The tag written here follows from the
  wheel's contents instead: ``objdump -T`` lists the ``GLIBC_x.y`` symbols each
  ``.so`` needs and the highest of them is the PEP 600 floor.

Both are applied in a single repack, because for the full release wheel the
alternative is decompressing and compressing about 1 GiB twice.

Usage:
    tools/fix_release_wheel.py --out dist dist/*.whl
    tools/fix_release_wheel.py --no-retag --out dist dist/*.whl
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

_GLIBC_VERSION = re.compile(r"GLIBC_(\d+)\.(\d+)")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
    "wheels", nargs="+", type=Path, help="Wheel files to normalize"
  )
  parser.add_argument(
    "--out",
    type=Path,
    help="Directory to write the fixed wheels to (default: beside each input)"
  )
  parser.add_argument(
    "--rpath", default="$ORIGIN", help="RPATH for every .so in the wheel"
  )
  parser.add_argument(
    "--no-retag", action="store_true", help="Keep the platform tag as built"
  )
  parser.add_argument(
    "--keep-original",
    action="store_true",
    help="Keep the input wheel when it was rewritten under a new name",
  )
  return parser.parse_args()


def _glibc_versions(elf: Path) -> set[str]:
  """The GLIBC symbol versions one .so references or defines.

  ``objdump -T`` lists both, so the floor can come out higher than auditwheel's
  -- which looks at undefined imports only. That direction is the safe one: a
  tag too high refuses a machine that could have run the wheel, a tag too low
  installs a wheel that fails to load.
  """
  output = subprocess.check_output(["objdump", "-T", str(elf)], text=True)
  return {f"{major}.{minor}" for major, minor in _GLIBC_VERSION.findall(output)}


def _version_key(text: str) -> tuple[int, int]:
  major, minor = text.split(".", 1)
  return int(major), int(minor)


def _platform_tag(glibc_versions: set[str]) -> str:
  """The PEP 600 tag for the highest GLIBC symbol the wheel needs."""
  if not glibc_versions:
    return "linux_x86_64"
  return f"manylinux_2_{max(glibc_versions, key=_version_key).split('.', 1)[1]}_x86_64"


def _parse_filename(wheel: Path) -> tuple[str, str, str, str]:
  if wheel.suffix != ".whl":
    raise SystemExit(f"fix_release_wheel: {wheel} is not a wheel")
  return wheel.stem.rsplit("-", 3)


def _hash_file(path: Path) -> tuple[str, str]:
  digest = hashlib.sha256(path.read_bytes()).digest()
  encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
  return f"sha256={encoded}", str(path.stat().st_size)


def _rewrite_record(extract_root: Path, dist_info: Path) -> None:
  rows: list[list[str]] = []
  for path in sorted(
    item for item in extract_root.rglob("*") if item.is_file()
  ):
    relative = path.relative_to(extract_root).as_posix()
    if relative == f"{dist_info.name}/RECORD":
      rows.append([relative, "", ""])
      continue
    digest, size = _hash_file(path)
    rows.append([relative, digest, size])
  with (dist_info / "RECORD").open("w", newline="") as record:
    csv.writer(record).writerows(rows)


def _set_rpath(elf: Path, rpath: str) -> None:
  subprocess.check_call(["patchelf", "--set-rpath", rpath, str(elf)])


def _retag_wheel_metadata(dist_info: Path, platform_tag: str) -> None:
  wheel_metadata = dist_info / "WHEEL"
  lines = wheel_metadata.read_text().splitlines()
  rewritten = []
  for line in lines:
    if line.startswith("Tag: "):
      python_tag, abi_tag, _ = line[5:].rsplit("-", 2)
      rewritten.append(f"Tag: {python_tag}-{abi_tag}-{platform_tag}")
      continue
    rewritten.append(line)
  wheel_metadata.write_text("\n".join(rewritten) + "\n")


def _fix_wheel(
  wheel: Path, out_dir: Path, rpath: str, retag: bool, keep_original: bool
) -> tuple[Path, str]:
  head, python_tag, abi_tag, built_tag = _parse_filename(wheel)
  with tempfile.TemporaryDirectory(prefix="ffpa_release_wheel_") as temp_dir:
    extract_root = Path(temp_dir) / "wheel"
    extract_root.mkdir(parents=True)
    with ZipFile(wheel) as archive:
      archive.extractall(extract_root)

    dist_info_dirs = list(extract_root.glob("*.dist-info"))
    if len(dist_info_dirs) != 1:
      raise SystemExit(
        f"fix_release_wheel: {wheel.name} has {len(dist_info_dirs)} .dist-info directories"
      )
    dist_info = dist_info_dirs[0]

    libraries = sorted(item for item in extract_root.rglob("*.so"))
    if not libraries:
      raise SystemExit(
        f"fix_release_wheel: {wheel.name} contains no .so; refusing to tag a pure-Python wheel"
      )
    glibc_versions: set[str] = set()
    for library in libraries:
      _set_rpath(library, rpath)
      glibc_versions.update(_glibc_versions(library))

    platform_tag = _platform_tag(glibc_versions)
    if retag:
      _retag_wheel_metadata(dist_info, platform_tag)
    _rewrite_record(extract_root, dist_info)

    final_tag = platform_tag if retag else built_tag
    fixed_path = out_dir / f"{head}-{python_tag}-{abi_tag}-{final_tag}.whl"
    with ZipFile(fixed_path, "w", compression=ZIP_DEFLATED) as archive:
      for path in sorted(
        item for item in extract_root.rglob("*") if item.is_file()
      ):
        # A fixed timestamp makes the repack reproducible for a given input.
        # Without it two runs produce different bytes, so the sha256 the index
        # records for a release would stop describing the asset it points at.
        info = ZipInfo(
          path.relative_to(extract_root).as_posix(), (1980, 1, 1, 0, 0, 0)
        )
        info.external_attr = stat.S_IMODE(path.stat().st_mode) << 16
        info.compress_type = ZIP_DEFLATED
        with archive.open(info, "w") as member, path.open("rb") as source:
          shutil.copyfileobj(source, member)

  if fixed_path != wheel and not keep_original:
    wheel.unlink()
  return fixed_path, platform_tag


def main() -> None:
  args = _parse_args()
  for command in ("patchelf", "objdump"):
    if not shutil.which(command):
      raise SystemExit(
        f"fix_release_wheel: {command} is required but not on PATH"
      )

  for wheel in args.wheels:
    out_dir = args.out or wheel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_path, platform_tag = _fix_wheel(
      wheel, out_dir, args.rpath, not args.no_retag, args.keep_original
    )
    size_mib = fixed_path.stat().st_size / (1024 * 1024)
    print(f"{fixed_path.name}\t{platform_tag}\t{size_mib:.1f} MiB")


if __name__ == "__main__":
  main()
