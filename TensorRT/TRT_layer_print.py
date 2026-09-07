#!/usr/bin/env python
"""
[TensorRT 레이어 이름 변화 — 터미널 출력]

build_trt_engine.py 가 덤프한 `<engine>.engine.layers.json` 을 읽어, 원본 ONNX(=PyTorch)
레이어가 TensorRT 커널로 오면서 **어떻게 합쳐지고 이름이 바뀌었는지** 텍스트로 출력한다.
(.md 정리본이 필요하면 → TensorRT/analysis.py)

    [1] TensorRT 커널 → 원본 ONNX 레이어   (엔진 실행 순서)
    [2] 원본 ONNX 레이어 → TensorRT 커널   (역방향)
    [3] best.onnx 에 있지만 엔진 metadata 에 안 보이는 노드  (fusion 흡수 / 상수 폴딩)
    요약: keep / rename / fuse / new 개수

분류(category):
    keep       원본 이름 그대로            예: /model.0/conv/Conv
    rename     1:1 인데 이름이 바뀜         예: .../attn/MatMul → /model_10/.../MatMul_myl109_2
    fuse(N)    원본 N개 → 커널 1개          예: SiLU(Sigmoid+Mul) → PointWiseV2 1개
    new-io     TRT 가 삽입한 재배치/복사    Reformat / NoOp (원본에 없음)
    new-int    TRT 내부 커널               shape_call, myelin fused blob 등 (metadata 없음)

실행:
    conda activate yolo
    python TensorRT/TRT_layer_print.py
    python TensorRT/TRT_layer_print.py --all                  # keep 포함 전부
    python TensorRT/TRT_layer_print.py --onnx model/best.onnx # [3] 섹션까지 (기본 경로면 자동)
    python TensorRT/TRT_layer_print.py --json model/layer_name_map.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for base in (here.parent, *here.parents, cwd, *cwd.parents):
        if (base / "yolo26" / "ultralytics" / "__init__.py").exists():
            return base
    return here.parent.parent


REPO_ROOT = find_repo_root()
ONNX_META_RE = re.compile(r"\[ONNX Layer:\s*(.*?)\]")
STAGE_RE = re.compile(r"model[._/](\d+)")


# --------------------------------------------------------------------------- #
# 파싱 & 분석  (analysis.py 도 이 함수들을 그대로 import 해서 씀)
# --------------------------------------------------------------------------- #
def load_engine_layers(path: Path):
    if not path.exists():
        sys.exit(f"[에러] 레이어 JSON 이 없습니다: {path}\n"
                 "       먼저 실행: python TensorRT/build_trt_engine.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data["Layers"], data.get("Bindings", [])) if isinstance(data, dict) else (data, [])


def onnx_layers_of(layer: dict) -> list[str]:
    """레이어 Metadata 의 '[ONNX Layer: X]' 들을 뽑는다 (fusion 원본 목록)."""
    return [s.strip() for s in ONNX_META_RE.findall(layer.get("Metadata", "")) if s.strip()]


def norm(s: str) -> str:
    return re.sub(r"[/_.\s]", "", s).lower()


def stage_of(name: str) -> str | None:
    m = STAGE_RE.search(name or "")
    return m.group(1) if m else None


def classify(layer: dict, onnx: list[str]) -> str:
    ltype = layer["LayerType"].lower()
    name = layer.get("Name", "")
    if not onnx:
        return "new-io" if ("reformat" in ltype or "noop" in ltype) else "new-int"
    if len(onnx) >= 2:
        return f"fuse({len(onnx)})"
    return "keep" if norm(name) == norm(onnx[0]) else "rename"


def reformat_target(name: str) -> str:
    """'Reformatting CopyNode for Input Tensor 0 to /model.1/conv/Conv' -> '/model.1/conv/Conv'"""
    return name.split(" to ", 1)[1].strip() if " to " in name else ""


def load_onnx_node_names(path: Path | None):
    if path is None or not Path(path).exists():
        return None
    try:
        import onnx
    except Exception as exc:
        print(f"[경고] onnx import 실패 → ONNX 대조 생략 ({exc})")
        return None
    g = onnx.load(str(path)).graph
    return [n.name if n.name else f"{n.op_type}_{i}" for i, n in enumerate(g.node)]


def analyze(engine_json: Path, onnx_path: Path | None = None) -> dict:
    """엔진 레이어 JSON(+ONNX)을 분석해 구조화된 dict 를 돌려준다."""
    layers, bindings = load_engine_layers(Path(engine_json))
    onnx_nodes = load_onnx_node_names(onnx_path)

    rows: list[dict] = []
    onnx_to_trt: dict[str, list[dict]] = defaultdict(list)
    referenced: set[str] = set()

    for idx, L in enumerate(layers):
        onnx = onnx_layers_of(L)
        cat = classify(L, onnx)
        note = ""
        if not onnx:
            tgt = reformat_target(L.get("Name", ""))
            note = tgt or "(내부/레이아웃)"
        row = dict(idx=idx, type=L["LayerType"], name=L.get("Name", ""),
                   onnx=onnx, cat=cat, note=note,
                   stage=stage_of(onnx[0] if onnx else L.get("Name", "")))
        rows.append(row)
        for o in onnx:
            referenced.add(o)
            onnx_to_trt[o].append(dict(idx=idx, type=L["LayerType"],
                                       trt_name=L.get("Name", ""), fused_n=len(onnx)))

    ref_norm = {norm(x) for x in referenced}
    eliminated = None
    if onnx_nodes is not None:
        eliminated = [n for n in onnx_nodes
                      if n not in referenced and norm(n) not in ref_norm]

    return dict(
        engine_json=str(engine_json),
        onnx_path=str(onnx_path) if onnx_path else None,
        bindings=bindings,
        n_layers=len(layers),
        onnx_nodes=onnx_nodes,
        rows=rows,
        onnx_to_trt={k: v for k, v in onnx_to_trt.items()},
        referenced=sorted(referenced),
        eliminated=eliminated,
    )


def summary_counts(res: dict) -> dict:
    cats = Counter(r["cat"].split("(")[0] for r in res["rows"])
    fuse_kernels = sum(1 for r in res["rows"] if r["cat"].startswith("fuse"))
    fuse_src = sum(len(r["onnx"]) for r in res["rows"] if r["cat"].startswith("fuse"))
    out = dict(keep=cats.get("keep", 0), rename=cats.get("rename", 0),
               fuse_kernels=fuse_kernels, fuse_src=fuse_src,
               new_io=cats.get("new-io", 0), new_int=cats.get("new-int", 0),
               n_kernels=res["n_layers"])
    if res["onnx_nodes"] is not None:
        out["onnx_total"] = len(res["onnx_nodes"])
        out["onnx_eliminated"] = len(res["eliminated"])
        out["onnx_tracked"] = len(res["onnx_nodes"]) - len(res["eliminated"])
    return out


# --------------------------------------------------------------------------- #
# 터미널 출력
# --------------------------------------------------------------------------- #
def _cut(s: str, w: int) -> str:
    return s if (w <= 0 or len(s) <= w) else s[: w - 1] + "…"


def print_report(res: dict, show_all: bool, width: int) -> None:
    rows = res["rows"]
    print("═" * 78)
    print("  TensorRT 레이어 이름 변화")
    print("═" * 78)
    print(f"  engine json : {res['engine_json']}")
    line = f"  TRT 커널    : {res['n_layers']}개"
    if res["onnx_nodes"] is not None:
        line += f"   |   원본 ONNX 노드 : {len(res['onnx_nodes'])}개 ({res['onnx_path']})"
    print(line)
    print(f"  bindings    : {res['bindings']}")

    # [1]
    print("\n── [1] TensorRT 커널 → 원본 ONNX  (엔진 실행 순서)"
          + ("" if show_all else "   ※ keep 은 --all 로") + " ──")
    for r in rows:
        if r["cat"] == "keep" and not show_all:
            continue
        src = " + ".join(r["onnx"]) if r["onnx"] else (
            f"→ {r['note']}" if r["note"] else "(원본 없음)")
        print(f"  {r['idx']:>4}  {r['cat']:<9}  {r['type']:<20}  {_cut(r['name'], width)}")
        print(f"  {'':>4}  {'':<9}  {'':<20}  ↳ {_cut(src, width * 2 if width else 0)}")

    # [2]
    print("\n── [2] 원본 ONNX 노드 → TensorRT 커널  (역방향) ──")
    for o in sorted(res["onnx_to_trt"]):
        for j, t in enumerate(res["onnx_to_trt"][o]):
            head = o if j == 0 else ""
            if norm(o) == norm(t["trt_name"]):
                tag = "이름 유지"
            elif t["fused_n"] >= 2:
                tag = f"{t['fused_n']}→1 fusion"
            else:
                tag = "이름 변경"
            print(f"  {head:<40}  →  #{t['idx']:<4} {t['type']:<18} "
                  f"\"{_cut(t['trt_name'], 46)}\"   ({tag})")

    # [3]
    if res["eliminated"] is not None:
        g = res["eliminated"]
        print(f"\n── [3] best.onnx 에 있지만 엔진 metadata 에 안 보이는 노드 "
              f"({len(g)}개) — fusion 흡수 / 상수 폴딩 ──")
        for i in range(0, len(g), 3):
            print("  " + " · ".join(g[i:i + 3]))

    # 요약
    c = summary_counts(res)
    print("\n── 요약 ──")
    print(f"  keep {c['keep']} · rename {c['rename']} · "
          f"fuse {c['fuse_kernels']}  (원본 {c['fuse_src']}개 → 커널 {c['fuse_kernels']}개)")
    print(f"  new-io {c['new_io']} · new-int {c['new_int']}   |   TRT 커널 총 {c['n_kernels']}개")
    if "onnx_total" in c:
        print(f"  원본 ONNX {c['onnx_total']}개 중  추적됨 {c['onnx_tracked']} / "
              f"흡수·제거 {c['onnx_eliminated']}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TensorRT 커널 ↔ 원본 ONNX 레이어 이름 대응 (터미널 출력)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--engine-json", default="model/best.engine.layers.json",
                   help="build_trt_engine.py 가 만든 레이어 정보 JSON — repo 루트 기준")
    p.add_argument("--onnx", default="model/best.onnx",
                   help="원본 ONNX (있으면 [3] 섹션 출력). 없으면 자동 생략")
    p.add_argument("--all", action="store_true", help="[1] 에서 keep 줄도 전부 출력")
    p.add_argument("--width", type=int, default=0, help="이름 출력 최대폭 (0=자르지 않음)")
    p.add_argument("--json", default=None, help="대응표를 JSON 으로도 저장할 경로")
    return p.parse_args()


def main() -> None:
    os.chdir(REPO_ROOT)          # 이후 모든 상대경로는 repo 루트 기준
    args = parse_args()
    onnx_path = Path(args.onnx) if args.onnx and Path(args.onnx).exists() else None

    res = analyze(Path(args.engine_json), onnx_path)
    print_report(res, show_all=args.all, width=args.width)

    if args.json:
        payload = dict(
            engine_json=res["engine_json"], onnx=res["onnx_path"],
            summary=summary_counts(res),
            trt_layers=[{k: r[k] for k in ("idx", "type", "name", "onnx", "cat")}
                        for r in res["rows"]],
            onnx_to_trt=res["onnx_to_trt"],
            eliminated_onnx=res["eliminated"],
        )
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n[저장] {args.json}")


if __name__ == "__main__":
    main()
