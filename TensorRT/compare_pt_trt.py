#!/usr/bin/env python
"""
[.pt vs TensorRT 구조 비교]

원본 `best.pt` 의 24개 nn.Module(`model.0`~`model.23`) 각각이 TensorRT 엔진에서
**커널 몇 개로, 어떤 타입으로, 어떤 정밀도로** 바뀌었는지 나란히 표로 정리한다.

    .pt 레이어 구조 그래프   → TensorRT/ori_visualize_model.py
    ONNX ↔ TRT 커널 이름     → TensorRT/TRT_layer_print.py  (ONNX 기준)
    이 스크립트              → .pt 모듈 ↔ TRT 커널 집계 비교  (.pt 기준)

`.pt` 모듈 이름(`model.N`)은 export 후에도 ONNX 노드 이름(`/model.N/...`)과
TRT 커널 metadata(`[ONNX Layer: /model.N/...]`)에 그대로 남는다. 그걸로 316개
커널을 원본 모듈로 되묶어서 1:1 로 비교한다.

실행:
    conda activate yolo                         # torch + ultralytics 필요 (.pt 로드)
    python TensorRT/compare_pt_trt.py
    python TensorRT/compare_pt_trt.py --weights model/best.pt \
        --engine-json model/best.engine.layers.json --out pt_vs_trt.md
    python TensorRT/compare_pt_trt.py --no-shapes            # forward 패스 생략

입력 JSON 이 없으면 먼저:
    python TensorRT/build_trt_engine.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 형제 스크립트 import 용
from ori_visualize_model import (  # noqa: E402
    add_local_fork_to_path, collect_layers, find_repo_root, human, load_model, shape_str,
)
from TRT_visualize_model import family, load_layers, out_info  # noqa: E402
from TRT_layer_print import load_onnx_node_names, onnx_layers_of  # noqa: E402

REPO_ROOT = find_repo_root()
STAGE_NUM_RE = re.compile(r"model[._/](\d+)")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=".pt 모듈 ↔ TensorRT 커널 집계 비교 → Markdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default="model/best.pt",
                   help="원본 가중치 (.pt) — repo 루트 기준")
    p.add_argument("--engine-json", default="model/best.engine.layers.json",
                   help="build_trt_engine.py 가 만든 레이어 정보 JSON")
    p.add_argument("--onnx", default="",
                   help="(선택) 원본 ONNX 경로. 주면 표 가운데에 ONNX 노드 수도 끼워 "
                        "'.pt→ONNX→TRT' 로 보여준다. 기본은 끔 = '.pt→TRT' 만")
    p.add_argument("--out", default="pt_vs_trt.md", help="출력 Markdown 경로 (repo 루트 기준)")
    p.add_argument("--imgsz", type=int, default=640, help="shape 관측용 더미 입력 크기")
    p.add_argument("--no-shapes", action="store_true",
                   help="forward 패스를 건너뛰고 .pt 출력 shape 표기를 생략")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def kernel_stage(L: dict) -> str | None:
    """TRT 커널 하나가 원본 어느 `model.N` 단계에 속하는지.

    fused 커널은 metadata 에 `[ONNX Layer: /model.N/...]` 이 여러 개 붙으므로
    그 중 **다수결**로 정한다 (첫 매치만 쓰면 엉뚱한 단계로 감).
    metadata 가 없으면 커널 이름에서, 그것도 없으면 None(=post).
    """
    votes = Counter()
    for o in onnx_layers_of(L):
        m = STAGE_NUM_RE.search(o)
        if m:
            votes[m.group(1)] += 1
    if votes:
        return votes.most_common(1)[0][0]
    m = STAGE_NUM_RE.search(L.get("Name", ""))
    return m.group(1) if m else None


def onnx_stage_counts(onnx_path: Path | None) -> dict[str, int]:
    """원본 ONNX 노드를 `model.N` 단계별로 센다 (TRT 가 융합하기 '전' 개수).
    ONNX 없거나 로드 실패면 빈 dict."""
    names = load_onnx_node_names(onnx_path) if onnx_path else None
    if not names:
        return {}
    c: Counter = Counter()
    for nm in names:
        m = STAGE_NUM_RE.search(nm)
        if m:
            c[m.group(1)] += 1
    return dict(c)


def aggregate_trt(layers_trt: list[dict]) -> dict:
    """stage(문자열 '0'..'23' 또는 None) -> 집계 dict."""
    by_stage: dict[str | None, list[int]] = defaultdict(list)
    for i, L in enumerate(layers_trt):
        by_stage[kernel_stage(L)].append(i)

    agg: dict = {}
    for st, idxs in by_stage.items():
        fam = Counter()
        dt = Counter()
        dims_last = ""
        for i in idxs:
            L = layers_trt[i]
            fam[family(L["LayerType"])[0]] += 1
            d, t = out_info(L)
            if t:
                dt[t] += 1
            if d:
                dims_last = d
        agg[st] = dict(
            n=len(idxs),
            fam=fam,
            dt=dt,
            dims=dims_last,
            kernels=[(layers_trt[i].get("Name", ""), layers_trt[i]["LayerType"]) for i in idxs],
        )
    return agg


# --------------------------------------------------------------------------- #
def md_cell(s: str) -> str:
    return str(s).replace("|", "\\|")


def note_for(pt_type: str, a: dict | None) -> str:
    t = pt_type.lower()
    if a is None or a["n"] == 0:
        return "TRT fusion 흡수 — 독립 커널 없음"
    if "detect" in t:
        return "end2end decode+NMS 펼침 (`i64` 텐서 등장)"
    if "concat" in t:
        return "일부만 커널로 남음 (대부분 zero-copy)"
    fam = a["fam"]
    bits = []
    if pt_type in ("Conv", "C3k2", "C2PSA", "SPPF") and "conv" in fam:
        bits.append("Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise")
    if fam.get("reformat", 0) or fam.get("noop", 0):
        bits.append(f"레이아웃 변환 {fam.get('reformat', 0) + fam.get('noop', 0)}개 삽입")
    return " · ".join(bits)


def count_cell(pt_n: int, onnx_n: int | None, trt_n: int) -> str:
    """'.pt 내부 레이어 → [ONNX 노드] → TRT 커널' 체인.

    `.pt` 모듈 1개는 그 자체로 '커널 수'가 없으므로, 안에 든 leaf 서브모듈
    (Conv2d·BN·SiLU·MaxPool …) 수를 왼쪽 값으로 쓴다. ONNX(`--onnx`)가 있으면
    중간에 노드 수도 끼워 'export 로 펼쳐졌다가 TRT 가 융합' 흐름을 보여준다."""
    mid = f"{onnx_n} → " if onnx_n is not None else ""
    return f"{pt_n} → {mid}{trt_n}"


def to_markdown(layers_pt: list[dict], agg: dict, onnx_counts: dict, bindings,
                backbone_len: int, weights: str, engine_json: str, onnx_path: str | None,
                n_kernels: int) -> str:
    L: list[str] = []
    ap = L.append
    have_onnx = bool(onnx_counts)

    total_params = sum(x["params"] for x in layers_pt)
    mapped = [i for i in range(len(layers_pt)) if str(i) in agg and agg[str(i)]["n"] > 0]
    fused_away = [i for i in range(len(layers_pt)) if str(i) not in agg or agg[str(i)]["n"] == 0]
    # 연산 커널(conv/gemm/pointwise/pool/resize/post) 없이 레이아웃 변환만 남은 모듈
    compute_fams = {"conv", "gemm", "pointwise", "pool", "resize", "post"}
    structure_only = [i for i in range(len(layers_pt))
                      if str(i) in agg and agg[str(i)]["n"] > 0
                      and not (compute_fams & set(agg[str(i)]["fam"]))]
    post = agg.get(None, {}).get("n", 0)
    mapped_kernels = sum(agg[str(i)]["n"] for i in mapped)

    ap("# `.pt` vs TensorRT 구조 비교")
    ap("")
    ap("원본 `best.pt` 의 nn.Module(`model.N`) 이 TensorRT 엔진에서 커널 몇 개로 "
       "바뀌었는지 `.pt` 기준으로 나란히 본 문서. (`TensorRT/compare_pt_trt.py` 자동 생성)")
    ap("")
    onnx_mapped_total = sum(v for k, v in onnx_counts.items() if k.isdigit()) if have_onnx else None

    ap(f"- 원본 가중치: `{weights}`")
    ap(f"- 엔진 레이어 정보: `{engine_json}`")
    if onnx_path:
        ap(f"- 원본 ONNX: `{onnx_path}`")
    ap(f"- 엔진 IO: `{bindings}`")
    ap(f"- 생성일: {date.today().isoformat()}")
    ap("")

    pt_leaf_total = sum(x.get("leaf", 0) for x in layers_pt)
    col_hdr = "레이어 수 (.pt→ONNX→TRT)" if have_onnx else "레이어 수 (.pt→TRT)"

    ap("## 요약")
    ap("")
    ap("| | 값 |")
    ap("|---|---|")
    ap(f"| `.pt` 모듈 | {len(layers_pt)}개 · 내부 leaf 레이어 {pt_leaf_total}개 · params {human(total_params)} |")
    if have_onnx:
        ap(f"| ONNX 노드 (`model.N` 매칭) | {onnx_mapped_total}개 |")
    ap(f"| TRT 커널 | {n_kernels}개 (매핑 {mapped_kernels} + post {post}) |")
    ap(f"| 전체 흐름 | `.pt` leaf {pt_leaf_total} "
       + (f"→ ONNX {onnx_mapped_total} " if have_onnx else "")
       + f"→ TRT 커널 {mapped_kernels} |")
    ap(f"| 레이아웃 변환만 남은 모듈 | {len(structure_only)}개 "
       f"({', '.join('model.' + str(i) for i in structure_only) or '없음'}) — "
       f"Concat 은 zero-copy, reformat 1개씩만 |")
    if fused_away:
        ap(f"| 커널이 아예 없는 모듈 | {len(fused_away)}개 "
           f"({', '.join('model.' + str(i) for i in fused_away)}) |")
    ap("")
    ap(f"> 커널이 가장 많은 단계: **model.23** (Detect, {agg.get('23', {}).get('n', 0)}개 — "
       f"end2end decode/NMS 까지 펼쳐짐).")
    ap("")

    ap("## 모듈별 대응표")
    ap("")
    ap(f"`{col_hdr}` = 그 `model.N` 의 **`.pt` 내부 leaf 서브모듈 수**"
       + (" → ONNX 노드 수" if have_onnx else "")
       + " → **TRT 커널 수**. "
       "`.pt` 모듈 1개엔 '커널 수'가 없으므로 안에 든 Conv2d·BN·SiLU·MaxPool 등을 센다"
       " (SiLU 는 공유돼 1로 세짐, `torch.cat`·`+` 같은 함수형 연산은 안 세짐 — 어림값). "
       "`.pt` 출력 shape 는 forward 관측값, `TRT 출력` 은 그 단계 마지막 커널의 출력.")
    ap("")
    ap(f"| 단계 | `.pt` 타입 | params | `.pt` 출력 | {col_hdr} | TRT 커널 타입 | TRT 출력 | 비고 |")
    ap("|---|---|--:|---|:--:|---|---|---|")
    for i, x in enumerate(layers_pt):
        a = agg.get(str(i))
        trt_n = a["n"] if a else 0
        onnx_n = onnx_counts.get(str(i)) if have_onnx else None
        seg = "backbone" if i < backbone_len else ("head" if i < len(layers_pt) - 1 else "detect")
        fam_s = " ".join(f"{k}×{v}" for k, v in a["fam"].most_common()) if a else "—"
        prec_s = "/".join(k for k, _ in a["dt"].most_common(2)) if a and a["dt"] else "—"
        trt_out = f"{a['dims']} · {prec_s}" if a and a["dims"] else (prec_s if a else "—")
        ap(f"| **model.{i}** ({seg}) | {x['type']} | {human(x['params'])} "
           f"| {md_cell(shape_str(x['shape']) or '—')} "
           f"| {count_cell(x.get('leaf', 0), onnx_n, trt_n)} "
           f"| {md_cell(fam_s)} "
           f"| {md_cell(trt_out)} "
           f"| {md_cell(note_for(x['type'], a))} |")
    if None in agg:
        a = agg[None]
        fam_s = " ".join(f"{k}×{v}" for k, v in a["fam"].most_common())
        ap(f"| _post-process_ | — | — | — | — → {a['n']} | {md_cell(fam_s)} "
           f"| {md_cell(a['dims'])} | end2end 후처리 / 이름 매핑 안 되는 내부 커널 |")
    ap("")

    ap("## 단계별 TRT 커널 목록")
    ap("")
    for i in range(len(layers_pt)):
        a = agg.get(str(i))
        if not a or a["n"] == 0:
            continue
        onnx_n = onnx_counts.get(str(i)) if have_onnx else None
        head = f".pt leaf {layers_pt[i].get('leaf', 0)} → "
        head += f"ONNX {onnx_n} → " if onnx_n is not None else ""
        head += f"커널 {a['n']}개"
        ap(f"<details><summary><b>model.{i}</b> ({layers_pt[i]['type']}) — {head}</summary>")
        ap("")
        ap("| # | 타입 | 커널 이름 |")
        ap("|--:|---|---|")
        for j, (nm, ty) in enumerate(a["kernels"]):
            ap(f"| {j} | {ty} | {md_cell('`' + nm + '`')} |")
        ap("")
        ap("</details>")
        ap("")
    if None in agg:
        a = agg[None]
        ap(f"<details><summary><b>post-process</b> — 커널 {a['n']}개</summary>")
        ap("")
        ap("| # | 타입 | 커널 이름 |")
        ap("|--:|---|---|")
        for j, (nm, ty) in enumerate(a["kernels"]):
            ap(f"| {j} | {ty} | {md_cell('`' + nm + '`')} |")
        ap("")
        ap("</details>")
        ap("")

    ap("## 읽는 법")
    ap("")
    ap(f"- **`{col_hdr}`** — 왼쪽은 `.pt` 모듈 안의 leaf 서브모듈 수(정확한 연산 수 아님, "
       "어림값), 오른쪽은 그 단계로 매핑된 TRT 커널 수. "
       + ("가운데는 ONNX 노드 수(참고용, `--onnx` 있을 때만). " if have_onnx else
          "`--onnx model/best.onnx` 를 주면 가운데에 ONNX 노드 수도 표시된다. ")
       + "ONNX 없이도 `.pt→TRT` 는 그대로 나온다.")
    ap("- **커널 타입** 은 LayerType 계열명: `conv`(CaskConvolution) / `gemm`(MatMul·"
       "FullyConnected) / `pointwise`(활성화·elementwise) / `pool` / `reformat`·`noop`"
       "(레이아웃 변환) / `resize` / `post`(slice·concat·topk·nms).")
    ap("- `.pt` 의 `Conv` = Conv2d+BN+SiLU. TRT 에서 BN 은 conv 가중치에 접히고(fold), "
       "SiLU(Sigmoid+Mul)는 `PointWiseV2` 1개로 fusion 된다.")
    ap("- `Concat`(model.12/15/18/21) 은 대부분 zero-copy 로 처리돼 독립 커널이 "
       "거의 없다 → `1 → 1` (reformat 1개).")
    ap("- `Detect`(model.23) 는 학습용 구조가 사라지고 추론 경로 + DFL + box decode + "
       "TopK/NMS 가 그래프로 펼쳐져 커널 수가 가장 많다.")
    ap("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def print_console(layers_pt, agg, onnx_counts, n_kernels):
    have_onnx = bool(onnx_counts)
    print()
    print("═" * 92)
    print("  .pt 모듈 (내부 leaf 레이어)  →  " + ("ONNX 노드  →  " if have_onnx else "") + "TensorRT 커널")
    print("═" * 92)
    col = ".pt→ONNX→TRT" if have_onnx else ".pt→TRT"
    print(f"  {'단계':<10}{'.pt 타입':<12}{'params':>9}   {col:>16}   커널 타입")
    print("  " + "─" * 88)
    for i, x in enumerate(layers_pt):
        a = agg.get(str(i))
        n = a["n"] if a else 0
        mid = f"{onnx_counts.get(str(i), 0)} → " if have_onnx else ""
        cell = f"{x.get('leaf', 0)} → {mid}{n}"
        fam_s = " ".join(f"{k}×{v}" for k, v in a["fam"].most_common(4)) if a else "(fusion 흡수)"
        print(f"  model.{i:<4}{x['type']:<12}{human(x['params']):>9}   {cell:>16}   {fam_s}")
    if None in agg:
        a = agg[None]
        fam_s = " ".join(f"{k}×{v}" for k, v in a["fam"].most_common(4))
        cell = f"— → {a['n']}"
        print(f"  {'post':<10}{'—':<12}{'—':>9}   {cell:>16}   {fam_s}")
    print("  " + "─" * 88)
    total_mapped = sum(a["n"] for k, a in agg.items() if k is not None)
    pt_leaf_total = sum(x.get("leaf", 0) for x in layers_pt)
    flow = f"  전체: .pt leaf {pt_leaf_total} → "
    if have_onnx:
        flow += f"ONNX {sum(v for k, v in onnx_counts.items() if k.isdigit())} → "
    flow += f"TRT 커널 {total_mapped} (+ post {agg.get(None, {}).get('n', 0)})"
    print(flow)
    compute_fams = {"conv", "gemm", "pointwise", "pool", "resize", "post"}
    structure_only = [i for i in range(len(layers_pt))
                      if str(i) in agg and agg[str(i)]["n"] > 0
                      and not (compute_fams & set(agg[str(i)]["fam"]))]
    print(f"  레이아웃 변환만 남은 모듈(Concat): "
          f"{', '.join('model.' + str(i) for i in structure_only) or '없음'}")


# --------------------------------------------------------------------------- #
def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()

    add_local_fork_to_path()
    net = load_model(args.weights)
    yaml = getattr(net, "yaml", {}) or {}
    backbone_len = len(yaml.get("backbone", [])) or 11
    layers_pt = collect_layers(net, args.imgsz, want_shapes=not args.no_shapes)
    # 각 .pt 모듈 안의 leaf 서브모듈 수 (Conv2d·BN·SiLU·MaxPool …). 모듈 1개엔
    # '커널 수'가 없으므로 이걸 '융합 전' 왼쪽 값으로 쓴다 (함수형 연산은 빠짐 — 어림값).
    for i, m in enumerate(net.model):
        layers_pt[i]["leaf"] = sum(1 for x in m.modules() if not list(x.children()))

    layers_trt, bindings = load_layers(Path(args.engine_json))
    agg = aggregate_trt(layers_trt)

    onnx_path = Path(args.onnx) if args.onnx and Path(args.onnx).exists() else None
    onnx_counts = onnx_stage_counts(onnx_path)
    if args.onnx and onnx_path is None:
        print(f"[정보] ONNX 없음 ({args.onnx}) → 'ONNX→TRT' 개수 표기 생략")

    print_console(layers_pt, agg, onnx_counts, len(layers_trt))

    md = to_markdown(layers_pt, agg, onnx_counts, bindings, backbone_len,
                     args.weights, args.engine_json,
                     str(onnx_path) if onnx_path else None, len(layers_trt))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n[완료] {out}")


if __name__ == "__main__":
    main()
