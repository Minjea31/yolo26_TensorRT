#!/usr/bin/env python
"""
[TensorRT 레이어 변환 분석 → Markdown]

build_trt_engine.py 가 덤프한 `<engine>.engine.layers.json` (+ `best.onnx`) 을 분석해,
**원래 레이어가 어떻게 합쳐졌고(fusion) 어떤 이름으로 바뀌었는지** 를 `.md` 문서로 정리한다.
분석 로직은 TRT_layer_print.py 의 `analyze()` 를 그대로 재사용한다.

실행:
    conda activate yolo
    python TensorRT/analysis.py
    python TensorRT/analysis.py --engine-json model/best.engine.layers.json \
        --onnx model/best.onnx --out model/TRT_layer_analysis.md

입력 JSON 이 없으면 먼저:
    python TensorRT/build_trt_engine.py
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # TRT_layer_print 임포트용
from TRT_layer_print import analyze, find_repo_root, norm, summary_counts  # noqa: E402

REPO_ROOT = find_repo_root()


# --------------------------------------------------------------------------- #
def code(s: str) -> str:
    """마크다운 표 셀용: 파이프 이스케이프 + 백틱."""
    return "`" + str(s).replace("|", "\\|") + "`"


def _stage_key(s):
    return (0, int(s)) if (s and str(s).isdigit()) else (1, str(s))


def to_markdown(res: dict) -> str:
    rows = res["rows"]
    c = summary_counts(res)
    L = []
    ap = L.append

    ap("# TensorRT 레이어 변환 분석")
    ap("")
    ap("`best.pt` → ONNX → TensorRT 변환에서 **원본 레이어가 어떻게 합쳐지고(fusion) "
       "이름이 바뀌었는지** 정리한 문서. (`TensorRT/analysis.py` 자동 생성)")
    ap("")
    ap(f"- 엔진 레이어 정보: `{res['engine_json']}`")
    if res["onnx_path"]:
        ap(f"- 원본 ONNX: `{res['onnx_path']}`")
    ap(f"- 엔진 IO: `{res['bindings']}`")
    ap(f"- 생성일: {date.today().isoformat()}")
    ap("")

    # ---- 요약 ----
    ap("## 요약")
    ap("")
    ap("| 분류 | 개수 | 뜻 |")
    ap("|---|--:|---|")
    ap(f"| keep | {c['keep']} | 원본 이름 그대로 |")
    ap(f"| rename | {c['rename']} | 1:1 인데 이름만 바뀜 |")
    ap(f"| fuse | {c['fuse_kernels']} | 원본 {c['fuse_src']}개 → 커널 {c['fuse_kernels']}개로 fusion |")
    ap(f"| new-io | {c['new_io']} | TRT 가 삽입한 재배치/복사 (Reformat·NoOp) |")
    ap(f"| new-int | {c['new_int']} | TRT 내부 커널 (shape_call·myelin blob 등) |")
    if "onnx_total" in c:
        ap(f"| (흡수·제거) | {c['onnx_eliminated']} | `best.onnx` 노드 중 엔진에서 사라짐 (상수폴딩/fusion) |")
    ap("")
    tail = f"  ·  원본 ONNX {c['onnx_total']}개 → 추적 {c['onnx_tracked']} / 흡수 {c['onnx_eliminated']}" \
        if "onnx_total" in c else ""
    ap(f"> TRT 커널 총 **{c['n_kernels']}개**{tail}")
    ap("")

    # ---- 1. Fusion ----
    ap("## 1. Fusion — 원본 여러 개 → 커널 1개")
    ap("")
    ap("원본 `model.N` 단계별로 묶음. `+` 로 이어진 원본 op 들이 한 커널로 합쳐졌다.")
    ap("")
    fused = [r for r in rows if r["cat"].startswith("fuse")]
    by_stage: dict[str, list] = {}
    for r in fused:
        by_stage.setdefault(r["stage"] or "기타/post", []).append(r)
    for st in sorted(by_stage, key=_stage_key):
        ap(f"### model.{st}" if str(st).isdigit() else f"### {st}")
        ap("")
        ap("| TRT 커널 | 타입 | ← 합쳐진 원본 ONNX |")
        ap("|---|---|---|")
        for r in by_stage[st]:
            src = " + ".join(code(o) for o in r["onnx"])
            ap(f"| {code(r['name'])} | {r['type']} | {src} |")
        ap("")

    # ---- 2. rename ----
    ap("## 2. 이름만 바뀐 레이어 (rename)")
    ap("")
    ren = [r for r in rows if r["cat"] == "rename"]
    if ren:
        ap("| 원본 ONNX | → TRT 커널 | 타입 |")
        ap("|---|---|---|")
        for r in ren:
            ap(f"| {code(r['onnx'][0])} | {code(r['name'])} | {r['type']} |")
    else:
        ap("_없음_")
    ap("")

    # ---- 3. TRT 신규 삽입 ----
    ap("## 3. TRT 가 새로 삽입한 레이어 (new-io / new-int)")
    ap("")
    ap("원본 ONNX 에 없던 것. `new-io` = 텐서 레이아웃/포맷 변환용 복사, `new-int` = 내부 커널.")
    ap("")
    new = [r for r in rows if r["cat"] in ("new-io", "new-int")]
    ap(f"<details><summary>{len(new)}개 펼치기</summary>")
    ap("")
    ap("| # | 분류 | 타입 | 이름 | 대상/비고 |")
    ap("|--:|---|---|---|---|")
    for r in new:
        ap(f"| {r['idx']} | {r['cat']} | {r['type']} | {code(r['name'])} | "
           f"{code(r['note']) if r['note'] else ''} |")
    ap("")
    ap("</details>")
    ap("")

    # ---- 4. 흡수/제거 ----
    if res["eliminated"] is not None:
        g = res["eliminated"]
        ap(f"## 4. 원본에 있었지만 엔진에서 사라진 노드 ({len(g)}개)")
        ap("")
        ap("상수 폴딩되거나 다른 커널에 흡수됨. 대부분 `Constant_*` / `Split` / `Concat` / "
           "`Reshape` 와 end2end decode 블록(`/model.23/*`).")
        ap("")
        ap("<details><summary>전체 목록</summary>")
        ap("")
        ap(" · ".join(code(n) for n in g))
        ap("")
        ap("</details>")
        ap("")

    # ---- 5. keep ----
    kept = [r for r in rows if r["cat"] == "keep"]
    ap(f"## 5. 이름 유지 (keep) — 참고 ({len(kept)}개)")
    ap("")
    ap("<details><summary>펼치기</summary>")
    ap("")
    ap(" · ".join(code(r["name"]) for r in kept))
    ap("")
    ap("</details>")
    ap("")

    return "\n".join(L)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TensorRT 레이어 변환(fusion·rename) 분석 → Markdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--engine-json", default="model/best.engine.layers.json",
                   help="build_trt_engine.py 가 만든 레이어 정보 JSON — repo 루트 기준")
    p.add_argument("--onnx", default="model/best.onnx",
                   help="원본 ONNX (있으면 '사라진 노드' 섹션 포함)")
    p.add_argument("--out", default="model/TRT_layer_analysis.md",
                   help="출력 Markdown 경로")
    return p.parse_args()


def main() -> None:
    os.chdir(REPO_ROOT)          # 이후 모든 상대경로는 repo 루트 기준
    args = parse_args()
    onnx_path = Path(args.onnx) if args.onnx and Path(args.onnx).exists() else None

    res = analyze(Path(args.engine_json), onnx_path)
    md = to_markdown(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    c = summary_counts(res)
    print(f"[완료] {out}")
    print(f"       keep {c['keep']} · rename {c['rename']} · fuse {c['fuse_kernels']} "
          f"(원본 {c['fuse_src']}→{c['fuse_kernels']}) · new {c['new_io'] + c['new_int']}")


if __name__ == "__main__":
    main()
