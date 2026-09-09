#!/usr/bin/env python
"""
[.pt vs TensorRT 엔진 — 정확도/속도 비교]

같은 데이터셋(`dataset.yaml`)의 한 split(기본 `test`)을 **batch=1** 로 돌려
원본 `model/best.pt` 와 TensorRT 엔진 `model/best.engine` 의
검증 지표(mAP50-95 / mAP50 / mAP75 / Precision / Recall)와
추론 속도(preprocess / inference / postprocess, ms per image)를 나란히 비교한다.

    .pt 모듈 ↔ TRT 커널 "구조" 비교   → TensorRT/compare_pt_trt.py
    이 스크립트                        → .pt vs 엔진 "정확도·속도" 비교 (Ultralytics val)

실행:
    conda activate yolo                         # torch + ultralytics + tensorrt 필요
    python yolo26/eval.py                       # model/best.pt vs model/best.engine, split=test, batch=1
    python yolo26/eval.py --split val
    python yolo26/eval.py --pt-only             # .pt 만
    python yolo26/eval.py --engine-only         # 엔진만
    python yolo26/eval.py --engine model/best.engine --batch 1

엔진이 없으면 먼저 만든다:
    python TensorRT/build_trt_engine.py

결과:
    - 콘솔에 비교 표
    - `eval_pt_vs_trt.md` (repo 루트) 로 저장 (`--out` 으로 변경)
    - val 산출물(PR curve 등)은 `runs/eval/<pt|trt>/` (`--no-plots` 로 끔)

※ `model/best.engine` 은 정적 batch=1 로 빌드돼 있어 batch 는 1 로 고정하는 것을 권장한다.
  (build_trt_engine.py 기본값. `--dynamic` 없이 빌드하면 다른 batch 로는 추론 불가.)
"""
from __future__ import annotations

import argparse
import os
import unicodedata
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # yolo26/eval.py → repo 루트
RUNS_DIR = REPO_ROOT / "runs" / "eval"


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=".pt vs TensorRT 엔진 정확도/속도 비교 (Ultralytics val, batch=1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pt", default="model/best.pt",
                   help="원본 가중치 (.pt) — repo 루트 기준 상대경로")
    p.add_argument("--engine", default="model/best.engine",
                   help="TensorRT 엔진 (.engine) — repo 루트 기준 상대경로")
    p.add_argument("--data", default="dataset.yaml", help="데이터셋 yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="평가에 쓸 split")
    p.add_argument("--imgsz", type=int, default=640, help="입력 해상도")
    p.add_argument("--batch", type=int, default=1,
                   help="배치 크기 (정적 엔진은 1 이어야 함)")
    p.add_argument("--workers", type=int, default=4, help="데이터로더 workers")
    p.add_argument("--out", default="eval_pt_vs_trt.md",
                   help="비교 결과 Markdown 저장 경로 (repo 루트 기준)")
    p.add_argument("--pt-only", action="store_true", help=".pt 만 평가")
    p.add_argument("--engine-only", action="store_true", help="엔진만 평가")
    p.add_argument("--no-plots", action="store_true",
                   help="val 플롯(PR curve·혼동행렬 등) 생성 안 함")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def evaluate(weights: Path, args: argparse.Namespace, tag: str) -> dict:
    """weights 를 Ultralytics val 로 돌려 지표/속도를 dict 로 반환."""
    from ultralytics import YOLO

    print("\n" + "=" * 78)
    print(f"  [{tag}]  {weights}   (split={args.split}, imgsz={args.imgsz}, batch={args.batch})")
    print("=" * 78)

    model = YOLO(str(weights), task="detect")
    res = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        plots=not args.no_plots,
        project=str(RUNS_DIR),
        name=tag,
        exist_ok=True,
        verbose=False,
    )
    b, sp = res.box, res.speed
    total_ms = sp.get("preprocess", 0.0) + sp.get("inference", 0.0) + sp.get("postprocess", 0.0)
    return {
        "tag": tag,
        "weights": str(weights),
        "map50_95": float(b.map),
        "map50": float(b.map50),
        "map75": float(b.map75),
        "precision": float(b.mp),
        "recall": float(b.mr),
        "pre_ms": float(sp.get("preprocess", 0.0)),
        "inf_ms": float(sp.get("inference", 0.0)),
        "post_ms": float(sp.get("postprocess", 0.0)),
        "total_ms": float(total_ms),
    }


# --------------------------------------------------------------------------- #
METRIC_ROWS = [
    ("mAP50-95", "map50_95", 4, "높을수록"),
    ("mAP50", "map50", 4, "높을수록"),
    ("mAP75", "map75", 4, "높을수록"),
    ("Precision", "precision", 4, "높을수록"),
    ("Recall", "recall", 4, "높을수록"),
]
SPEED_ROWS = [
    ("전처리 (ms/img)", "pre_ms", 3),
    ("추론 (ms/img)", "inf_ms", 3),
    ("후처리 (ms/img)", "post_ms", 3),
    ("합계 (ms/img)", "total_ms", 3),
]


def _delta(trt: float, pt: float, nd: int) -> str:
    d = trt - pt
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(d):.{nd}f}"


def _pad(s: str, width: int) -> str:
    """한글(전각) 폭을 고려해 왼쪽 정렬."""
    disp = sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)
    return s + " " * max(0, width - disp)


def print_table(pt: dict | None, trt: dict | None) -> None:
    print("\n" + "─" * 78)
    print("  .pt  vs  TensorRT engine")
    print("─" * 78)
    if pt and trt:
        print(f"  {_pad('지표', 18)}{'.pt':>12}{'TRT engine':>14}{'Δ (TRT−.pt)':>16}")
        print("  " + "-" * 74)
        for name, key, nd, _ in METRIC_ROWS:
            print(f"  {_pad(name, 18)}{pt[key]:>12.{nd}f}{trt[key]:>14.{nd}f}"
                  f"{_delta(trt[key], pt[key], nd):>16}")
        print("  " + "-" * 74)
        for name, key, nd in SPEED_ROWS:
            extra = ""
            if key in ("inf_ms", "total_ms") and trt[key] > 0:
                extra = f"  ({pt[key] / trt[key]:.2f}× 빠름)"
            print(f"  {_pad(name, 18)}{pt[key]:>12.{nd}f}{trt[key]:>14.{nd}f}"
                  f"{_delta(trt[key], pt[key], nd):>16}{extra}")
    else:
        one = pt or trt
        print(f"  {_pad('지표', 18)}{one['tag']:>14}")
        print("  " + "-" * 34)
        for name, key, nd, _ in METRIC_ROWS:
            print(f"  {_pad(name, 18)}{one[key]:>14.{nd}f}")
        for name, key, nd in SPEED_ROWS:
            print(f"  {_pad(name, 18)}{one[key]:>14.{nd}f}")
    print("─" * 78)


def to_markdown(pt: dict | None, trt: dict | None, args: argparse.Namespace) -> str:
    L: list[str] = []
    ap = L.append
    ap("# `.pt` vs TensorRT 엔진 — 정확도/속도 비교")
    ap("")
    ap("원본 `best.pt` 와 TensorRT 엔진을 같은 데이터셋 split 에서 Ultralytics `val` 로 "
       "돌려 지표·속도를 비교한 문서. (`yolo26/eval.py` 자동 생성)")
    ap("")
    ap(f"- 데이터셋: `{args.data}` · split=**{args.split}**")
    ap(f"- 입력: imgsz={args.imgsz} · **batch={args.batch}**")
    if pt:
        ap(f"- `.pt`: `{pt['weights']}`")
    if trt:
        ap(f"- 엔진: `{trt['weights']}`")
    ap(f"- 생성일: {date.today().isoformat()}")
    ap("")

    if pt and trt:
        ap("## 정확도")
        ap("")
        ap("| 지표 | `.pt` | TRT engine | Δ (TRT − `.pt`) |")
        ap("|---|--:|--:|--:|")
        for name, key, nd, _ in METRIC_ROWS:
            ap(f"| {name} | {pt[key]:.{nd}f} | {trt[key]:.{nd}f} | {_delta(trt[key], pt[key], nd)} |")
        ap("")
        ap("## 속도 (per image)")
        ap("")
        ap("| 단계 | `.pt` | TRT engine | Δ | 배속 |")
        ap("|---|--:|--:|--:|--:|")
        for name, key, nd in SPEED_ROWS:
            spd = f"{pt[key] / trt[key]:.2f}×" if trt[key] > 0 and key in ("inf_ms", "total_ms") else ""
            ap(f"| {name} | {pt[key]:.{nd}f} | {trt[key]:.{nd}f} | {_delta(trt[key], pt[key], nd)} | {spd} |")
        ap("")
        ap("## 요약")
        ap("")
        d_map = trt["map50_95"] - pt["map50_95"]
        pct = (d_map / pt["map50_95"] * 100) if pt["map50_95"] else 0.0
        spd = pt["total_ms"] / trt["total_ms"] if trt["total_ms"] else 0.0
        tail = " — 사실상 동일" if abs(pct) < 1.0 else ""
        ap(f"- 엔진이 이미지당 **{spd:.2f}× 빠름** "
           f"({pt['total_ms']:.2f} → {trt['total_ms']:.2f} ms).")
        ap(f"- mAP50-95 는 **{d_map:+.4f}** ({pct:+.1f}%){tail}.")
    else:
        one = pt or trt
        ap(f"## {one['tag']} 단독 결과")
        ap("")
        ap("| 지표 | 값 |")
        ap("|---|--:|")
        for name, key, nd, _ in METRIC_ROWS:
            ap(f"| {name} | {one[key]:.{nd}f} |")
        for name, key, nd in SPEED_ROWS:
            ap(f"| {name} | {one[key]:.{nd}f} |")
    ap("")
    ap("> `val` 산출물(PR curve·혼동행렬 등)은 `runs/eval/<pt|trt>/` 에 저장된다.")
    ap("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> None:
    os.chdir(REPO_ROOT)                       # 어느 폴더에서 실행해도 상대경로 고정
    args = parse_args()

    if args.pt_only and args.engine_only:
        raise SystemExit("[에러] --pt-only 와 --engine-only 는 같이 못 씀")

    pt_res = trt_res = None

    if not args.engine_only:
        pt_path = Path(args.pt)
        if not pt_path.exists():
            raise SystemExit(f"[에러] .pt 없음: {pt_path.resolve()}")
        pt_res = evaluate(pt_path, args, "pt")

    if not args.pt_only:
        eng_path = Path(args.engine)
        if not eng_path.exists():
            msg = (f"[에러] 엔진 없음: {eng_path.resolve()}\n"
                   f"       먼저 빌드: python TensorRT/build_trt_engine.py")
            if pt_res is not None:
                print(msg + "\n       → .pt 결과만 저장한다.")
            else:
                raise SystemExit(msg)
        else:
            if args.batch != 1:
                print(f"[경고] batch={args.batch} — 정적 엔진(batch=1)이면 실패할 수 있음.")
            trt_res = evaluate(eng_path, args, "trt")

    print_table(pt_res, trt_res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(pt_res, trt_res, args), encoding="utf-8")
    print(f"\n[완료] {out}")


if __name__ == "__main__":
    main()
