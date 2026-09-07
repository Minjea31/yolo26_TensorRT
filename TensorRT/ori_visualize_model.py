#!/usr/bin/env python
"""
[원본 모델 그래프]  best.pt (Ultralytics YOLO 계열) 의 레이어 구조를 방향 그래프로
그려서 PNG 로 저장하는 스크립트.  (TensorRT 적용 후 그래프는 TRT_visualize_model.py)

실행 예:
    conda activate yolo            # torch 2.x + ultralytics 가 있는 환경
    python TensorRT/ori_visualize_model.py \
        --weights model/best.pt \
        --out     model/ori_structure.png

주의:
    - 이 저장소의 best.pt 는 repo 안의 커스텀 ultralytics 포크
      (yolo26/ultralytics) 로 학습된 체크포인트다. 그래서 스크립트가
      그 경로를 sys.path 맨 앞에 넣어 같은 포크로 언피클(unpickle)한다.
    - 그래프 렌더링에는 Graphviz 의 `dot` 실행파일이 필요하다.
      (없으면: sudo apt-get install graphviz)
      python `graphviz` 패키지가 있으면 그것을 우선 사용하고,
      없으면 `dot` 를 직접 호출한다. 둘 다 없으면 .dot 소스만 남긴다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

def find_repo_root() -> Path:
    """스크립트 위치와 현재 작업 디렉터리에서 위로 올라가며
    'yolo26/ultralytics/__init__.py' 를 가진 폴더(=repo 루트)를 찾는다.
    (TensorRT/ 같은 하위 폴더에서 실행해도 동작하도록)"""
    here = Path(__file__).resolve()
    seen = []
    for base in (here.parent, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if base in seen:
            continue
        seen.append(base)
        if (base / "yolo26" / "ultralytics" / "__init__.py").exists():
            return base
    return here.parent          # 못 찾으면 스크립트 상위를 그대로 사용


# repo 루트 (기본 경로 계산 및 포크 import 에 사용)
REPO_ROOT = find_repo_root()


# --------------------------------------------------------------------------- #
# 1. 인자 파싱
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO best.pt 구조 -> 그래프 이미지")
    # 경로 기본값은 repo 루트 기준 상대경로. main() 에서 os.chdir(REPO_ROOT) 하므로
    # 어느 폴더에서 실행해도 model/... 로 해석된다.
    p.add_argument("--weights", default="model/best.pt",
                   help="가중치 파일 경로 (.pt) — repo 루트 기준")
    p.add_argument("--out", default="model/ori_structure.png",
                   help="출력 이미지 경로 (.png / .svg / .pdf 확장자로 포맷 결정)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="레이어별 출력 shape 를 구하기 위한 더미 입력 크기")
    p.add_argument("--no-shapes", action="store_true",
                   help="forward 패스를 건너뛰고 shape 표기를 생략")
    p.add_argument("--rankdir", default="TB", choices=["TB", "LR"],
                   help="그래프 방향 (TB=위에서 아래, LR=왼쪽에서 오른쪽)")
    p.add_argument("--dpi", type=int, default=150, help="PNG 해상도(dpi)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 2. 모델 로드
# --------------------------------------------------------------------------- #
def add_local_fork_to_path() -> None:
    """repo 안의 커스텀 ultralytics 포크를 우선 import 하도록 sys.path 조정.
    포크를 `pip install -e yolo26/` 로 설치해 뒀으면 이 줄이 없어도 되지만,
    미설치 환경에서도 바로 돌아가도록 안전하게 넣어 둔다."""
    fork = Path("yolo26")          # main() 에서 os.chdir(REPO_ROOT) 완료 후 호출됨
    if (fork / "ultralytics" / "__init__.py").exists():
        sys.path.insert(0, str(fork))


def load_model(weights: str):
    """가중치를 읽어 nn.Module (DetectionModel 등) 을 돌려준다."""
    try:
        import torch  # noqa: F401  (아래 collect_layers 에서 사용)
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        sys.exit(
            "[에러] torch / ultralytics import 실패.\n"
            "       torch 2.x + ultralytics 가 설치된 conda 환경에서 실행하세요.\n"
            "       예) conda activate yolo && python TensorRT/ori_visualize_model.py\n"
            f"       (원인: {exc})"
        )

    import ultralytics
    print(f"[정보] ultralytics {ultralytics.__version__}  <-  "
          f"{os.path.relpath(ultralytics.__file__)}")

    net = YOLO(weights).model      # 실제 nn.Module
    net.eval().float()             # 추론 모드 + fp32 (CPU forward 용)
    return net


# --------------------------------------------------------------------------- #
# 3. 레이어 메타데이터 수집
# --------------------------------------------------------------------------- #
def collect_layers(net, imgsz: int, want_shapes: bool) -> list[dict]:
    """
    net.model (nn.Sequential) 을 순회하며 레이어별로
      i      : 레이어 인덱스
      f      : 입력을 받아오는 레이어 인덱스 (-1 = 바로 이전, list = 여러 입력)
      type   : 모듈 이름 (Conv, C3k2, Concat, SPPF, Detect ...)
      params : 파라미터 수
      shape  : forward 패스에서 관측된 출력 텐서 모양 (옵션)
    를 뽑는다. i / f / type / np 는 ultralytics 가 레이어에 직접 붙여둔 속성이다.
    """
    import torch

    seq = net.model
    shapes: dict[int, object] = {}

    # (옵션) 더미 입력을 한 번 흘려서 레이어별 출력 shape 를 hook 으로 캡처
    if want_shapes:
        handles = []

        def make_hook(idx: int):
            def hook(_module, _inp, out):
                if isinstance(out, (list, tuple)):
                    shapes[idx] = [tuple(o.shape) for o in out if hasattr(o, "shape")]
                elif hasattr(out, "shape"):
                    shapes[idx] = tuple(out.shape)
            return hook

        for idx, layer in enumerate(seq):
            handles.append(layer.register_forward_hook(make_hook(idx)))
        try:
            with torch.no_grad():
                net(torch.zeros(1, 3, imgsz, imgsz))
        except Exception as exc:
            print(f"[경고] forward 패스 실패 -> shape 표기 생략 ({exc})")
            shapes.clear()
        finally:
            for h in handles:      # hook 은 반드시 정리
                h.remove()

    layers: list[dict] = []
    for idx, layer in enumerate(seq):
        raw_type = getattr(layer, "type",
                           f"{layer.__class__.__module__}.{layer.__class__.__name__}")
        layers.append({
            "i": getattr(layer, "i", idx),
            "f": getattr(layer, "f", -1),
            "type": raw_type.split(".")[-1],                 # 'conv.Conv' -> 'Conv'
            "params": int(getattr(layer, "np",
                                  sum(p.numel() for p in layer.parameters()))),
            "shape": shapes.get(idx),
        })
    return layers


# --------------------------------------------------------------------------- #
# 4. 표시용 헬퍼
# --------------------------------------------------------------------------- #
def human(n: int) -> str:
    """1234567 -> '1.2M' 처럼 사람이 읽기 쉬운 문자열."""
    f = float(n)
    for unit in ("", "K", "M", "B"):
        if abs(f) < 1000:
            return f"{f:.0f}" if unit == "" else f"{f:.1f}{unit}"
        f /= 1000
    return f"{f:.1f}T"


def shape_str(s) -> str:
    if s is None:
        return ""
    if isinstance(s, list):                                  # 출력이 여러 개(Detect 등)
        return " / ".join("x".join(map(str, t)) for t in s)
    return "x".join(map(str, s))


# --------------------------------------------------------------------------- #
# 5. Graphviz DOT 문자열 생성
# --------------------------------------------------------------------------- #
def build_dot(layers: list[dict], backbone_len: int,
              rankdir: str, dpi: int, title: str) -> str:
    def node_color(L: dict) -> str:
        t = L["type"].lower()
        if L["i"] == len(layers) - 1 or "detect" in t:       # 마지막 = Detect head
            return "#ffd8a8"                                  # 주황
        if "concat" in t or "upsample" in t:
            return "#e5dbff"                                  # 라벤더 (연결/업샘플)
        if L["i"] < backbone_len:
            return "#d0ebff"                                  # 하늘 (backbone)
        return "#d3f9d8"                                      # 연두 (neck / head)

    lines = [
        "digraph model {",
        f"  graph [rankdir={rankdir}, dpi={dpi}, fontname=\"Helvetica\", "
        f"labelloc=\"t\", fontsize=20, label=\"{title}\"];",
        "  node  [fontname=\"Helvetica\", shape=box, style=\"rounded,filled\", fontsize=11];",
        "  edge  [fontname=\"Helvetica\", fontsize=9, color=\"#5b6b7c\"];",
    ]

    # 노드
    for L in layers:
        parts = [f'{L["i"]}  {L["type"]}']
        if L["params"]:
            parts.append(f'params {human(L["params"])}')
        ss = shape_str(L["shape"])
        if ss:
            parts.append(ss)
        label = "\\n".join(parts)
        lines.append(f'  n{L["i"]} [label="{label}", fillcolor="{node_color(L)}"];')

    # 엣지 (f -> i). -1 은 '바로 이전 레이어(i-1)' 로 해석.
    for L in layers:
        srcs = L["f"] if isinstance(L["f"], (list, tuple)) else [L["f"]]
        multi = len(srcs) > 1                                 # 여러 입력 = skip/concat
        for s in srcs:
            s = L["i"] - 1 if s == -1 else s
            if s < 0:
                continue
            style = ' [color="#e8590c", penwidth=2.0]' if multi else ""
            lines.append(f'  n{s} -> n{L["i"]}{style};')

    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. 렌더링 (.dot -> 이미지)
# --------------------------------------------------------------------------- #
def render(dot: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out.with_suffix(".dot")
    dot_path.write_text(dot, encoding="utf-8")
    fmt = out.suffix.lstrip(".") or "png"

    # (1) python `graphviz` 패키지가 있으면 그것으로
    try:
        import graphviz
        graphviz.Source(dot).render(out.with_suffix(""), format=fmt, cleanup=True)
        print(f"[완료] 이미지 저장: {out}")
        print(f"[완료] DOT 소스 : {dot_path}")
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"[경고] graphviz 패키지 렌더 실패 -> dot 실행파일로 재시도 ({exc})")

    # (2) 없으면 Graphviz `dot` 실행파일 직접 호출
    dot_bin = shutil.which("dot")
    if not dot_bin:
        sys.exit(
            "[에러] Graphviz 가 없습니다. `sudo apt-get install graphviz` 후 다시 실행하세요.\n"
            f"       (DOT 소스는 저장해 두었습니다: {dot_path})"
        )
    subprocess.run([dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out)], check=True)
    print(f"[완료] 이미지 저장: {out}")
    print(f"[완료] DOT 소스 : {dot_path}")


# --------------------------------------------------------------------------- #
# 7. 콘솔 요약표
# --------------------------------------------------------------------------- #
def print_table(layers: list[dict]) -> None:
    print()
    print(f'{"idx":>3}  {"type":<12}{"from":<14}{"params":>10}   shape')
    print("-" * 64)
    for L in layers:
        print(f'{L["i"]:>3}  {L["type"]:<12}{str(L["f"]):<14}'
              f'{human(L["params"]):>10}   {shape_str(L["shape"])}')
    total = sum(L["params"] for L in layers)
    print("-" * 64)
    print(f'{"":>3}  {"TOTAL":<12}{"":<14}{human(total):>10}')


# --------------------------------------------------------------------------- #
# 8. main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.chdir(REPO_ROOT)          # 이후 모든 상대경로는 repo 루트 기준
    args = parse_args()

    add_local_fork_to_path()
    net = load_model(args.weights)

    yaml = getattr(net, "yaml", {}) or {}
    backbone_len = len(yaml.get("backbone", []))              # backbone / head 경계
    names = getattr(net, "names", {})
    cls_list = list(names.values()) if isinstance(names, dict) else list(names)
    title = (f'{Path(args.weights).name}   |   task={getattr(net, "task", "?")}   '
             f'|   {len(cls_list)} class(es): {cls_list}')

    layers = collect_layers(net, args.imgsz, want_shapes=not args.no_shapes)
    print_table(layers)

    dot = build_dot(layers, backbone_len, args.rankdir, args.dpi, title)
    render(dot, Path(args.out))


if __name__ == "__main__":
    main()
