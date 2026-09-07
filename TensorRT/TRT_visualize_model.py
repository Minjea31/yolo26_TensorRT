#!/usr/bin/env python
"""
[TensorRT 적용 후 그래프]  build_trt_engine.py 가 덤프한 엔진 레이어 정보
(<engine>.engine.layers.json)를 읽어, TensorRT 가 fusion·정밀도 적용을 끝낸
실제 실행 그래프를 방향 그래프로 그려 이미지로 저장한다.

    원본(.pt) 구조   → TensorRT/ori_visualize_model.py
    TensorRT 후 구조 → 이 스크립트

두 가지 상세도:
    --level stage  (기본)  원본 model.N 단계 단위로 묶은 요약 그래프.
                           원본 그래프(24개 노드)와 1:1 비교하기 좋다.
    --level layer          TensorRT 커널 레이어를 전부 펼친 상세 그래프
                           (수백 노드 → 이미지가 매우 큼).

실행 예:
    conda activate yolo
    python TensorRT/TRT_visualize_model.py
    python TensorRT/TRT_visualize_model.py --level layer --hide noop,shape_call,reformat
    python TensorRT/TRT_visualize_model.py --engine-json model/best.engine.layers.json \
        --out model/TRT_structure.png --rankdir TB

입력 JSON 이 없으면 먼저:
    python TensorRT/build_trt_engine.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TensorRT 엔진 구조(fusion/정밀도 반영) -> 그래프 이미지",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 경로는 repo 루트 기준 상대경로. main() 에서 os.chdir(REPO_ROOT) 한다.
    p.add_argument("--engine-json", default="model/best.engine.layers.json",
                   help="build_trt_engine.py 가 만든 레이어 정보 JSON — repo 루트 기준")
    p.add_argument("--out", default="model/TRT_structure.png",
                   help="출력 이미지 (.png/.svg/.pdf)")
    p.add_argument("--level", choices=["stage", "layer"], default="stage",
                   help="stage=원본 단계 요약, layer=커널 레이어 전체")
    p.add_argument("--hide", default="noop,shape_call",
                   help="(layer 모드) 접어서 지나갈 LayerType 부분일치, 콤마구분")
    p.add_argument("--rankdir", default="TB", choices=["TB", "LR"])
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# --------------------------------------------------------------------------- #
def load_layers(path: Path):
    if not path.exists():
        sys.exit(f"[에러] 레이어 JSON 이 없습니다: {path}\n"
                 "       먼저 실행: python TensorRT/build_trt_engine.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    layers = data["Layers"] if isinstance(data, dict) else data
    bindings = data.get("Bindings", []) if isinstance(data, dict) else []
    return layers, bindings


ONNX_RE = re.compile(r"\[ONNX Layer:\s*([^\]\x1e]+)\]")
STAGE_RE = re.compile(r"model[._/]?(\d+)")
DTYPE_SHORT = {"Float": "f32", "Half": "f16", "Int8": "i8", "Int32": "i32",
               "Int64": "i64", "Bool": "b", "BFloat16": "bf16"}

# LayerType 계열 -> (짧은이름, 색)
FAMILIES = [
    (("caskconvolution", "cudnnconvolution"), "conv", "#d0ebff"),
    (("caskgemmconvolution", "caskgemm", "gemm", "matmul", "fullyconnected"), "gemm", "#e5dbff"),
    (("pointwise", "pwn", "elementwise", "activation"), "pointwise", "#d3f9d8"),
    (("pooling",), "pool", "#c3fae8"),
    (("reformat", "copy", "shuffle", "transpose"), "reformat", "#f1f3f5"),
    (("noop",), "noop", "#f1f3f5"),
    (("resize",), "resize", "#ffe8cc"),
    (("slice", "concat", "gather", "topk", "nms", "kgen", "myl"), "post", "#ffd8a8"),
]


def family(ltype: str):
    t = ltype.lower()
    for keys, short, col in FAMILIES:
        if any(k in t for k in keys):
            return short, col
    return ltype.lower(), "#ffffff"


def stage_of(layer: dict):
    for src in (layer.get("Name", ""), layer.get("Metadata", "")):
        m = STAGE_RE.search(src)
        if m:
            return m.group(1)
    return None


def out_info(layer: dict):
    outs = layer.get("Outputs", [])
    if not outs:
        return "", ""
    dims = "x".join(map(str, outs[0].get("Dimensions", [])))
    dt = DTYPE_SHORT.get(outs[0].get("Format/Datatype", ""), outs[0].get("Format/Datatype", ""))
    return dims, dt


def dot_escape(s: str) -> str:
    # 먼저 backslash 이스케이프 → 그 다음 실제 개행문자를 DOT 줄바꿈(\n)으로
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# --------------------------------------------------------------------------- #
# producer map (텐서 이름 -> 그 텐서를 출력하는 레이어 idx 들)
# --------------------------------------------------------------------------- #
def build_producers(layers):
    """텐서 이름 -> 그 이름을 출력하는 레이어 idx 리스트(오름차순).

    단일 dict[name]=i 로 접으면 안 되는 이유: TensorRT 는
      * zero-copy Concat fusion  — 하나의 concat 버퍼를 여러 레이어가 나눠 씀
        (예: `/model.15/Concat_output_0` 를 model.4 활성화와 model.14 reformat 이 둘 다 출력)
      * myelin 내부 스크래치 텐서 — 같은 이름(`__myln_*`, `__mye*`)을
        서로 다른 fused 서브그래프(model.10/22/23 attention·decode)가 재사용
    때문에 한 텐서 이름을 여러 레이어가 출력한다. dict 로 접으면 '마지막'
    생산자만 남아, 이른 소비자(model.5, model.7 …)가 뒤쪽(더 높은 stage)
    레이어에 연결돼 그래프가 거꾸로(위로) 그려진다.
    """
    prod = defaultdict(list)
    for i, L in enumerate(layers):
        for o in L.get("Outputs", []):
            prod[o["Name"]].append(i)
    return prod


def producer_for(prod, name, consumer_idx):
    """`name` 을 출력하는 레이어 중 소비자 바로 앞(가장 가까운 선행) 것.
    선행이 없으면 가장 이른 것. TensorRT 실행 순서상 '읽기 직전의 쓰기' 가
    실제 생산자이므로, 이름이 재사용돼도 올바른 producer 를 고른다."""
    cands = prod.get(name)
    if not cands:
        return None
    before = [p for p in cands if p < consumer_idx]
    return before[-1] if before else cands[0]


# --------------------------------------------------------------------------- #
# STAGE 레벨
# --------------------------------------------------------------------------- #
def graph_stage(layers, bindings):
    prod = build_producers(layers)

    def skey(k):
        return (0, int(k)) if (k and k.isdigit()) else (1, k or "zz")

    # 단계별 레이어 모으기
    members = defaultdict(list)
    for i, L in enumerate(layers):
        members[stage_of(L) or "post"].append(i)

    stages = sorted(members, key=skey)
    sidx = {s: n for n, s in enumerate(stages)}

    nodes = []
    for s in stages:
        mem = members[s]
        fam = Counter(family(layers[i]["LayerType"])[1] for i in mem)      # 색 빈도
        color = fam.most_common(1)[0][0]
        fam_names = Counter(family(layers[i]["LayerType"])[0] for i in mem)
        dcnt = Counter()
        for i in mem:
            _, dt = out_info(layers[i])
            if dt:
                dcnt[dt] += 1
        # 라벨은 좁게: 3줄, 각 줄 짧게 (세로 그래프가 옆으로 안 퍼지도록)
        ops = " ".join(f"{k}×{v}" for k, v in fam_names.most_common(3))
        prec = "/".join(k for k, _ in dcnt.most_common(2)) or "-"
        head = f"model.{s}" if s != "post" else "post-process"
        label = f"{head}\n{ops}\n{len(mem)} layers · {prec}"
        nodes.append(dict(id=sidx[s], stage=s, label=label, color=color))

    # 단계 간 엣지: cross-stage 텐서 의존
    real = set()
    dropped_back = 0
    for i, L in enumerate(layers):
        dst = stage_of(L) or "post"
        for inp in L.get("Inputs", []):
            p = producer_for(prod, inp["Name"], i)
            if p is None:
                continue
            src = stage_of(layers[p]) or "post"
            if src == dst:
                continue
            # YOLO 는 순방향: model.N 은 항상 더 낮은 N 에서만 입력을 받는다.
            # 그래도 역방향(높은 stage -> 낮은 stage)이 남으면 TRT 이름 충돌로
            # 잘못 이어진 가짜 엣지 -> 버린다 (그래프가 거꾸로 그려지던 원인).
            if src.isdigit() and dst.isdigit() and int(src) > int(dst):
                dropped_back += 1
                continue
            real.add((sidx[src], sidx[dst]))

    # 세로 정렬용 보이지 않는 척추(spine): 연속 단계 사이
    spine = [(n, n + 1) for n in range(len(stages) - 1)]
    sub = f"{len(stages)} stages"
    if dropped_back:
        sub += f" (역방향 가짜 엣지 {dropped_back}개 제거)"
    return nodes, sorted(real), spine, sub


# --------------------------------------------------------------------------- #
# LAYER 레벨
# --------------------------------------------------------------------------- #
def short_name(layer: dict, limit=40):
    name = layer.get("Name", "")
    if name.startswith("Reformatting"):
        s = "Reformat→" + name.split(" to ")[-1].strip().lstrip("/")
    else:
        toks = re.findall(r"model[._/][\w./]*", name)
        if toks:
            s = toks[0]
        else:
            onnx = ONNX_RE.findall(layer.get("Metadata", ""))
            s = onnx[0].lstrip("/") if onnx else name
    s = s.replace("/", ".")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def graph_layer(layers, bindings, hide):
    prod = build_producers(layers)

    def hidden(i):
        t = layers[i]["LayerType"].lower()
        return any(h and h in t for h in hide)

    def vis_ancestors(i, seen=None):
        seen = seen if seen is not None else set()
        res = set()
        for inp in layers[i].get("Inputs", []):
            p = producer_for(prod, inp["Name"], i)
            if p is None or p in seen:
                continue
            seen.add(p)
            res |= vis_ancestors(p, seen) if hidden(p) else {p}
        return res

    nodes = []
    for i, L in enumerate(layers):
        if hidden(i):
            continue
        short, col = family(L["LayerType"])
        dims, dt = out_info(L)
        tail = " ".join(x for x in (dims, dt) if x)
        label = f'{i}  {short_name(L)}\n{L["LayerType"]}'
        if tail:
            label += f"\n{tail}"
        nodes.append(dict(id=i, label=label, color=col, dtype=dt))

    keep = {n["id"] for n in nodes}
    edges = set()
    for i in keep:
        for a in vis_ancestors(i):
            if a in keep:
                edges.add((a, i))
    # layer 모드도 세로 정렬용 spine: id 오름차순(대체로 실행 순서)
    order = [n["id"] for n in nodes]
    spine = list(zip(order, order[1:]))
    return nodes, sorted(edges), spine, f"{len(keep)} layers (hidden: {sorted(hide)})"


# --------------------------------------------------------------------------- #
def build_dot(nodes, edges, spine, rankdir, dpi, title, layer_mode):
    """rankdir 기본 TB(=세로). spine 은 style=invis 인 연속-노드 엣지로,
    dot 이 그래프를 옆으로 퍼뜨리지 않고 세로 컬럼으로 쌓게 강제한다
    (원본 ori_visualize_model.py 와 같은 세로 형태)."""
    lines = [
        "digraph trt {",
        f'  graph [rankdir={rankdir}, dpi={dpi}, nodesep=0.25, ranksep=0.45, '
        f'fontname="Helvetica", labelloc="t", fontsize=20, label="{dot_escape(title)}"];',
        '  node  [fontname="Helvetica", shape=box, style="rounded,filled", '
        f'fontsize={9 if layer_mode else 11}];',
        '  edge  [fontname="Helvetica", fontsize=8, color="#5b6b7c"];',
    ]
    for n in nodes:
        pen = ""
        if layer_mode and n.get("dtype") == "f16":
            pen = ', color="#1971c2", penwidth=2'
        elif layer_mode and n.get("dtype") == "i8":
            pen = ', color="#e03131", penwidth=2'
        lines.append(f'  n{n["id"]} [label="{dot_escape(n["label"])}", '
                     f'fillcolor="{n["color"]}"{pen}];')
    real = set(edges)
    for a, b in edges:
        lines.append(f"  n{a} -> n{b};")
    for a, b in spine:                       # 세로 척추 (실제 엣지와 겹치면 생략)
        if (a, b) not in real:
            lines.append(f'  n{a} -> n{b} [style=invis, weight=10];')
    lines.append("}")
    return "\n".join(lines)


def render(dot: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out.with_suffix(".dot")
    dot_path.write_text(dot, encoding="utf-8")
    fmt = out.suffix.lstrip(".") or "png"
    try:
        import graphviz
        graphviz.Source(dot).render(out.with_suffix(""), format=fmt, cleanup=True)
        print(f"[완료] 이미지 저장: {out}")
        print(f"[완료] DOT 소스 : {dot_path}")
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"[경고] graphviz 패키지 렌더 실패 -> dot 재시도 ({exc})")
    dot_bin = shutil.which("dot")
    if not dot_bin:
        sys.exit(f"[에러] Graphviz 없음. `sudo apt-get install graphviz`.\n       DOT: {dot_path}")
    subprocess.run([dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out)], check=True)
    print(f"[완료] 이미지 저장: {out}")
    print(f"[완료] DOT 소스 : {dot_path}")


# --------------------------------------------------------------------------- #
def main():
    os.chdir(REPO_ROOT)          # 이후 모든 상대경로는 repo 루트 기준
    args = parse_args()
    jpath = Path(args.engine_json)
    layers, bindings = load_layers(jpath)

    tcnt = Counter(L["LayerType"] for L in layers)
    dcnt = Counter()
    for L in layers:
        _, dt = out_info(L)
        if dt:
            dcnt[dt] += 1
    print(f"엔진 레이어 : 총 {len(layers)}개")
    print("LayerType   :", ", ".join(f"{k}×{v}" for k, v in tcnt.most_common()))
    print("정밀도(출력):", ", ".join(f"{k}×{v}" for k, v in dcnt.most_common()))

    if args.level == "stage":
        nodes, edges, spine, sub = graph_stage(layers, bindings)
    else:
        hide = {h.strip().lower() for h in args.hide.split(",") if h.strip()}
        nodes, edges, spine, sub = graph_layer(layers, bindings, hide)
    print(f"그래프({args.level}) : {sub},  edges {len(edges)}")

    prec = "+".join(k for k, _ in dcnt.most_common(3))
    title = (f"{jpath.stem}  |  TensorRT engine ({args.level})  |  "
             f"{len(layers)} kernel layers  |  precision: {prec}  |  IO: {bindings}")
    dot = build_dot(nodes, edges, spine, args.rankdir, args.dpi, title,
                    layer_mode=(args.level == "layer"))
    render(dot, Path(args.out))


if __name__ == "__main__":
    main()
