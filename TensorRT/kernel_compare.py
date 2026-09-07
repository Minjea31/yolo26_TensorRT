#!/usr/bin/env python
"""
[.pt 구조 vs TensorRT 구조 — 커널 사용 위치 비교 그래프]

커널은 결국 "메모리에서 값을 읽어 계산하는" GPU 실행 단위다. 이 스크립트는
    · 원본 best.pt 가 **어디서 어떤 연산(커널)을 하는지**
    · TensorRT 가 fusion 한 뒤 **실제 커널이 어느 단계에서 몇 개 도는지**
를 같은 `model.N` 축으로 나란히 놓은 **한 장의 좌우 비교 그래프**로 만든다.

  왼쪽  : .pt 24개 모듈(model.N) 체인 + skip 연결. 각 모듈 안 leaf 연산 종류.
          (eager 실행에선 Conv2d·BN·SiLU·MaxPool·cat 하나하나가 커널 실행 지점)
  오른쪽: TensorRT 엔진 커널을 같은 model.N 으로 되묶은 구조 + skip.
          단계별 커널 수 · 종류(conv/gemm/pointwise/reformat…) · 정밀도.
  가운데: model.N 을 잇는 점선 = "leaf N개 → 커널 M개"
          초록=줄거나 그대로 · 주황=reformat 삽입으로 늘어남 · 빨강=커널 0(흡수)

관련 스크립트:
    .pt 구조만        → TensorRT/ori_visualize_model.py
    TRT 구조만        → TensorRT/TRT_visualize_model.py
    숫자 표(.pt↔TRT)  → TensorRT/compare_pt_trt.py
    좌우 비교 그래프   → 이 스크립트

실행:
    conda activate yolo
    python TensorRT/kernel_compare.py
    python TensorRT/kernel_compare.py --out model/kernel_compare.png --rankdir TB

입력 JSON 이 없으면 먼저:
    python TensorRT/build_trt_engine.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 형제 스크립트 import 용
from ori_visualize_model import (  # noqa: E402
    add_local_fork_to_path, collect_layers, find_repo_root, human, load_model, shape_str,
)
from TRT_visualize_model import (  # noqa: E402
    FAMILIES, build_producers, dot_escape, load_layers, producer_for, render,
)
from compare_pt_trt import aggregate_trt, kernel_stage  # noqa: E402

REPO_ROOT = find_repo_root()
SHORT_COLOR = {short: col for _keys, short, col in FAMILIES}
LEAF_ABBR = {"Conv2d": "Conv", "BatchNorm2d": "BN", "MaxPool2d": "Pool",
             "Upsample": "Up", "ConvTranspose2d": "ConvT", "Identity": "Id"}


def _lbl(*parts: str) -> str:
    """비어있지 않은 조각들을 실제 개행으로 잇는다 (dot_escape 가 \\n 으로 변환)."""
    return "\n".join(p for p in parts if p)


def leaf_str(cnt: Counter, top: int = 4) -> str:
    return " ".join(f"{LEAF_ABBR.get(k, k)}×{v}" for k, v in cnt.most_common(top))


def leaf_multiset(module) -> Counter:
    """자식 없는 leaf 서브모듈을 **참조 횟수**로 센다.

    Ultralytics `Conv` 는 `default_act = nn.SiLU()` 가 클래스 속성이라 모든 Conv 가
    같은 SiLU **객체 하나**를 공유한다. `module.modules()` 는 객체를 dedup 하므로
    C3k2 안에 Conv 가 9개여도 `SiLU×1` 로 세진다. 여기선 트리를 직접 내려가며
    도달할 때마다 세서 `SiLU×9` 처럼 실제 적용 횟수에 맞춘다."""
    c: Counter = Counter()

    def rec(m):
        kids = list(m.children())
        if not kids:
            c[type(m).__name__] += 1
        else:
            for k in kids:
                rec(k)

    rec(module)
    return c


# --------------------------------------------------------------------------- #
_LAUNCH_NAMES = {"cudaLaunchKernel", "cudaLaunchKernelExC",
                 "cudaLaunchCooperativeKernel"}

# aten op 이름 -> 짧은 계열명 (그래프 라벨용)
_OP_TABLE = [
    (("convolution", "conv2d", "conv1d", "conv3d", "cudnn_convolution",
      "mkldnn_convolution", "slow_conv", "thnn_conv"), "conv"),
    (("batch_norm", "cudnn_batch_norm", "native_batch_norm",
      "_native_batch_norm"), "bn"),
    (("silu", "hardswish", "gelu", "relu", "leaky_relu", "elu"), "silu"),
    (("sigmoid",), "sigmoid"),
    (("mul", "mul_"), "mul"),
    (("add", "add_", "__iadd__", "sub", "sub_"), "add"),
    (("cat", "stack", "_cat"), "cat"),
    (("max_pool2d", "max_pool2d_with_indices", "avg_pool2d",
      "adaptive_avg_pool2d"), "pool"),
    (("upsample_nearest2d", "upsample_bilinear2d", "interpolate"), "resize"),
    (("matmul", "mm", "bmm", "linear", "addmm", "einsum"), "matmul"),
    (("softmax", "_softmax", "log_softmax"), "softmax"),
    (("slice", "split", "chunk", "narrow", "select", "unbind", "index",
      "gather", "scatter"), "slice"),
    (("contiguous", "clone", "copy_", "_to_copy", "to"), "copy"),
    (("transpose", "permute", "reshape", "view", "flatten", "unsqueeze",
      "squeeze", "expand", "as_strided"), "shape"),
]


def _op_short(aten_name: str) -> str:
    n = aten_name.replace("aten::", "").lstrip("_")
    for keys, short in _OP_TABLE:
        if n in keys or any(n.startswith(k) for k in keys):
            return short
    return n.split("_")[0] or "op"


def _nearest_aten(ev) -> str:
    cur = ev
    while cur is not None:
        if cur.name.startswith("aten::"):
            return cur.name
        cur = getattr(cur, "cpu_parent", None)
    return ev.name


def kernel_op_str(cnt: Counter, top: int = 6) -> str:
    s = " ".join(f"{k}×{v}" for k, v in cnt.most_common(top))
    if len(cnt) > top:
        s += f" +{sum(v for _, v in cnt.most_common()[top:])}"
    return s


def stage_kernel_launches(net, imgsz: int, warmup: int = 3):
    """eager `.pt` 를 CUDA 에서 실제로 돌려, `model.N` 단계별
    **실제 CUDA 커널 런치**(`cudaLaunchKernel` 이벤트)를 세고, 각 런치를 호출한
    aten op(`conv`/`bn`/`silu`/`add`/`cat` …)별로 분해한다.
    반환: `{stage_idx: Counter(op_short -> launch 수)}`. CUDA 없거나 실패하면 None.
    leaf 모듈 수가 아니라 진짜 호출 횟수 (autotuning 으로 실행마다 ±몇 개 흔들림)."""
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile, record_function
    except Exception:
        return None
    if not torch.cuda.is_available():
        print("[정보] CUDA 없음 → eager 커널 런치 대신 leaf 수 사용")
        return None

    net = net.eval().cuda()
    hooks = []

    class _Scope:
        def __init__(self, i):
            self.i, self.cm = i, None

        def pre(self, _m, _inp):
            self.cm = record_function(f"kc_stage_{self.i}")
            self.cm.__enter__()

        def post(self, _m, _inp, _out):
            if self.cm is not None:
                self.cm.__exit__(None, None, None)
                self.cm = None

    for i, m in enumerate(net.model):
        sc = _Scope(i)
        hooks.append(m.register_forward_pre_hook(sc.pre))
        hooks.append(m.register_forward_hook(sc.post))

    try:
        x = torch.zeros(1, 3, imgsz, imgsz, device="cuda")
        with torch.no_grad():
            for _ in range(warmup):
                net(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                net(x)
            torch.cuda.synchronize()
    except Exception as exc:
        print(f"[경고] eager 커널 프로파일 실패 → leaf 수로 대체 ({exc})")
        return None
    finally:
        for h in hooks:
            h.remove()
        net.cpu()

    def _collect(ev, ops: Counter):
        for c in ev.cpu_children:
            if c.name in _LAUNCH_NAMES:
                ops[_op_short(_nearest_aten(ev))] += 1
            else:
                _collect(c, ops)

    per: dict[int, Counter] = {}
    for ev in prof.events():
        if ev.name.startswith("kc_stage_"):
            i = int(ev.name.rsplit("_", 1)[1])
            per.setdefault(i, Counter())
            _collect(ev, per[i])
    return per


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=".pt 구조 vs TensorRT 구조 — 커널 사용 위치 좌우 비교 그래프",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default="model/best.pt",
                   help="원본 가중치 (.pt) — repo 루트 기준")
    p.add_argument("--engine-json", default="model/best.engine.layers.json",
                   help="build_trt_engine.py 가 만든 레이어 정보 JSON")
    p.add_argument("--out", default="model/kernel_compare.png",
                   help="출력 이미지 (.png/.svg/.pdf)")
    p.add_argument("--imgsz", type=int, default=640, help="shape 관측용 더미 입력 크기")
    p.add_argument("--no-shapes", action="store_true",
                   help="forward 패스를 건너뛰고 .pt 출력 shape 표기를 생략")
    p.add_argument("--no-profile", action="store_true",
                   help="eager CUDA 커널 런치 프로파일을 건너뛰고 leaf 서브모듈 수(어림값) 사용")
    p.add_argument("--rankdir", default="TB", choices=["TB", "LR"])
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# --------------------------------------------------------------------------- #
def pt_color(L: dict, backbone_len: int, n: int) -> str:
    t = L["type"].lower()
    if L["i"] == n - 1 or "detect" in t:
        return "#ffd8a8"                       # detect head
    if "concat" in t or "upsample" in t:
        return "#e5dbff"                       # 연결/업샘플
    if L["i"] < backbone_len:
        return "#d0ebff"                       # backbone
    return "#d3f9d8"                           # neck/head


def trt_color(fam: Counter) -> str:
    for k, _ in fam.most_common():
        if k in SHORT_COLOR:
            return SHORT_COLOR[k]
    return "#ffffff"


# --------------------------------------------------------------------------- #
def pt_structure_edges(layers_pt: list[dict]):
    """(src_i, dst_i, is_multi_input) — ori_visualize_model.build_dot 과 동일 규칙."""
    edges = []
    for L in layers_pt:
        i = L["i"]
        srcs = L["f"] if isinstance(L["f"], (list, tuple)) else [L["f"]]
        multi = len(srcs) > 1
        for s in srcs:
            s = i - 1 if s == -1 else s
            if s < 0:
                continue
            edges.append((s, i, multi))
    return edges


def trt_structure_edges(layers_trt: list[dict]):
    """단계(model.N) 간 텐서 의존 엣지. compare_pt_trt.kernel_stage(다수결) +
    TRT_visualize_model 의 producer 보정(producer_for) 사용, 순방향만."""
    prod = build_producers(layers_trt)
    edges = set()
    for i, L in enumerate(layers_trt):
        dst = kernel_stage(L) or "post"
        for inp in L.get("Inputs", []):
            p = producer_for(prod, inp["Name"], i)
            if p is None:
                continue
            src = kernel_stage(layers_trt[p]) or "post"
            if src == dst:
                continue
            if src.isdigit() and dst.isdigit() and int(src) > int(dst):
                continue                        # 역방향 = 이름 충돌 가짜 엣지
            edges.add((src, dst))
    return edges


# --------------------------------------------------------------------------- #
def build_dot(layers_pt, agg, pt_edges, trt_edges, backbone_len,
              rankdir, dpi, title) -> str:
    n = len(layers_pt)
    have_post = None in agg
    have_launch = any(L.get("launch") is not None for L in layers_pt)

    def left_n(L):
        return L["launch"] if have_launch and L.get("launch") is not None else L["leaf"]

    lh = "eager: 실제 CUDA 커널 런치 수" if have_launch else "eager: leaf 서브모듈 수(어림값)"
    lines = [
        "digraph kernel_compare {",
        f'  graph [rankdir={rankdir}, newrank=true, dpi={dpi}, nodesep=0.30, '
        f'ranksep=0.60, fontname="Helvetica", labelloc="t", fontsize=16, '
        f'label="{dot_escape(title)}"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8, color="#5b6b7c"];',
        f'  PT_HEAD  [label="{dot_escape(_lbl("■ .pt 원본", lh))}", '
        'shape=plaintext, fontsize=13, fontcolor="#495057"];',
        f'  TRT_HEAD [label="{dot_escape(_lbl("■ TensorRT engine (fusion 후)", "남은 실제 커널"))}", '
        'shape=plaintext, fontsize=13, fontcolor="#495057"];',
        '  { rank=same; PT_HEAD; TRT_HEAD; }',
        '  PT_HEAD  -> L0 [style=invis];',
        '  TRT_HEAD -> R0 [style=invis];',
    ]

    # ---- 노드 ----
    unit = "kernel launches" if have_launch else "leaf 모듈"
    for L in layers_pt:
        i = L["i"]
        shape = shape_str(L["shape"]) or ""
        ops = L.get("launch_ops")
        if have_launch and ops:
            breakdown = kernel_op_str(ops)             # 실측: conv×13 bn×9 silu×9 add×2 cat×2
        else:
            breakdown = f'[{leaf_str(L["leaf_types"])}]'
        lab = _lbl(f'{i}  {L["type"]}' + (f'  ·  {shape}' if shape else ""),
                   f'{left_n(L)} {unit}',
                   breakdown)
        lines.append(f'  L{i} [label="{dot_escape(lab)}", '
                     f'fillcolor="{pt_color(L, backbone_len, n)}"];')

    for L in layers_pt:
        i = L["i"]
        a = agg.get(str(i))
        if not a or a["n"] == 0:
            lines.append(f'  R{i} [label="{i}  (fusion 흡수 · 커널 0)", '
                         f'fillcolor="#f1f3f5", fontcolor="#adb5bd", '
                         f'style="rounded,filled,dashed"];')
            continue
        prec = "/".join(k for k, _ in a["dt"].most_common(2)) or "-"
        lab = _lbl(f'{i}  {a["n"]} kernels', leaf_str(a["fam"]), prec)
        lines.append(f'  R{i} [label="{dot_escape(lab)}", '
                     f'fillcolor="{trt_color(a["fam"])}"];')

    if have_post:
        a = agg[None]
        lab = _lbl("post-process", f'{a["n"]} kernels', leaf_str(a["fam"]))
        lines.append(f'  Rpost [label="{dot_escape(lab)}", fillcolor="#ffffff"];')

    # ---- 세로 정렬용 invisible spine (양 컬럼 곧게) ----
    for i in range(n - 1):
        lines.append(f'  L{i} -> L{i + 1} [style=invis, weight=100];')
        lines.append(f'  R{i} -> R{i + 1} [style=invis, weight=100];')
    if have_post:
        lines.append(f'  R{n - 1} -> Rpost [style=invis, weight=100];')

    # ---- .pt 구조 엣지 (skip 은 constraint=false 로 spine 방해 안 함) ----
    for sidx, didx, multi in pt_edges:
        consecutive = didx == sidx + 1
        attrs = []
        if multi:
            attrs.append('color="#e8590c"')
            attrs.append("penwidth=1.8")
        if not consecutive:
            attrs.append("constraint=false")
        a = f' [{", ".join(attrs)}]' if attrs else ""
        lines.append(f'  L{sidx} -> L{didx}{a};')

    # ---- TRT 구조 엣지 ----
    def rn(stage):
        return "Rpost" if stage == "post" else f"R{stage}"
    for src, dst in sorted(trt_edges):
        if dst == "post" and not have_post:
            continue
        consecutive = src.isdigit() and dst.isdigit() and int(dst) == int(src) + 1
        a = "" if consecutive else " [constraint=false]"
        lines.append(f'  {rn(src)} -> {rn(dst)}{a};')

    # ---- trace 로 못 이은 자리 보강 (점선) ----
    # Concat zero-copy fusion 때문에, 바로 앞 1x1 Conv(model.17/20 등)가 concat
    # 버퍼에 직접 써버려 텐서 이름으로는 다음 단계와 안 이어진다. 실제로는
    # 연결돼 있으므로 다음 단계로 점선을 그어 준다.
    out_stages = {s for s, _ in trt_edges if s.isdigit()}
    for i in range(n - 1):
        if str(i) not in out_stages:
            lines.append(f'  R{i} -> R{i + 1} [style=dotted, color="#ced4da", '
                         f'constraint=false, label="추정", fontsize=7, fontcolor="#adb5bd"];')
    if have_post and "post" not in {d for _, d in trt_edges}:
        lines.append(f'  R{n - 1} -> Rpost [style=dotted, color="#ced4da", constraint=false];')

    # ---- 행 정렬 + 가운데 점선 (eager 커널 런치 수 → TRT 커널 수) ----
    for L in layers_pt:
        i = L["i"]
        lines.append(f'  {{ rank=same; L{i}; R{i}; }}')
        k = agg.get(str(i), {}).get("n", 0)
        b = left_n(L)
        if k == 0:
            ecol = "#e03131"
        elif k < b:
            ecol = "#2f9e44"                    # 줄어듦 (fusion)
        elif k > b:
            ecol = "#f08c00"                    # 늘어남 (reformat 삽입)
        else:
            ecol = "#868e96"
        elab = f"{b} → {k}"
        lines.append(f'  L{i} -> R{i} [constraint=false, style=dashed, penwidth=1.2, '
                     f'color="{ecol}", fontcolor="{ecol}", label="{dot_escape(elab)}"];')

    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def print_summary(layers_pt, agg, have_launch):
    unit = "eager CUDA 커널 런치" if have_launch else "leaf 서브모듈(어림값)"

    def left_of(L):
        return L.get("launch") if have_launch and L.get("launch") is not None else L["leaf"]

    left_total = sum(left_of(L) for L in layers_pt)
    mapped = sum(a["n"] for s, a in agg.items() if s is not None)
    post = agg.get(None, {}).get("n", 0)
    print(f"  {unit} {left_total}회  →  TRT 커널 {mapped}개 (+ post {post})")
    diff = mapped - left_total
    print(f"  → 전체 {'−' if diff < 0 else '+'}{abs(diff)}  "
          f"({'감소' if diff < 0 else '증가'})")
    if have_launch:
        tot = Counter()
        for L in layers_pt:
            if L.get("launch_ops"):
                tot.update(L["launch_ops"])
        print(f"  eager 런치 분해: {kernel_op_str(tot, top=10)}")
        print("  (nn.Module 27개여도 런치 35개인 이유: cat·add 는 모듈 아님, "
              "conv 1개가 알고리즘 따라 커널 2~3개)")
    grew = [i for i in range(len(layers_pt))
            if agg.get(str(i), {}).get("n", 0) > left_of(layers_pt[i])]
    print(f"  커널이 늘어난 단계(reformat 삽입): "
          f"{', '.join('model.' + str(i) for i in grew) or '없음'}")


# --------------------------------------------------------------------------- #
def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()

    add_local_fork_to_path()
    net = load_model(args.weights)
    yaml = getattr(net, "yaml", {}) or {}
    backbone_len = len(yaml.get("backbone", [])) or 11
    layers_pt = collect_layers(net, args.imgsz, want_shapes=not args.no_shapes)
    for i, m in enumerate(net.model):
        cnt = leaf_multiset(m)                 # 참조 횟수 (공유 SiLU 도 적용마다 셈)
        layers_pt[i]["leaf"] = sum(cnt.values())
        layers_pt[i]["leaf_types"] = cnt

    launches = None if args.no_profile else stage_kernel_launches(net, args.imgsz)
    for i in range(len(layers_pt)):
        ops = launches.get(i) if launches else None
        layers_pt[i]["launch_ops"] = ops
        layers_pt[i]["launch"] = sum(ops.values()) if ops is not None else None
    have_launch = launches is not None

    layers_trt, bindings = load_layers(Path(args.engine_json))
    agg = aggregate_trt(layers_trt)

    pt_edges = pt_structure_edges(layers_pt)
    trt_edges = trt_structure_edges(layers_trt)

    left_total = sum((L["launch"] if have_launch and L["launch"] is not None else L["leaf"])
                     for L in layers_pt)
    mapped = sum(a["n"] for s, a in agg.items() if s is not None)
    left_desc = ("eager CUDA 커널 런치" if have_launch else "eager leaf 연산(어림값)")
    title = _lbl(
        f"{Path(args.weights).stem}  —  .pt 구조 vs TensorRT 구조 (커널 사용 위치)",
        f"{left_desc} {left_total}회    ↔    "
        f"TRT {len(layers_trt)} 커널 (매핑 {mapped} + post {agg.get(None, {}).get('n', 0)})",
        "왼쪽 노드 = eager 런치 수 + 호출한 aten op 분해(conv/bn/silu/add/cat)   ·   "
        "오른쪽 노드 = 남은 TRT 커널 종류   ·   "
        "점선 = 런치 수 → TRT 커널 수  (초록 감소 · 주황 증가)",
    )

    dot = build_dot(layers_pt, agg, pt_edges, trt_edges, backbone_len,
                    args.rankdir, args.dpi, title)
    render(dot, Path(args.out))
    print_summary(layers_pt, agg, have_launch)


if __name__ == "__main__":
    main()
