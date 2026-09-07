# TensorRT/kernel_compare.py

**[.pt 구조 vs TensorRT 구조 — 커널 사용 위치 비교 그래프]**
커널은 "메모리에서 값을 읽어 계산하는" GPU 실행 단위다. 이 스크립트는
**원본 `best.pt` 가 어디서 연산을 하는지** 와 **TensorRT fusion 후 실제 커널이 어느
단계에서 몇 개 도는지** 를 같은 `model.N` 축으로 좌우에 놓은 **한 장의 비교 그래프**로
만든다 → `model/kernel_compare.png` (+ `.dot`).

| | 스크립트 |
|---|---|
| `.pt` 구조만 | [`ori_visualize_model.py`](ori_visualize_model.md) |
| TRT 구조만 | [`TRT_visualize_model.py`](TRT_visualize_model.md) |
| 숫자 표(`.pt`↔TRT) | [`compare_pt_trt.py`](compare_pt_trt.md) |
| **좌우 비교 그래프** | 이 스크립트 |

## 그래프 읽는 법

```
  ■ .pt 원본 (eager)                          ■ TensorRT engine (fusion 후)
  ┌──────────────────────────────┐  35 → 26  ┌────────────────────────────┐
  │ 6  C3k2 · 1x128x40x40        │ ┄┄┄┄┄┄▶ │ 6  26 kernels              │
  │ 35 kernel launches           │ (초록=감소) │ pointwise×9 conv×7 noop×6 … │
  │ conv×13 bn×9 silu×9 add×2 …  │           │ f32                        │
  └──────────────────────────────┘           └────────────────────────────┘
        │ (체인 + skip)                             │ (체인 + skip)
        ▼                                           ▼
```

- **왼쪽 노드** = `.pt` 모듈.
  - `N kernel launches` = **eager 실행 시 실제로 호출되는 CUDA 커널 수**
    (`torch.profiler` 로 `cudaLaunchKernel` 이벤트를 단계별로 카운트).
  - 3번째 줄 `conv×13 bn×9 silu×9 add×2 cat×2 …` = 그 런치들을 **호출한 aten op**
    별 분해 (합 = 런치 수). `nn.Module` 개수가 아니다 — `torch.cat`·`+` 처럼
    모듈이 아닌 연산도 잡히고, `conv` 하나가 알고리즘에 따라 커널 2~3개로 쪼개진다.
  - `+N` 은 상위 6개 외 나머지 합. `--no-profile` 이거나 CUDA 없으면 대신
    `[Conv×9 BN×9 SiLU×9]` (모듈 참조 횟수)를 보여준다.
- **오른쪽 노드** = 그 `model.N` 으로 매핑된 **실제 TRT 커널** 수 + 종류
  (`conv`/`gemm`/`pointwise`/`pool`/`reformat`·`noop`/`resize`/`post`) + 출력 정밀도.
- **가운데 점선** = `eager 런치 수 → TRT 커널 수`.
  **초록 = 줄어듦(fusion)**, **주황 = 늘어남(reformat 삽입)**, 빨강(커널 0), 회색(동일).
- **컬럼 안 실선** = 구조(체인 + skip 연결). `.pt` 의 skip 은 주황(다중 입력),
  TRT 는 회색 곡선.
- **컬럼 안 점선(연회색 `추정`)** = 텐서 이름으로는 못 이었지만 구조상 이어지는 자리
  (아래 "왜 TRT 쪽 일부가 안 이어지나" 참고).
- **행 정렬**: `model.N` 이 좌우 같은 높이 → 한 줄로 비교.

### 왜 TRT 쪽 일부가 다음 단계로 안 이어지나

`model.17`·`model.20`(1×1 Conv) 처럼 **fused Concat 바로 앞 단계**는 텐서 이름 추적으로는
다음 단계와 안 붙는다. TensorRT 가 neck 의 `Concat`(model.18/21)을 **zero-copy 버퍼**로
없애면서:

- 그 Concat 의 **긴 skip 입력** 쪽만 `reformat` 커널 1개로 남기고
  (model.18 의 reformat 은 model.13 갈래, model.21 은 model.10 갈래를 나름),
- **바로 앞 Conv**(model.17/20)의 출력은 concat 버퍼에 **직접 써버린다** → 텐서 이름이
  바뀌어 `model.17 → model.18` 링크가 추적에서 사라진다.

엔진에선 실제로 연결돼 있고 데이터도 흐른다. 재구성이 못 볼 뿐이라, 이 스크립트는
**"추정" 점선**으로 다음 단계에 이어 준다. `post-process`(`shape_call` 스텁)도 JSON 에
텐서 I/O 가 없어 엣지가 안 붙어서 `model.23` 에서 점선으로 잇는다.
(같은 원인이 `TRT_visualize_model.py` 의 "거꾸로 그려지던" 문제였다.)

### 무엇이 보이나

| | eager 런치 → TRT 커널 | 그래프에서 |
|---|---|---|
| **전체** | **493 → 316 (−180, 감소)** | 대부분 초록 점선 |
| `model.23` Detect | `171 → 66` | decode/NMS 융합, 가장 큰 감소. `post×7`, `i64` 여기만 |
| `model.6/8/13/16` C3k2 | `35~39 → 24~26` | Conv+BN+SiLU+add 융합 |
| `model.9` SPPF | `10 → 13` (주황) | MaxPool 앞뒤로 `reformat`/`noop` 삽입 → 늘어남 |
| `model.10` C2PSA | `26 → 30` (주황) | attention → myelin blob + `reformat` |
| `model.12/15/18/21` Concat | `1 → 1` | zero-copy, `reformat` 1개만 |

구조적으로 보이는 것: `BN` 은 conv 에 fold 되어 오른쪽에서 사라지고, `SiLU` 는
적용마다 `pointwise` 커널로 나뉘며, 원본에 없던 `reformat`/`noop`(레이아웃 변환)이 삽입된다.

### 왜 `nn.Module` 수 ≠ 커널 런치 수 (model.6 예: 모듈 27 vs 런치 35)

| 항목 | 수 | 메모 |
|---|--:|---|
| `conv` | 13 | Conv2d 는 9개인데, 3×3 conv 4개가 커널 2개씩(cuDNN: 변환 커널 + GEMM) → 9+4 |
| `bn` | 9 | BatchNorm2d 9개, 1커널씩 |
| `silu` | 9 | SiLU 9번, 1커널씩 |
| `add` | 2 | bottleneck residual `x + …` — `nn.Module` 이 아님 |
| `cat` | 2 | `torch.cat` — C3k2 1 + 내부 C3k 1. 역시 모듈 아님 |
| **합** | **35** | |

`[Conv×9 BN×9 SiLU×9]`(모듈, 27) 이 아니라 이 분해(35)가 라벨에 나온다.

> **"커널이 늘어난 거 아니냐"** 는 오해는 왼쪽을 `nn.Module` 수(SiLU 공유객체 1회로 셈,
> `cat`/`+` 안 셈, conv 1개=커널 1개로 가정 — 과소)로 볼 때 생긴다. 실제
> `cudaLaunchKernel` 횟수로 세면 **493 → 316 으로 줄어든다.** 늘어난 건 model.9/10 둘뿐.

## 입력

- `model/best.pt` — 원본 가중치 (torch + ultralytics 필요, `yolo` 환경)
- `model/best.engine.layers.json` — `build_trt_engine.py` 산출물. 없으면 먼저:
  ```bash
  python TensorRT/build_trt_engine.py
  ```
- **CUDA GPU** — 왼쪽 "실제 커널 런치 수" 프로파일용. 없으면 자동으로 leaf 서브모듈
  어림값으로 대체(그래프에 `leaf` 로 표기)되고 `--no-profile` 와 동일.
- Graphviz `dot` (또는 python `graphviz`) — 렌더용.

## 사용법

```bash
conda activate yolo

python TensorRT/kernel_compare.py                       # model/kernel_compare.png (CUDA 프로파일 포함)
python TensorRT/kernel_compare.py --no-profile          # 프로파일 생략, leaf 어림값 사용(빠름)
python TensorRT/kernel_compare.py --no-shapes           # forward shape 관측 생략
python TensorRT/kernel_compare.py --out model/kernel_compare.svg --rankdir TB
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `model/best.pt` | 원본 가중치 |
| `--engine-json` | `model/best.engine.layers.json` | 엔진 레이어 정보 JSON |
| `--out` | `model/kernel_compare.png` | 출력 이미지 (.png/.svg/.pdf) |
| `--imgsz` | `640` | shape 관측·프로파일용 더미 입력 크기 |
| `--no-shapes` | off | forward 패스 생략, `.pt` 출력 shape 표기 생략 |
| `--no-profile` | off | eager CUDA 커널 런치 프로파일 생략 → 왼쪽을 leaf 어림값으로 |
| `--rankdir` | `TB` | `TB` 세로 / `LR` 가로 |
| `--dpi` | `150` | 해상도 |

## 동작 원리

| 함수 | 역할 |
|---|---|
| `collect_layers()` (import) | `best.pt` → 모듈별 `type`/`f`(입력 연결)/`params`/출력 shape |
| `main()` leaf 수집 | `net.model[i].modules()` 중 자식 없는 것 = leaf 연산. 종류별 `Counter` (fallback 용) |
| `stage_kernel_launches()` | eager `.pt` 를 CUDA 에서 `torch.profiler` 로 돌려, 각 `model.N` forward 안에서 발생한 `cudaLaunchKernel` 이벤트 수를 카운트 = **실제 커널 호출 횟수**. warmup 3회 후 측정. CUDA 없으면 `None` |
| `aggregate_trt()` / `kernel_stage()` (import from `compare_pt_trt.py`) | 커널을 `model.N` 별로 묶고 수·종류·정밀도 집계 (metadata `[ONNX Layer: /model.N/…]` 다수결) |
| `pt_structure_edges()` | `layers_pt[i]['f']` → `.pt` 체인/skip 엣지 |
| `trt_structure_edges()` | `build_producers()`+`producer_for()` 로 단계 간 텐서 의존. 순방향만 (역방향은 이름충돌 가짜) |
| `build_dot()` | 좌우 두 컬럼 + invisible spine(곧게) + `rank=same`(행 정렬) + 가운데 점선(초록 감소/주황 증가) |
| `render()` (import) | python `graphviz` → 실패 시 `dot` 실행파일 |

## 참고

- 왼쪽 `leaf 연산 수` 는 정확한 커널 launch 수가 아니다. `torch.cat`/`+`/`chunk`/
  `F.interpolate` 같은 함수형 연산과, 재사용되는 `SiLU` 의 반복 적용은 안 세진다.
  "대략 어디서 얼마나 잘게 연산하나" 감을 잡는 용도.
- 단계 배분(`kernel_stage`)은 `TRT_visualize_model.py` 그래프의 `stage_of()`(첫 매치)와
  경계 reformat 몇 개가 다르게 갈 수 있다.
- `post-process` = `shape_call` 등 `model.N` 이름이 안 붙는 TRT 내부 커널.
