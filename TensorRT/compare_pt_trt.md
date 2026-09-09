# TensorRT/compare_pt_trt.py

**[`.pt` vs TensorRT 구조 비교]** 원본 `best.pt` 의 24개 nn.Module(`model.0`~`model.23`)
각각이 TensorRT 엔진에서 **커널 몇 개로, 어떤 타입·정밀도로** 바뀌었는지 `.pt` 기준으로
나란히 표로 정리한다 → `pt_vs_trt.md` (repo 루트).

| 비교 축 | 스크립트 |
|---|---|
| `.pt` 레이어 구조 그래프 | [`ori_visualize_model.py`](ori_visualize_model.md) |
| ONNX ↔ TRT 커널 이름 (ONNX 기준) | [`TRT_layer_print.py`](TRT_layer_print.md) |
| **`.pt` 모듈 ↔ TRT 커널 집계 (`.pt` 기준)** | 이 스크립트 |

## 원리

**비교 대상은 `.pt` ↔ TRT 딱 둘.** `.pt` 모듈 이름(`model.N`)이 엔진 커널 metadata
`[ONNX Layer: /model.N/...]` 에 남아 있어서, 그 문자열로 316개 커널을 원본 24개 모듈에
되묶는다. (변환 경로가 `.pt`→ONNX→엔진이라 metadata 에 ONNX 이름이 박혀 있을 뿐,
**ONNX 파일 자체는 안 읽는다.**)

`.pt` 모듈 1개엔 "커널 수" 개념이 없으므로, 왼쪽 값으로 그 모듈 안의
**leaf 서브모듈 수**(`Conv2d`·`BatchNorm2d`·`SiLU`·`MaxPool2d` …)를 센다. 정확한
연산 수는 아니고 어림값이다 (`SiLU` 는 재사용돼 1로, `torch.cat`·`+`·`chunk` 같은
함수형 연산은 안 세짐).

> `--onnx model/best.onnx` 를 **명시하면** 표 가운데에 ONNX 노드 수도 끼워
> `.pt→ONNX→TRT` 3단으로 보여준다. 기본은 꺼져 있고 `.pt→TRT` 2단만 나온다.

| 함수 | 역할 |
|---|---|
| `load_model()` + `collect_layers()` | (import from `ori_visualize_model.py`) `best.pt` → 모듈별 `type`/`params`/출력 shape. `main()` 이 여기에 `leaf`(내부 서브모듈 수) 추가 |
| `load_layers()` | (import from `TRT_visualize_model.py`) `best.engine.layers.json` → 커널 리스트 + Bindings |
| `kernel_stage()` | 커널 하나의 소속 `model.N` 판정. metadata 의 `[ONNX Layer: /model.N/…]` **다수결** → 없으면 커널 이름 → 없으면 `post` |
| `aggregate_trt()` | stage 별로 커널 수·타입 분포(`family()`)·출력 정밀도·마지막 출력 dims 집계 |
| `onnx_stage_counts()` | (`--onnx` 명시 시에만) ONNX 노드를 `model.N` 별로 카운트 |
| `count_cell()` | `"9 → 10"` (또는 `--onnx` 시 `"9 → 15 → 10"`) 형태 셀 생성 |
| `to_markdown()` / `print_console()` | 요약 + 모듈별 대응표 + 단계별 커널 목록(`<details>`) |

## 입력

- `model/best.pt` — 원본 가중치 (torch + ultralytics 필요, `yolo` 환경)
- `model/best.engine.layers.json` — `build_trt_engine.py` 산출물. 없으면 먼저:
  ```bash
  python TensorRT/build_trt_engine.py
  ```
- `model/best.onnx` — **안 씀** (기본). `--onnx` 로 명시할 때만 참고용으로 노드 수를 읽는다.

## 사용법

```bash
conda activate yolo

python TensorRT/compare_pt_trt.py                       # .pt → TRT  (기본) → pt_vs_trt.md (repo 루트)
python TensorRT/compare_pt_trt.py --no-shapes           # forward 패스 생략(빠름, .pt shape 열 비움)
python TensorRT/compare_pt_trt.py --onnx model/best.onnx  # (선택) 가운데에 ONNX 노드 수도

# FP16 엔진 판 → pt_vs_trt_fp16.md
python TensorRT/compare_pt_trt.py \
    --engine-json model/best_fp16.engine.layers.json --out pt_vs_trt_fp16.md
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `model/best.pt` | 원본 가중치 |
| `--engine-json` | `model/best.engine.layers.json` | 엔진 레이어 정보 JSON |
| `--onnx` | *(꺼짐)* | 명시하면 표 가운데에 ONNX 노드 수 추가 (`.pt→ONNX→TRT`) |
| `--out` | `pt_vs_trt.md` | 출력 Markdown (repo 루트) |
| `--imgsz` | `640` | shape 관측용 더미 입력 크기 |
| `--no-shapes` | off | forward 패스를 건너뛰고 `.pt` 출력 shape 표기 생략 |

## 산출물 (`pt_vs_trt.md`, repo 루트)

1. **요약** — `.pt` 모듈 수·내부 leaf 레이어 수·params, TRT 커널 수,
   전체 흐름(`.pt leaf → TRT`), 레이아웃 변환만 남은 모듈(Concat)
2. **모듈별 대응표** — `model.N` × (`.pt` 타입/params/출력 shape) ↔
   (**`레이어 수 (.pt→TRT)`** / TRT 커널 타입 / 출력 / 비고)
3. **단계별 TRT 커널 목록** — 모듈마다 `<details>` 안에 `.pt leaf N → 커널 M` + 커널 이름 전체

### `레이어 수 (.pt→TRT)` 열 읽는 법

- 왼쪽 = `.pt` 모듈 안 leaf 서브모듈 수 (어림값). 오른쪽 = 그 단계로 매핑된 TRT 커널 수.
- `9 → 10`, `3 → 2` 처럼 대체로 비슷하거나 약간 준다.
- `7 → 13` (model.9 SPPF) 처럼 **늘기도** 한다 = TRT 가 레이아웃 변환용 `reformat`/`noop` 를 끼워넣음.
- `1 → 1` : `Concat` (zero-copy, 경계 reformat 1개만).
- `88 → 66` (model.23 Detect) : end2end decode/NMS 까지 펼쳐졌는데도 leaf 수보다 적음.
- `--onnx` 를 주면 `9 → 15 → 10` 처럼 가운데에 ONNX 노드 수가 들어간다 (참고용).

## 이 저장소 기준 결과

| 엔진 | TRT 커널 | 산출물 |
|---|--:|---|
| FP32 (`best.engine`) | 316개 (매핑 313 + post 3) | [`pt_vs_trt.md`](../pt_vs_trt.md) |
| FP16 (`best_fp16.engine`) | 221개 (매핑 218 + post 3) | [`pt_vs_trt_fp16.md`](../pt_vs_trt_fp16.md) |

아래 발췌는 FP32 기준. FP16 은 SiLU 가 conv epilogue 로 더 fused 되고 (`caskjitconv`) 포맷
변환 커널이 줄어 C3k2 블록이 `19 → 24~26` 대신 `19 → 13` 수준으로 내려간다.

| | 값 (FP32) |
|---|---|
| `.pt` 모듈 | 24개 · 내부 leaf 레이어 277개 · 2.5M params |
| TRT 커널 | 316개 (매핑 313 + post 3) |
| 전체 흐름 | `.pt` leaf 277 → TRT 커널 313 (+ post 3) |

단계별 발췌 (`.pt` leaf → TRT 커널):

| 단계 | `.pt` | .pt→TRT | 메모 |
|---|---|---|---|
| model.0 | Conv | `3 → 3` | Conv2d+BN+SiLU → reformat+conv+pointwise |
| model.3 | Conv | `3 → 2` | 최소 형태 (BN fold, SiLU fusion, reformat 불필요) |
| model.2/4 | C3k2 | `9 → 10` | |
| model.6/8/13/16/19 | C3k2 | `19 → 24~26` | |
| model.9 | SPPF | `7 → 13` | reformat/noop 삽입으로 커널이 더 많아짐 |
| model.10 | C2PSA | `19 → 30` | attention, myelin `__myl_*` blob + reformat 다수 |
| model.22 | C3k2 | `23 → 35` | attention 포함 |
| **model.23** | Detect | **`88 → 66`** | one2one cv2/cv3 + end2end decode/NMS. `i64` 여기서만 |
| model.12/15/18/21 | Concat | `1 → 1` | zero-copy, 경계 reformat 만 |

## 참고

- `kernel_stage()` 다수결은 `TRT_visualize_model.py` 의 그래프용 `stage_of()`(첫 매치)와
  살짝 다르게 배분될 수 있다. 예: `/model.14/Resize_output_0 copy` (metadata
  `[ONNX Layer: /model.15/Concat]`) 는 그래프에선 `model.14`, 이 표에선 `model.15` 로 감.
- 왼쪽 `.pt` leaf 수는 정확한 연산 수가 아니다. `torch.cat`/`+`/`chunk`/`F.interpolate`
  같은 함수형 연산과, 재사용되는 `SiLU` 의 반복 적용은 안 세진다. "대략 얼마나
  잘게 쪼개졌나" 감만 잡는 용도.
- `TRT 출력` 열은 그 단계 **마지막 커널**의 출력이라, end2end decode 가 이어지는
  `model.23` 은 최종 `output0`(`1x300x6`)이 아니라 중간 텐서(`1x300x4`)로 보일 수 있다.
