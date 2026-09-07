# TensorRT/TRT_visualize_model.py

**[TensorRT 적용 후 그래프]** `build_trt_engine.py` 가 덤프한 엔진 레이어 정보
(`<engine>.engine.layers.json`)를 읽어, TensorRT 가 **fusion·정밀도 적용을 끝낸 실제 실행
그래프**를 세로 방향 그래프로 그려 이미지로 저장한다.

| | 스크립트 |
|---|---|
| 원본(.pt) 구조 | [`ori_visualize_model.py`](ori_visualize_model.md) |
| **TensorRT 후 구조** | 이 스크립트 |

## 입력

`model/best.engine.layers.json` — `build_trt_engine.py` 가 `EngineInspector` +
`ProfilingVerbosity.DETAILED` 로 만든 파일. 없으면 먼저:

```bash
python TensorRT/build_trt_engine.py            # (또는 --fp16 등)
```

## 두 가지 상세도 (`--level`)

| level | 노드 | 설명 |
|---|---|---|
| `stage` (기본) | ~21 (원본 `model.N` 단위) | 각 원본 단계가 TRT 커널 몇 개로 fusion 됐는지, 단계별 정밀도 요약. **원본 그래프(24 노드)와 세로로 나란히 비교**하기 좋음 |
| `layer` | 수백 (커널 레이어 전부) | `CaskConvolution` / `PointWiseV2` / `Reformat` … 개별 커널까지 펼침. 이미지가 매우 큼(세로로 길다) |

두 모드 다 **세로(`rankdir=TB`)** 로 그린다. 보이지 않는 척추(spine) 엣지로 dot 이
옆으로 퍼지지 않고 세로 컬럼으로 쌓이게 강제한다 (원본 그래프와 같은 형태).

## 사용법

```bash
conda activate yolo

python TensorRT/TRT_visualize_model.py                        # stage 모드, model/TRT_structure.png
python TensorRT/TRT_visualize_model.py --level layer --hide noop,shape_call,reformat
python TensorRT/TRT_visualize_model.py \
    --engine-json model/best.engine.layers.json \
    --out         model/TRT_structure.png --rankdir TB
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--engine-json` | `model/best.engine.layers.json` | 입력 JSON |
| `--out` | `model/TRT_structure.png` | 출력 이미지 (.png/.svg/.pdf) |
| `--level` | `stage` | `stage` 요약 / `layer` 전체 |
| `--hide` | `noop,shape_call` | (layer 모드) 접어서 지나갈 LayerType 부분일치, 콤마구분. 예: `noop,shape_call,reformat` |
| `--rankdir` | `TB` | `TB` 세로 / `LR` 가로 |
| `--dpi` | `150` | 해상도 |

## 그래프 읽는 법

- **노드 색** = LayerType 계열: 🟦 conv, 🟪 gemm/matmul, 🟩 pointwise, 🟫 pool,
  🟧 resize·decode/NMS, ⬜ reformat·noop
- **노드 테두리** (layer 모드): 파랑=FP16 출력, 빨강=INT8 출력 (없으면 FP32)
- **라벨**: `model.N` / 계열별 개수 / `L layers · 정밀도`  (layer 모드는 `idx 이름 / LayerType / dims dtype`)
- FP32 엔진이면 대부분 `f32`. `--fp16` 로 빌드한 엔진을 넣으면 `f16` 이 섞여 보인다.

## 산출물

| 파일 | 내용 |
|---|---|
| `<out>` | 세로 방향 구조 그래프 |
| `<out>.dot` | Graphviz DOT 소스 |
| stdout | 총 레이어 수, LayerType 분포, 정밀도 분포, 그래프 노드/엣지 수 |

## 동작 원리

| 함수 | 역할 |
|---|---|
| `find_repo_root()` + `main()` `os.chdir()` | repo 루트 탐색 후 그리로 이동 → 모든 경로는 repo 루트 기준 상대경로(`model/…`). 어디서 실행해도 동작 |
| `load_layers()` | `layers.json` → `Layers` 리스트 + `Bindings` |
| `build_producer()` | 텐서 이름 → 그 텐서를 만든 레이어 idx |
| `stage_of()` | 레이어 Name/Metadata 에서 `model.(\d+)` 추출 → 원본 단계 번호 |
| `family()` | LayerType → (짧은 계열명, 색) |
| `graph_stage()` | 단계별로 레이어를 묶고, cross-stage 텐서 의존으로 엣지 생성. 연속 단계 사이 invisible spine 추가 |
| `graph_layer()` | 커널 레이어 노드화. `--hide` 대상은 건너뛰되 조상↔자손을 이어줌(bridge). id 순서 spine 추가 |
| `build_dot()` | `rankdir=TB` + `weight=10` invisible spine 으로 세로 강제. 색/테두리 적용 |
| `render()` | python `graphviz` → 실패 시 `dot` 실행파일 |

## 참고

- 원본 24단계 중 `model.12/15/18/21` (Concat) 은 TRT 가 이웃 커널로 fusion 해서
  독립 노드로 안 나올 수 있다 — 이게 정상이며 fusion 됐다는 뜻.
- `post-process` 노드(end2end decode/NMS, `__myl_*` fused kernel)는 텐서 이름이
  많이 바뀌어 상위 단계와 엣지가 안 붙을 수 있다.
