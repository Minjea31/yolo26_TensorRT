# TensorRT/TRT_layer_print.py

**[TensorRT 레이어 이름 변화 — 터미널 출력]**
`build_trt_engine.py` 가 덤프한 `best.engine.layers.json` (+ `best.onnx`) 를 읽어,
원본 ONNX(=PyTorch) 레이어가 TensorRT 커널로 오면서 **어떻게 합쳐지고 이름이 바뀌었는지**
를 stdout 으로 출력한다. `--json` 으로 대응표를 파일로도 저장한다.

> `.pt` 모듈 단위로 "커널 몇 개로 분해됐나" 를 보려면 → [`compare_pt_trt.py`](compare_pt_trt.md)

## 입력

- `model/best.engine.layers.json` — `build_trt_engine.py` 결과. 없으면 먼저 그걸 실행.
- `model/best.onnx` (선택) — 있으면 `[3]` "엔진에서 사라진 노드" 섹션까지. 기본 경로면 자동 사용.

## 출력 (stdout)

| 섹션 | 내용 |
|---|---|
| `[1]` | TensorRT 커널 → 원본 ONNX (엔진 실행 순서). `keep` 은 기본 생략, `--all` 로 포함 |
| `[2]` | 원본 ONNX 노드 → TensorRT 커널 (역방향) |
| `[3]` | `best.onnx` 에 있지만 엔진 metadata 에 안 보이는 노드 (상수폴딩/fusion 흡수) |
| 요약 | keep / rename / fuse / new-io / new-int 개수 |

### category

| 값 | 뜻 |
|---|---|
| `keep` | 원본 이름 그대로 (예: `/model.0/conv/Conv`) |
| `rename` | 1:1 인데 이름만 바뀜 (예: `.../attn/MatMul` → `/model_10/.../MatMul_myl109_2`) |
| `fuse(N)` | 원본 N개 → 커널 1개 (예: SiLU = `Sigmoid`+`Mul` → `PointWiseV2` 1개) |
| `new-io` | TRT 가 삽입한 재배치/복사 `Reformat`·`NoOp` (원본에 없음) |
| `new-int` | TRT 내부 커널 (`shape_call`, myelin `__myl_*` blob 등, metadata 없음) |

## 사용법

```bash
conda activate yolo
python TensorRT/TRT_layer_print.py
python TensorRT/TRT_layer_print.py --all                       # keep 포함 전부
python TensorRT/TRT_layer_print.py --onnx model/best.onnx
python TensorRT/TRT_layer_print.py --json model/layer_name_map.json   # 대응표 JSON 저장
python TensorRT/TRT_layer_print.py --width 60                   # 긴 이름 자르기
```

## 동작 원리

| 함수 | 역할 |
|---|---|
| `find_repo_root()` + `main()` `os.chdir()` | repo 루트로 이동 → 모든 경로 상대 |
| `load_engine_layers()` | JSON → `Layers` + `Bindings` |
| `onnx_layers_of()` | 레이어 `Metadata` 의 `[ONNX Layer: X]` 들 추출 (= fusion 원본 목록) |
| `classify()` | `Name` vs `Metadata` 비교 → keep/rename/fuse/new-io/new-int |
| `analyze()` | 위를 모아 구조화된 dict 반환 |
| `onnx_layers_of()` | `compare_pt_trt.py` 가 stage 판정에 import |
| `print_report()` | dict → 터미널 텍스트 |

## 현재 엔진 기준 (FP32) 대략

`keep 97 · rename 39 · fuse 106 (원본 241 → 커널 106) · new-io 71 · new-int 3`,
TRT 커널 316개, 원본 ONNX 452개 중 91개가 상수폴딩/fusion 으로 흡수.
(빌드마다 몇 개씩 달라질 수 있음)
