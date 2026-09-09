# TensorRT/build_trt_engine.py

`model/best.pt` 를 **ONNX 로 변환**한 뒤 **TensorRT 엔진(.engine)** 으로 빌드하는 2단계 스크립트.
직접 실행하는 용도.

```
[1] PyTorch(.pt)  ──►  ONNX        Ultralytics  model.export(format="onnx")
[2] ONNX          ──►  TensorRT    TensorRT Python API (Builder + OnnxParser)
```

## 사전 준비

| 항목 | 비고 |
|---|---|
| torch 2.x + ultralytics 포크 | `cd yolo26 && pip install -r requirements.txt` (이미 `yolo` env) |
| onnx / onnxruntime | ONNX 유효성·IO 확인용 (선택, 이미 설치됨) |
| **TensorRT** | `tensorrt-cu12==10.7.0` — `yolo26/requirements.txt` 에 포함, `yolo` env 에 설치 완료. 다른 CUDA 면 `tensorrt-cu13` 등 |
| NVIDIA GPU + 드라이버 | [2] 단계는 GPU 필요 (여기 환경: RTX 3050 Ti, driver 570, CUDA 12.x) |

> TensorRT 미설치 환경이면 [2] 단계에서 설치 방법을 안내하고 종료한다. [1] 단계(ONNX)는 GPU 없이도 됨.

## 사용법

```bash
conda activate yolo

# 전체 파이프라인 (FP32 엔진) → model/best.engine
python TensorRT/build_trt_engine.py

# FP16 엔진 → model/best_fp16.engine  (FP32 를 안 덮어씀)
python TensorRT/build_trt_engine.py --fp16

# 경로/해상도 직접 지정 (--engine 을 주면 접미사 규칙 대신 그 경로 사용)
python TensorRT/build_trt_engine.py \
    --weights model/best.pt --imgsz 640 --fp16 --workspace 4 \
    --onnx model/best.onnx --engine model/best_fp16.engine

# [1]만 — ONNX 까지만 만들기 (GPU 불필요)
python TensorRT/build_trt_engine.py --skip-engine

# [2]만 — 기존 FP32 ONNX 재사용해서 FP16 엔진만 → model/best_fp16.engine
python TensorRT/build_trt_engine.py --skip-onnx --fp16

# 동적 batch (1 ~ max-batch)
python TensorRT/build_trt_engine.py --dynamic --max-batch 8 --fp16
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `model/best.pt` | 입력 가중치 |
| `--onnx` / `--engine` | weights 이름 + 정밀도 접미사 | 출력 경로. 직접 주면 그 경로를 그대로 사용 |
| `--imgsz` | `640` | 입력 해상도(정사각) |
| `--opset` | ultralytics 기본값 | ONNX opset |
| `--batch` | `1` | 정적 batch 크기 |
| `--dynamic` / `--max-batch` | off / `8` | 동적 batch ONNX + optimization profile |
| `--simplify` | off | `onnxslim` 으로 ONNX 단순화 (패키지 필요) |
| `--half` | off | ONNX 자체를 FP16 로 export (CUDA 필요) → `best_fp16.onnx` |
| `--fp16` | off | TensorRT FP16 모드 → `best_fp16.engine` |
| `--int8` | off | TensorRT INT8 플래그만 (캘리브레이터 없음 → 정확도 주의) → `best_int8.engine` |
| `--workspace` | `4.0` | TensorRT workspace (GiB) |
| `--verbose` | off | TensorRT 로그 VERBOSE |
| `--skip-onnx` / `--skip-engine` | off | 단계 건너뛰기 |
| `--no-layer-info` | (덤프 함) | 엔진 레이어 정보 JSON 덤프 비활성화 |

## 산출물

경로를 안 주면 `--weights` 이름(`best`)에 **정밀도 접미사**가 붙는다:

| 실행 | 엔진 파일 |
|---|---|
| (기본) | `model/best.engine` |
| `--fp16` | `model/best_fp16.engine` |
| `--int8` | `model/best_int8.engine` |
| `--fp16 --int8` | `model/best_fp16_int8.engine` |

| 파일 | 내용 |
|---|---|
| `<weights>.onnx` | 변환된 ONNX. [Netron](https://netron.app) 으로 그래프 확인. `--fp16` 만으론 FP32 그대로(`best.onnx`), `--half` 면 `best_fp16.onnx` |
| `<weights>[_fp16].engine` | 직렬화된 TensorRT 엔진 (위 표) |
| `<engine>.layers.json` | TensorRT 가 **fusion·정밀도 적용을 끝낸 뒤**의 레이어 목록. `EngineInspector` + `ProfilingVerbosity.DETAILED` 로 추출. **"TensorRT 적용 후 그래프"** 를 그릴 때 이 파일을 입력으로 쓰면 됨 |

콘솔에는 각 단계 진행 상황, ONNX/엔진 IO 텐서(이름·shape·dtype), 레이어 개수가 출력된다.

## 동작 원리

| 함수 | 역할 |
|---|---|
| `find_repo_root()` | repo 루트 탐색 후 `main()` 이 `os.chdir()` → 이후 모든 경로는 repo 루트 기준 상대경로(`model/best.onnx`, `model/best.engine` …). 어디서 실행해도 동작 |
| `export_onnx()` | `YOLO(weights).export(format="onnx", ...)`. `--skip-onnx` 면 기존 파일 재사용. 끝나면 `_check_onnx()` 로 `onnx.checker` + onnxruntime IO 출력 |
| `build_engine()` | `trt.Builder` + `trt.OnnxParser` 로 네트워크 구성 → `BuilderConfig` 에 workspace / FP16 / INT8 / (동적 shape면) optimization profile 설정 → `build_serialized_network()` 로 빌드 → `.engine` 저장. TRT 8/10 API 차이를 버전 분기로 처리 (`EXPLICIT_BATCH`, `set_memory_pool_limit` vs `max_workspace_size`, `num_io_tensors` vs bindings) |
| `_summarize_engine()` | 엔진 역직렬화 → IO 텐서 출력 → `EngineInspector.get_engine_information(JSON)` 을 `<engine>.layers.json` 으로 저장 |

## Troubleshooting

### `[2]` 에서 `CUDA initialization failure with error: 100` / `Aborted (core dumped)`

TensorRT 가 GPU 를 못 잡는 경우다. **torch 는 자기 CUDA 라이브러리를 번들**해서 되는데
TRT 는 셸 환경을 타서 안 되는 상황. 원인은 거의 항상 둘 중 하나:

- `CUDA_VISIBLE_DEVICES` 가 `''` 또는 없는 인덱스(`-1`, `1` …)로 설정됨
- 구버전 시스템 CUDA(`/usr/local/cuda-10.x`)가 `LD_LIBRARY_PATH` 앞쪽에 있음
  (특히 ROS `setup.bash` 를 source 한 셸)

확인:
```bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```
`device_count()==0` 이면 스크립트가 core dump 대신 안내 메시지를 내고 종료한다(가드 추가됨).

해결:
```bash
unset CUDA_VISIBLE_DEVICES
# 또는 깨끗한 환경으로 실행 (ONNX 는 이미 있으니 [2]만):
env -u LD_LIBRARY_PATH python TensorRT/build_trt_engine.py --skip-onnx
# 또는 ROS 안 물린 새 터미널에서 conda 환경만 활성화 후 재실행
```

## 다음 단계 (그래프)

- **변환 직후 구조** → `.onnx` 를 Netron 으로 열거나, 별도 onnx-그래프 스크립트로 렌더
- **TensorRT 적용 후 구조** → `.engine.layers.json` (레이어·정밀도·fusion) 파싱해서 그래프로.
  현재 엔진 기준 340개 레이어 (`CaskConvolution`, `PointWiseV2`(fused), `Reformat` 등).
  → 후속 `TensorRT/visualize_engine.py` 로 추가 예정
