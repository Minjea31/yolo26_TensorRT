# TensorRT/

**TensorRT 관련 `.py` 를 한곳에서 관리**하는 디렉터리.
`best.pt` → ONNX → TensorRT 엔진 변환, 엔진 추론/벤치마크, 정밀도(FP16/INT8) 실험,
그리고 그 파이프라인을 위한 모델 검사/시각화 스크립트까지 모두 여기에 둔다.

## 규칙

- TensorRT 관련 코드/스크립트는 이 폴더 안에 만든다.
- 코드 파일을 추가/수정하면 **같은 이름의 `.md`** 문서를 함께 두고,
  repo 루트 `README.md` 의 "스크립트 설명" 표도 갱신한다.
  (예: `export_engine.py` ↔ `export_engine.md`)
- 생성되는 엔진 파일(`*.engine`, `*.plan`), ONNX(`*.onnx`), 캐시(`*.cache`) 같은
  대용량/빌드 산출물은 커밋하지 않는 것을 권장 (필요 시 `.gitignore` 에 추가).

## 현재 구성 (파이프라인 순서)

| 파일 | 용도 |
|---|---|
| `ori_visualize_model.py` | **원본** `best.pt` 레이어 구조 → 세로 그래프 (`model/ori_structure.png`). 상세: [`ori_visualize_model.md`](ori_visualize_model.md) |
| `build_trt_engine.py` | `best.pt` → ONNX → TensorRT 엔진(.engine) 2단계 변환 + `best.engine.layers.json` 덤프. 상세: [`build_trt_engine.md`](build_trt_engine.md) |
| `TRT_visualize_model.py` | **TensorRT 적용 후** 구조 (`best.engine.layers.json`) → 세로 그래프 (`model/TRT_structure.png`). 상세: [`TRT_visualize_model.md`](TRT_visualize_model.md) |
| `TRT_layer_print.py` | 원본 ONNX ↔ TRT 커널 **이름 변화(fusion/rename)** 를 터미널에 출력 + `--json` 대응표 저장. 상세: [`TRT_layer_print.md`](TRT_layer_print.md) |
| `compare_pt_trt.py` | **`.pt` 모듈(24개) ↔ TRT 커널(316개)** 집계 비교 — **표** → `model/pt_vs_trt.md`. 각 `model.N` 을 `내부 leaf 레이어 수 → TRT 커널 수` 로. `--onnx` 명시 시 가운데 ONNX 노드 수 추가(기본 꺼짐). 상세: [`compare_pt_trt.md`](compare_pt_trt.md) |
| `kernel_compare.py` | **`.pt` 구조 ↔ TRT 구조** 좌우 비교 — **그래프** → `model/kernel_compare.png`. 왼쪽=eager 연산 종류, 오른쪽=TRT 커널 종류, 가운데 점선=`연산 수→커널 수`, 컬럼 안 체인+skip. 상세: [`kernel_compare.md`](kernel_compare.md) |

## (예정) 구성

| 파일 | 용도 |
|---|---|
| `infer_engine.py` | 빌드된 엔진으로 추론 |
| `benchmark.py` | PyTorch vs TensorRT 지연시간/처리량 비교 |
