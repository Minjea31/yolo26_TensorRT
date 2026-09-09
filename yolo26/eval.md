# yolo26/eval.py

**[`.pt` vs TensorRT 엔진 — 정확도/속도 비교]** 같은 데이터셋(`dataset.yaml`)의 한 split
(기본 `test`)을 **batch=1** 로 돌려 원본 `model/best.pt` 와 **이미 빌드된** TensorRT 엔진
`model/best.engine` 을 Ultralytics `val` 로 나란히 평가한다 → `eval_pt_vs_trt.md` (repo 루트).

> ⚠️ 이 스크립트는 **`model/best.engine` 을 그대로 불러 쓴다.** `.pt`→ONNX→엔진 재변환은
> 하지 않는다. 엔진을 새로 만들려면 [`TensorRT/build_trt_engine.py`](../TensorRT/build_trt_engine.md).

| 비교 축 | 스크립트 |
|---|---|
| `.pt` 모듈 ↔ TRT 커널 **구조** 집계 | [`TensorRT/compare_pt_trt.py`](../TensorRT/compare_pt_trt.md) |
| `.pt` 구조 ↔ TRT 구조 좌우 **그래프** | [`TensorRT/kernel_compare.py`](../TensorRT/kernel_compare.md) |
| **`.pt` vs 엔진 정확도·속도 (`val`)** | 이 스크립트 |

## 원리

- 두 가중치(`model/best.pt`, `model/best.engine`)에 대해 각각 `YOLO(w).val(...)` 을
  **완전히 같은 인자**(`data` / `split` / `imgsz` / `batch` / `workers`)로 호출한다.
- 엔진 로딩·추론은 Ultralytics `AutoBackend` 가 TensorRT 런타임으로 처리한다.
  `best.engine` 은 `build_trt_engine.py` 가 raw TensorRT API 로 만든 것이라
  Ultralytics 메타데이터 헤더가 없지만, `val` 은 클래스 이름을 `dataset.yaml` 에서
  읽으므로 문제 없다 (엔진의 `names` 는 무시됨).
- `best.engine` 은 **정적 batch=1** (입력 `images (1,3,640,640)`) 로 빌드돼 있어
  다른 batch 로는 추론이 안 된다 → `--batch` 기본값 1, 바꾸면 경고를 찍는다.
- 결과 객체에서 `box.map` / `box.map50` / `box.map75` / `box.mp` / `box.mr` 와
  `speed`(preprocess·inference·postprocess, ms per image) 를 뽑아 표로 만든다.

| 함수 | 역할 |
|---|---|
| `evaluate(weights, args, tag)` | `YOLO(weights).val(...)` 실행 → 지표/속도 dict. val 산출물은 `runs/eval/<tag>/` (`exist_ok=True`) |
| `_delta()` | `TRT − .pt` 차이를 부호 붙여 문자열로 |
| `_pad()` | 콘솔 표에서 한글(전각) 폭 보정 왼쪽 정렬 |
| `print_table()` | 콘솔에 `.pt` vs `TRT engine` vs `Δ` 표 (한쪽만 평가하면 단독 표) |
| `to_markdown()` | 정확도 표 + 속도 표 + 요약(배속·mAP 변화) 을 Markdown 으로 |
| `main()` | `os.chdir(REPO_ROOT)` 후 인자 파싱 → 평가 → 표 출력 → `--out` 저장 |

## 입력

- `model/best.pt` — 원본 가중치 (torch + ultralytics, `yolo` 환경)
- `model/best.engine` — **이미 만들어져 있어야 함**. 없으면:
  ```bash
  python TensorRT/build_trt_engine.py
  ```
- `dataset.yaml` + `dataset/<split>/` — Roboflow `crash_test` (nc=1, `CAR`)

## 사용법

```bash
conda activate yolo

python yolo26/eval.py                     # model/best.pt vs model/best.engine · split=test · batch=1
python yolo26/eval.py --split val         # val split 으로
python yolo26/eval.py --pt-only           # .pt 만
python yolo26/eval.py --engine-only       # 엔진만
python yolo26/eval.py --no-plots          # PR curve 등 플롯 생략
```

`os.chdir(REPO_ROOT)` 를 하므로 **어느 폴더에서 실행해도** 상대경로가 repo 루트 기준으로 고정된다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--pt` | `model/best.pt` | 원본 가중치 (.pt) |
| `--engine` | `model/best.engine` | **기존** TensorRT 엔진 (.engine) |
| `--data` | `dataset.yaml` | 데이터셋 yaml |
| `--split` | `test` | `train` / `val` / `test` |
| `--imgsz` | `640` | 입력 해상도 |
| `--batch` | `1` | 배치 크기 (정적 엔진은 1 고정) |
| `--workers` | `4` | 데이터로더 workers |
| `--out` | `eval_pt_vs_trt.md` | 비교 결과 Markdown (repo 루트) |
| `--pt-only` / `--engine-only` | off | 한쪽만 평가 |
| `--no-plots` | off | val 플롯 생성 안 함 |

## 산출물

- **콘솔** — `.pt` vs `TRT engine` vs `Δ (TRT−.pt)` 표 (정확도 5행 + 속도 4행)
- **`eval_pt_vs_trt.md`** (repo 루트) — 정확도 표 · 속도 표 · 요약(배속, mAP50-95 변화율)
- **`runs/eval/pt/`, `runs/eval/trt/`** — Ultralytics `val` 산출물(PR curve·혼동행렬 등).
  `.gitignore` 에 `runs/` 로 제외돼 있다.

## 이 저장소 기준 결과 (test split · batch=1)

| 지표 | 원본 `.pt` | TRT FP32 | TRT FP16 |
|---|--:|--:|--:|
| mAP50-95 | 0.8534 | 0.8399 (−0.0135) | 0.8410 (−0.0124) |
| mAP50 | 0.9810 | 0.9827 (+0.0016) | 0.9827 (+0.0016) |
| mAP75 | 0.9582 | 0.9440 (−0.0141) | 0.9406 (−0.0175) |
| Precision | 0.9943 | 0.9592 (−0.0351) | 0.9634 (−0.0309) |
| Recall | 0.9271 | 0.9799 (+0.0528) | 0.9792 (+0.0521) |
| 추론 (ms/img) | ~7–8 | 3.02 | 1.60 |
| 합계 (ms/img) | ~7.7–8.9 | 3.66 | 2.21 |
| `.pt` 대비 배속 (추론 / 합계) | 1× / 1× | 2.40× / 2.11× | **5.24× / 4.05×** |

> 정확도는 FP32·FP16 모두 **사실상 동일**(mAP50-95 −1.5%p 이내), 속도는 FP32 2×·FP16 4× 이상.
> 엔진끼리 비교하면 FP16 이 FP32 보다 추론 1.9× 빠르다(3.02 → 1.60 ms).
> P↓ / R↑ 는 커널 fusion·연산 순서 차이로 confidence 분포가 미세하게 달라져 생기며 mAP 총합은 유지된다.
> `.pt` 추론 절대값은 GPU 부하로 실행마다 7~8 ms 흔들리므로 배속으로 비교한다 (배속 = 각 엔진을 잰 같은 실행의 `.pt` 기준).
> 원본 산출: [`eval_pt_vs_trt.md`](../eval_pt_vs_trt.md) (FP32) · [`eval_pt_vs_trt_fp16.md`](../eval_pt_vs_trt_fp16.md) (FP16)

## 참고

- Ultralytics 의 `runs_dir` 설정과 무관하게 `--project` 를 `runs/eval` 로 넘겨
  산출물을 repo 안에 모은다.
- `.pt` 는 학습용 그래프까지 forward 하지만 `val` 은 추론 경로만 쓰므로 공정한 비교다.
- FP16/INT8 엔진과 비교하려면 `build_trt_engine.py --half` 등으로 엔진을 다시 만든 뒤
  `--engine` 으로 그 경로를 지정하면 된다.
