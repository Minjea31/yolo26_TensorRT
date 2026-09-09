# TensorRT 커널(레이어) 타입 정리

`best.pt` → 엔진 변환 결과를 읽을 때 나오는 "커널 종류"를 정리한 문서.
두 층위가 있다.

| 층위 | 무엇 | 개수 | 고정된 목록? |
|---|---|---|---|
| **A. 표현 레벨** (`trt.LayerType`) | TensorRT 네트워크가 표현할 수 있는 연산 종류 | **49** (TRT 10.7) | ✅ 버전마다 고정 |
| **B. 실행 레벨** (`engine.layers.json` 의 `LayerType`) | 빌드 후 엔진에 실제로 박힌 커널(tactic)의 분류명 | 엔진마다 다름 (이 repo: FP32 10, FP16 11) | ❌ 버전·GPU·정밀도·fusion 에 따라 달라짐 |

- **A** 는 "TensorRT 가 할 줄 아는 것 전체" — 닫힌 카탈로그.
- **B** 는 "이 모델·이 GPU·이 정밀도에서 실제로 생성된 커널" — A 의 레이어들이 fusion 으로 합쳐지거나(`Conv`+`BN`+`SiLU` → 커널 1개) 쪼개져서(`Activation` SiLU → `Sigmoid`·`Mul` 2개) 나온 결과. tactic 이름(`sm80_xmma_fprop_implicit_gemm_...`)은 사실상 무한.

생성: `python TensorRT/build_trt_engine.py [--fp16]` → `model/best[_fp16].engine.layers.json`
분석: [`compare_pt_trt.py`](compare_pt_trt.md) (`.pt` 모듈 ↔ 커널 집계), [`TRT_layer_print.py`](TRT_layer_print.md) (ONNX ↔ 커널 이름)

---

## A. `trt.LayerType` — TensorRT 가 표현 가능한 레이어 49종 (TRT 10.7)

`import tensorrt as trt; trt.LayerType` 의 전체 목록. 네트워크 정의(ONNX 파싱) 단계에서 쓰이는 "연산 종류"다. 실제 GPU 커널이 아니라 **연산의 종류**임에 주의.

### 연산 (17)

| 타입 | 설명 |
|---|---|
| `CONVOLUTION` | 합성곱 |
| `DECONVOLUTION` | 전치 합성곱 (upconv) |
| `MATRIX_MULTIPLY` | 행렬곱 (attention·FC) |
| `EINSUM` | 아인슈타인 합 표기 일반 텐서 곱 |
| `POOLING` | max/avg 풀링 |
| `ACTIVATION` | ReLU·Sigmoid·Tanh·... (SiLU 는 Sigmoid+ELEMENTWISE 로 표현) |
| `PARAMETRIC_RELU` | 채널별 학습형 LeakyReLU |
| `ELEMENTWISE` | 원소별 이항 연산 (`+`,`*`,`max`,...) |
| `UNARY` | 원소별 단항 (`exp`,`sqrt`,`abs`,...) |
| `SCALE` | 채널별 아핀 (BN 추론형: `y = (x*scale + shift)^power`) |
| `REDUCE` | 축 축소 (sum/mean/max/...) |
| `SOFTMAX` | 소프트맥스 |
| `RAGGED_SOFTMAX` | 가변 길이 시퀀스 소프트맥스 |
| `NORMALIZATION` | LayerNorm·GroupNorm·InstanceNorm |
| `LRN` | Local Response Normalization |
| `GRID_SAMPLE` | 좌표 기반 샘플링 (`F.grid_sample`) |
| `ONE_HOT` | 원-핫 인코딩 |

### 양자화 (2)

| 타입 | 설명 |
|---|---|
| `QUANTIZE` | FP → INT8/FP8 양자화 (Q) |
| `DEQUANTIZE` | 역양자화 (DQ). QAT ONNX 의 Q/DQ 쌍 |

### 형상·재배열 (13)

| 타입 | 설명 |
|---|---|
| `SHUFFLE` | reshape + transpose (permute) |
| `SLICE` | 부분 잘라내기 / stride / pad |
| `CONCATENATION` | 축 이어붙이기 |
| `GATHER` | 인덱스로 모으기 (`gather`/`take`) |
| `SCATTER` | 인덱스로 흩뿌리기 |
| `PADDING` | 공간 패딩 |
| `RESIZE` | 업/다운샘플 (nearest/linear/cubic) |
| `REVERSE_SEQUENCE` | 시퀀스 뒤집기 |
| `SQUEEZE` / `UNSQUEEZE` | 크기-1 축 제거 / 삽입 |
| `SHAPE` | 텐서의 shape 를 텐서로 |
| `CAST` | dtype 변환 |
| `IDENTITY` | 통과 (dtype/포맷 변환 겸용) |

### 선택·비교 (4)

| 타입 | 설명 |
|---|---|
| `SELECT` | 조건부 원소 선택 (`where`) |
| `TOPK` | 상위 k |
| `NON_ZERO` | 0 아닌 원소 좌표 |
| `NMS` | Non-Max Suppression |

### 제어흐름 (7)

| 타입 | 설명 |
|---|---|
| `LOOP_OUTPUT` `RECURRENCE` `TRIP_LIMIT` `ITERATOR` | `ILoop` 구성요소 (RNN 류 반복) |
| `CONDITION` `CONDITIONAL_INPUT` `CONDITIONAL_OUTPUT` | `IIfConditional` 구성요소 (분기) |

### 상수·생성 (2)

| 타입 | 설명 |
|---|---|
| `CONSTANT` | 상수 텐서 |
| `FILL` | `linspace`/`iota`/난수 등 절차적 생성 |

### 플러그인·기타 (4)

| 타입 | 설명 |
|---|---|
| `PLUGIN` / `PLUGIN_V2` / `PLUGIN_V3` | 커스텀 플러그인 (예: `EfficientNMS_TRT`) |
| `ASSERTION` | 빌드/런타임 조건 검증 |

> **레이어 ≠ 커널.** 빌드하면 여러 레이어가 한 커널로 fuse 되거나(대표적으로 `CONVOLUTION`+`SCALE`(BN)+`ACTIVATION` → conv 커널 1개), 한 레이어가 여러 커널로 나뉜다. 그래서 아래 B 의 개수는 A 보다 훨씬 적고, 이름도 다르다.

---

## B. 이 repo 엔진에 실제로 쓰인 커널

`engine.layers.json` 의 `LayerType` 필드 기준. FP32(`best.engine`) 10종 / FP16(`best_fp16.engine`) 11종 — 합집합 11종 (`CaskJitConv` 만 FP16 전용).

| 커널 (`LayerType`) | family\* | FP32 | FP16 | 정체 |
|---|---|--:|--:|---|
| `CaskConvolution` | conv | 68 | **77** | 일반 합성곱 (implicit-GEMM / Winograd / depthwise) |
| `CaskGemmConvolution` | gemm | 29 | 13 | 합성곱을 순수 GEMM 타일로 (1×1·작은 채널에 유리) |
| `CaskJitConv` | (미분류) | — | **7** | JIT 컴파일 합성곱 + 에필로그(SiLU) 융합. **FP16 전용** |
| `PointWiseV2` | pointwise | **87** | 21 | 원소별 융합 커널 — SiLU(`Sigmoid`×`Mul`), residual `Add` |
| `Reformat` | reformat | 44 | 42 | 레이아웃/정밀도 변환 복사 (NHWC↔NC/32HW32, f32↔f16) |
| `NoOp` | noop | 61 | 36 | 거의 무비용 경계 정리용 복사 노드 |
| `CaskPooling` | pool | 3 | 3 | SPPF 의 MaxPool 3개 (`model.9`) |
| `Resize` | resize | 2 | 2 | head 업샘플 2회 (`model.11`, `model.14`, nearest) |
| `gemm` | gemm | 4 | 4 | **myelin** 융합 영역의 행렬곱 (attention Q·Kᵀ, ·V) |
| `kgen` | post | 15 | 13 | **myelin** 이 생성한 융합 커널 (softmax·transpose·add 등) |
| `shape_call` | post | 3 | 3 | **myelin** end2end 후처리의 shape 연산 |
| **합계** | | **316** | **221** | |

\* `family` = repo 문서 표(`pt_vs_trt.md` 등)에서 쓰는 8계열 짧은 이름. `TensorRT/TRT_visualize_model.py:88` 의 `FAMILIES`. `CaskJitConv`·`shape_call` 은 매핑이 없어 원래 이름으로 찍힌다.

### B.1 합성곱 계열

**`CaskConvolution`** — 가장 많은 커널. cuDNN 이 아니라 TensorRT 내장 `cask` 라이브러리의 합성곱. tactic 이름에서 구현이 보인다:
- FP32: `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f32f32_tf32f32_f32_nhwckrsc_nhwc_...` (implicit-GEMM, **TF32** 누산), `ampere_scudnn_winograd_128x128_...` (3×3 는 Winograd), `sm50_..._depthwiseFP32NHWC4` (depthwise 분기)
- FP16: `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x32_...` (**FP16** 누산, 텐서코어)

**`CaskGemmConvolution`** — 합성곱을 im2col 없이 순수 GEMM 타일로 푸는 tactic. 1×1 conv, `cv2`/`cv3` 병합 conv 등에 선택됨. FP32 `sm86_xmma_gemm_f32f32_tf32f32_...`, FP16 `ampere_h16816gemm_...`. FP16 에서 29→13 으로 줄어드는데, 그만큼이 `CaskConvolution`/`CaskJitConv` 로 재배정된다.

**`CaskJitConv`** (FP16 전용, 7개) — 빌드 타임에 커널을 **JIT 컴파일**해서 conv 뒤에 오는 pointwise 식(SiLU, `Add`)을 **에필로그로 붙여** 한 커널로 만든다. JSON 에 `PointWiseExpressionOperations` 필드가 붙어 있는 게 특징. 이름도 `/model.4/m.0/cv2/conv/Conv + PWN(PWN(PWN(.../Sigmoid), .../Mul), .../Add)` 처럼 conv+활성+residual 이 한 줄. FP32 에서는 이 융합이 안 돼서 conv 커널과 별도 `PointWiseV2` 로 남는다 → **FP16 이 316→221 로 줄어드는 핵심 원인**.

### B.2 pointwise 계열

**`PointWiseV2`** — 원소별 연산을 트리로 융합한 커널. 이름의 `PWN(...)` = **P**oint**W**ise**N**ode 중첩. 예:
- `PWN(PWN(/model.0/act/Sigmoid), PWN(/model.0/act/Mul))` = SiLU (`x * sigmoid(x)`)
- `PWN(PWN(PWN(.../Sigmoid), PWN(.../Mul)), PWN(.../Add))` = SiLU + residual add

FP32 87개 → FP16 21개. 줄어든 66개는 대부분 `CaskConvolution`/`CaskJitConv` 의 에필로그로 흡수됐다. 남은 21개는 앞 conv 와 포맷/브로드캐스트가 안 맞아 독립 커널로 남은 것.

### B.3 데이터 이동 계열

**`Reformat`** vs **`NoOp`** — 둘 다 "복사"지만:
- `Reformat` = 실제 **메모리 레이아웃/정밀도 변환** (예: `NHWC` → `NC/32HW32` 텐서코어 포맷, split 출력 재배치). `Origin` 필드 있음.
- `NoOp` = 변환이 실질적으로 필요 없어 **거의 0 비용**인 경계 정리 복사. 프로파일에서 시간 안 잡히는 경우가 많음.

`Split_output_N copy` (C3k2 의 채널 분할 후 정렬), `Reformatting CopyNode for Input Tensor 0 to ...` 형태가 대부분. FP16 에서 105 → 78 로 줄어드는데, FP16 이 포맷 통일성이 좋아 변환이 덜 필요하기 때문.

### B.4 pool / resize

- **`CaskPooling`** ×3 — `model.9` SPPF 의 `MaxPool` 3연속. FP32 tactic `..._pooling_max_nhwc_FP32FP32_WINDOWSIZE_5_...`, FP16 `..._pooling_coalescedC_NHWC_kMAX_3_...` (SPPF 는 5×5 를 3×3 두 번으로 등가 분해하기도 함).
- **`Resize`** ×2 — `model.11`/`model.14` 의 nearest 업샘플 ×2. `InterpolationMode`/`ResizeScales` 필드로 확인.

### B.5 myelin 융합 영역 (attention)

`model.10`(C2PSA)·`model.22`(C3k2 with attention) 의 self-attention 부분은 TensorRT 의 **myelin** 백엔드가 통째로 잡아 별도 커널군으로 만든다. 이름에 `__myl_*`, `_myl109_`, `_mye4273_` 같은 리전 ID 가 붙는다.

| 커널 | 이름 예 | 정체 |
|---|---|---|
| `gemm` | `/model_10/m/m_0/attn/MatMul_myl109_2`, `..._MatMul_1_myl109_5` | Q·Kᵀ 와 (softmax)·V 의 행렬곱 |
| `kgen` | `__myl_MovSliSliTra_...` | Move + Slice×2 + Transpose (Q/K/V 분리·전치) |
| `kgen` | `__myl_MulMaxSubExpSum_...` | **softmax** 그 자체 (scale Mul → Max → Sub → Exp → Sum) |
| `kgen` | `__myl_DivMulTra_...` | softmax 정규화 나눗셈 + scale + transpose |
| `kgen` | `__myl_Add_...` | attention residual add |
| `shape_call` | `dummy_shape_call__mye4273_0_myl109_0` | end2end 후처리에서 동적 shape 계산 (실연산 아님) |

`kgen` mnemonic 은 융합된 연산의 약자 나열이다 (`Mul`,`Max`,`Sub`,`Exp`,`Sum`,`Div`,`Tra`=transpose,`Sli`=slice,`Mov`=move,`Add`). FP32 15개 / FP16 13개.

---

## C. FP32 → FP16 무엇이 달라졌나 (316 → 221, −95)

| 커널 | FP32 | FP16 | 변화 요인 |
|---|--:|--:|---|
| `PointWiseV2` | 87 | 21 | **−66** SiLU/Add 가 conv 에필로그로 융합 |
| `NoOp` | 61 | 36 | −25 포맷 통일로 경계 복사 감소 |
| `CaskGemmConvolution` | 29 | 13 | −16 상당수가 `CaskConvolution`/`CaskJitConv` 로 재배정 |
| `kgen` | 15 | 13 | −2 |
| `Reformat` | 44 | 42 | −2 |
| `CaskConvolution` | 68 | 77 | +9 (GEMM/Jit 에서 넘어옴) |
| `CaskJitConv` | 0 | 7 | **+7** FP16 에서만 conv+pointwise JIT 융합 |
| pool / resize / gemm / shape_call | 동일 | 동일 | 정밀도 영향 없음 |

**핵심**: FP16 은 (1) SiLU 활성을 합성곱 커널 안으로 JIT 융합할 수 있고, (2) 텐서 포맷이 통일적이라 `Reformat`/`NoOp` 삽입이 덜 필요하다. 그래서 pointwise·noop 커널이 대거 사라지고 전체 커널 수가 30% 준다. 정확도는 거의 동일(mAP50-95 0.8534 → 0.8410, Δ −0.0124 — [`eval_pt_vs_trt_fp16.md`](../eval_pt_vs_trt_fp16.md)), 추론은 오히려 FP32 엔진보다 1.9× 빠르다.

---

## 참고

- 단계별(`model.N`) 커널 집계: [`pt_vs_trt.md`](../pt_vs_trt.md) (FP32) · [`pt_vs_trt_fp16.md`](../pt_vs_trt_fp16.md) (FP16)
- `LayerType` 원본 문자열 → 8계열 매핑: `TensorRT/TRT_visualize_model.py` 의 `FAMILIES` / `family()`
- `trt.LayerType` 49종 재확인:
  ```bash
  conda activate yolo
  python -c "import tensorrt as trt; print(len([x for x in dir(trt.LayerType) if x.isupper()]))"
  ```
