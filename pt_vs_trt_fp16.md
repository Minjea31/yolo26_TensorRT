# `.pt` vs TensorRT 구조 비교

원본 `best.pt` 의 nn.Module(`model.N`) 이 TensorRT 엔진에서 커널 몇 개로 바뀌었는지 `.pt` 기준으로 나란히 본 문서. (`TensorRT/compare_pt_trt.py` 자동 생성)

- 원본 가중치: `model/best.pt`
- 엔진 레이어 정보: `model/best_fp16.engine.layers.json`
- 엔진 IO: `['images', 'output0']`
- 생성일: 2026-09-09

## 요약

| | 값 |
|---|---|
| `.pt` 모듈 | 24개 · 내부 leaf 레이어 277개 · params 2.5M |
| TRT 커널 | 221개 (매핑 218 + post 3) |
| 전체 흐름 | `.pt` leaf 277 → TRT 커널 218 |
| 레이아웃 변환만 남은 모듈 | 4개 (model.12, model.15, model.18, model.21) — Concat 은 zero-copy, reformat 1개씩만 |

> 커널이 가장 많은 단계: **model.23** (Detect, 43개 — end2end decode/NMS 까지 펼쳐짐).

## 모듈별 대응표

`레이어 수 (.pt→TRT)` = 그 `model.N` 의 **`.pt` 내부 leaf 서브모듈 수** → **TRT 커널 수**. `.pt` 모듈 1개엔 '커널 수'가 없으므로 안에 든 Conv2d·BN·SiLU·MaxPool 등을 센다 (SiLU 는 공유돼 1로 세짐, `torch.cat`·`+` 같은 함수형 연산은 안 세짐 — 어림값). `.pt` 출력 shape 는 forward 관측값, `TRT 출력` 은 그 단계 마지막 커널의 출력.

| 단계 | `.pt` 타입 | params | `.pt` 출력 | 레이어 수 (.pt→TRT) | TRT 커널 타입 | TRT 출력 | 비고 |
|---|---|--:|---|:--:|---|---|---|
| **model.0** (backbone) | Conv | 464 | 1x16x320x320 | 3 → 4 | reformat×1 conv×1 noop×1 pointwise×1 | 1x16x320x320 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 2개 삽입 |
| **model.1** (backbone) | Conv | 4.7K | 1x32x160x160 | 3 → 2 | noop×1 conv×1 | 1x32x160x160 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 1개 삽입 |
| **model.2** (backbone) | C3k2 | 6.6K | 1x64x160x160 | 9 → 12 | conv×4 noop×4 reformat×4 | 1x64x160x160 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 8개 삽입 |
| **model.3** (backbone) | Conv | 37.0K | 1x64x80x80 | 3 → 1 | conv×1 | 1x64x80x80 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.4** (backbone) | C3k2 | 26.1K | 1x128x80x80 | 9 → 9 | reformat×4 conv×3 caskjitconv×1 noop×1 | 1x128x80x80 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 5개 삽입 |
| **model.5** (backbone) | Conv | 147.7K | 1x128x40x40 | 3 → 1 | conv×1 | 1x128x40x40 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.6** (backbone) | C3k2 | 87.0K | 1x128x40x40 | 19 → 13 | conv×5 reformat×3 pointwise×2 caskjitconv×2 gemm×1 | 1x128x40x40 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 3개 삽입 |
| **model.7** (backbone) | Conv | 295.4K | 1x256x20x20 | 3 → 1 | conv×1 | 1x256x20x20 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.8** (backbone) | C3k2 | 346.1K | 1x256x20x20 | 19 → 20 | conv×7 noop×6 pointwise×4 reformat×2 gemm×1 | 1x256x20x20 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 8개 삽입 |
| **model.9** (backbone) | SPPF | 164.6K | 1x256x20x20 | 7 → 13 | noop×4 pool×3 reformat×3 gemm×2 pointwise×1 | 1x256x20x20 · f16 | 레이아웃 변환 7개 삽입 |
| **model.10** (backbone) | C2PSA | 249.7K | 1x256x20x20 | 19 → 27 | noop×8 reformat×6 gemm×5 conv×4 post×4 | 1x256x20x20 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 14개 삽입 |
| **model.11** (head) | Upsample | 0 | 1x256x40x40 | 1 → 2 | noop×1 resize×1 | 1x256x40x40 · f16 | 레이아웃 변환 1개 삽입 |
| **model.12** (head) | Concat | 0 | 1x384x40x40 | 1 → 1 | reformat×1 | 1x256x40x40 · f16 | 일부만 커널로 남음 (대부분 zero-copy) |
| **model.13** (head) | C3k2 | 119.8K | 1x128x40x40 | 19 → 13 | conv×5 reformat×3 pointwise×2 caskjitconv×2 gemm×1 | 1x128x40x40 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 3개 삽입 |
| **model.14** (head) | Upsample | 0 | 1x128x80x80 | 1 → 1 | resize×1 | 1x128x80x80 · f16 |  |
| **model.15** (head) | Concat | 0 | 1x256x80x80 | 1 → 1 | reformat×1 | 1x128x80x80 · f16 | 일부만 커널로 남음 (대부분 zero-copy) |
| **model.16** (head) | C3k2 | 34.3K | 1x64x80x80 | 19 → 13 | conv×8 reformat×3 pointwise×2 | 1x64x80x80 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 3개 삽입 |
| **model.17** (head) | Conv | 37.0K | 1x64x40x40 | 3 → 1 | conv×1 | 1x64x40x40 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.18** (head) | Concat | 0 | 1x192x40x40 | 1 → 1 | reformat×1 | 1x128x40x40 · f16 | 일부만 커널로 남음 (대부분 zero-copy) |
| **model.19** (head) | C3k2 | 95.2K | 1x128x40x40 | 19 → 13 | conv×5 reformat×3 pointwise×2 caskjitconv×2 gemm×1 | 1x128x40x40 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 3개 삽입 |
| **model.20** (head) | Conv | 147.7K | 1x128x20x20 | 3 → 1 | conv×1 | 1x128x20x20 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.21** (head) | Concat | 0 | 1x384x20x20 | 1 → 1 | reformat×1 | 1x256x20x20 · f16 | 일부만 커널로 남음 (대부분 zero-copy) |
| **model.22** (head) | C3k2 | 463.1K | 1x256x20x20 | 23 → 24 | conv×6 reformat×6 gemm×5 post×4 noop×2 pointwise×1 | 1x256x20x20 · f16 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 8개 삽입 |
| **model.23** (detect) | Detect | 241.6K | 1x300x6 | 88 → 43 | conv×23 noop×8 pointwise×6 post×5 gemm×1 | 1x300x6 · f16/f32 | end2end decode+NMS 펼침 (`i64` 텐서 등장) |
| _post-process_ | — | — | — | — → 3 | shape_call×3 |  | end2end 후처리 / 이름 매핑 안 되는 내부 커널 |

## 단계별 TRT 커널 목록

<details><summary><b>model.0</b> (Conv) — .pt leaf 3 → 커널 4개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.0/conv/Conv` |
| 1 | CaskConvolution | `/model.0/conv/Conv` |
| 2 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.0/act/Sigmoid), PWN(/model.0/act/Mul))` |
| 3 | PointWiseV2 | `PWN(PWN(/model.0/act/Sigmoid), PWN(/model.0/act/Mul))` |

</details>

<details><summary><b>model.1</b> (Conv) — .pt leaf 3 → 커널 2개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), PWN(/model.1/act/Mul))` |
| 1 | CaskConvolution | `/model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), PWN(/model.1/act/Mul))` |

</details>

<details><summary><b>model.2</b> (C3k2) — .pt leaf 9 → 커널 12개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.2/cv1/conv/Conv + PWN(PWN(/model.2/cv1/act/Sigmoid), PWN(/model.2/cv1/act/Mul))` |
| 1 | NoOp | `Reformatting CopyNode for Output Tensor 0 to /model.2/cv1/conv/Conv + PWN(PWN(/model.2/cv1/act/Sigmoid), PWN(/model.2/cv1/act/Mul))` |
| 2 | Reformat | `/model.2/Split_1` |
| 3 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.2/m.0/cv1/conv/Conv + PWN(PWN(/model.2/m.0/cv1/act/Sigmoid), PWN(/model.2/m.0/cv1/act/Mul))` |
| 4 | CaskConvolution | `/model.2/m.0/cv1/conv/Conv + PWN(PWN(/model.2/m.0/cv1/act/Sigmoid), PWN(/model.2/m.0/cv1/act/Mul))` |
| 5 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.2/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.2/m.0/cv2/act/Sigmoid), PWN(/model.2/m.0/cv2/act/Mul)), PWN(/model.2/m.0/Add))` |
| 6 | CaskConvolution | `/model.2/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.2/m.0/cv2/act/Sigmoid), PWN(/model.2/m.0/cv2/act/Mul)), PWN(/model.2/m.0/Add))` |
| 7 | Reformat | `/model.2/Split_output_0 copy` |
| 8 | Reformat | `/model.2/Split_output_1 copy` |
| 9 | Reformat | `/model.2/m.0/Add_output_0 copy` |
| 10 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.2/cv2/conv/Conv + PWN(PWN(/model.2/cv2/act/Sigmoid), PWN(/model.2/cv2/act/Mul))` |
| 11 | CaskConvolution | `/model.2/cv2/conv/Conv + PWN(PWN(/model.2/cv2/act/Sigmoid), PWN(/model.2/cv2/act/Mul))` |

</details>

<details><summary><b>model.3</b> (Conv) — .pt leaf 3 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.3/conv/Conv + PWN(PWN(/model.3/act/Sigmoid), PWN(/model.3/act/Mul))` |

</details>

<details><summary><b>model.4</b> (C3k2) — .pt leaf 9 → 커널 9개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.4/cv1/conv/Conv + PWN(PWN(/model.4/cv1/act/Sigmoid), PWN(/model.4/cv1/act/Mul))` |
| 1 | Reformat | `/model.4/Split_4` |
| 2 | CaskConvolution | `/model.4/m.0/cv1/conv/Conv + PWN(PWN(/model.4/m.0/cv1/act/Sigmoid), PWN(/model.4/m.0/cv1/act/Mul))` |
| 3 | CaskJitConv | `/model.4/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.4/m.0/cv2/act/Sigmoid), PWN(/model.4/m.0/cv2/act/Mul)), PWN(/model.4/m.0/Add))` |
| 4 | Reformat | `/model.4/Split_output_0 copy` |
| 5 | Reformat | `/model.4/Split_output_1 copy` |
| 6 | Reformat | `/model.4/m.0/Add_output_0 copy` |
| 7 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.4/cv2/conv/Conv + PWN(PWN(/model.4/cv2/act/Sigmoid), PWN(/model.4/cv2/act/Mul))` |
| 8 | CaskConvolution | `/model.4/cv2/conv/Conv + PWN(PWN(/model.4/cv2/act/Sigmoid), PWN(/model.4/cv2/act/Mul))` |

</details>

<details><summary><b>model.5</b> (Conv) — .pt leaf 3 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.5/conv/Conv + PWN(PWN(/model.5/act/Sigmoid), PWN(/model.5/act/Mul))` |

</details>

<details><summary><b>model.6</b> (C3k2) — .pt leaf 19 → 커널 13개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.6/cv1/conv/Conv + PWN(PWN(/model.6/cv1/act/Sigmoid), PWN(/model.6/cv1/act/Mul))` |
| 1 | CaskGemmConvolution | `/model.6/m.0/cv1/conv/Conv \|\| /model.6/m.0/cv2/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.6/m.0/cv1/act/Sigmoid), PWN(/model.6/m.0/cv1/act/Mul))` |
| 3 | CaskConvolution | `/model.6/m.0/m/m.0/cv1/conv/Conv + PWN(PWN(/model.6/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.6/m.0/m/m.0/cv1/act/Mul))` |
| 4 | CaskJitConv | `/model.6/m.0/m/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.6/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.6/m.0/m/m.0/cv2/act/Mul)), PWN(/model.6/m.0/m/m.0/Add))` |
| 5 | CaskConvolution | `/model.6/m.0/m/m.1/cv1/conv/Conv + PWN(PWN(/model.6/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.6/m.0/m/m.1/cv1/act/Mul))` |
| 6 | CaskJitConv | `/model.6/m.0/m/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.6/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.6/m.0/m/m.1/cv2/act/Mul)), PWN(/model.6/m.0/m/m.1/Add))` |
| 7 | PointWiseV2 | `PWN(PWN(/model.6/m.0/cv2/act/Sigmoid), PWN(/model.6/m.0/cv2/act/Mul))` |
| 8 | Reformat | `/model.6/m.0/m/m.1/Add_output_0 copy` |
| 9 | CaskConvolution | `/model.6/m.0/cv3/conv/Conv + PWN(PWN(/model.6/m.0/cv3/act/Sigmoid), PWN(/model.6/m.0/cv3/act/Mul))` |
| 10 | Reformat | `/model.6/Split_output_0 copy` |
| 11 | Reformat | `/model.6/Split_output_1 copy` |
| 12 | CaskConvolution | `/model.6/cv2/conv/Conv + PWN(PWN(/model.6/cv2/act/Sigmoid), PWN(/model.6/cv2/act/Mul))` |

</details>

<details><summary><b>model.7</b> (Conv) — .pt leaf 3 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.7/conv/Conv + PWN(PWN(/model.7/act/Sigmoid), PWN(/model.7/act/Mul))` |

</details>

<details><summary><b>model.8</b> (C3k2) — .pt leaf 19 → 커널 20개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.8/cv1/conv/Conv + PWN(PWN(/model.8/cv1/act/Sigmoid), PWN(/model.8/cv1/act/Mul))` |
| 1 | CaskGemmConvolution | `/model.8/m.0/cv1/conv/Conv \|\| /model.8/m.0/cv2/conv/Conv` |
| 2 | NoOp | `Reformatting CopyNode for Output Tensor 0 to /model.8/m.0/cv1/conv/Conv \|\| /model.8/m.0/cv2/conv/Conv` |
| 3 | PointWiseV2 | `PWN(PWN(/model.8/m.0/cv1/act/Sigmoid), PWN(/model.8/m.0/cv1/act/Mul))` |
| 4 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.8/m.0/m/m.0/cv1/conv/Conv + PWN(PWN(/model.8/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.8/m.0/m/m.0/cv1/act/Mul))` |
| 5 | CaskConvolution | `/model.8/m.0/m/m.0/cv1/conv/Conv + PWN(PWN(/model.8/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.8/m.0/m/m.0/cv1/act/Mul))` |
| 6 | CaskConvolution | `/model.8/m.0/m/m.0/cv2/conv/Conv` |
| 7 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.8/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.8/m.0/m/m.0/cv2/act/Mul)), PWN(/model.8/m.0/m/m.0/Add))` |
| 8 | PointWiseV2 | `PWN(PWN(PWN(/model.8/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.8/m.0/m/m.0/cv2/act/Mul)), PWN(/model.8/m.0/m/m.0/Add))` |
| 9 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.8/m.0/m/m.1/cv1/conv/Conv + PWN(PWN(/model.8/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.8/m.0/m/m.1/cv1/act/Mul))` |
| 10 | CaskConvolution | `/model.8/m.0/m/m.1/cv1/conv/Conv + PWN(PWN(/model.8/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.8/m.0/m/m.1/cv1/act/Mul))` |
| 11 | CaskConvolution | `/model.8/m.0/m/m.1/cv2/conv/Conv` |
| 12 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.8/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.8/m.0/m/m.1/cv2/act/Mul)), PWN(/model.8/m.0/m/m.1/Add))` |
| 13 | PointWiseV2 | `PWN(PWN(PWN(/model.8/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.8/m.0/m/m.1/cv2/act/Mul)), PWN(/model.8/m.0/m/m.1/Add))` |
| 14 | PointWiseV2 | `PWN(PWN(/model.8/m.0/cv2/act/Sigmoid), PWN(/model.8/m.0/cv2/act/Mul))` |
| 15 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.8/m.0/cv3/conv/Conv + PWN(PWN(/model.8/m.0/cv3/act/Sigmoid), PWN(/model.8/m.0/cv3/act/Mul))` |
| 16 | CaskConvolution | `/model.8/m.0/cv3/conv/Conv + PWN(PWN(/model.8/m.0/cv3/act/Sigmoid), PWN(/model.8/m.0/cv3/act/Mul))` |
| 17 | Reformat | `/model.8/Split_output_0 copy` |
| 18 | Reformat | `/model.8/Split_output_1 copy` |
| 19 | CaskConvolution | `/model.8/cv2/conv/Conv + PWN(PWN(/model.8/cv2/act/Sigmoid), PWN(/model.8/cv2/act/Mul))` |

</details>

<details><summary><b>model.9</b> (SPPF) — .pt leaf 7 → 커널 13개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskGemmConvolution | `/model.9/cv1/conv/Conv` |
| 1 | NoOp | `Reformatting CopyNode for Output Tensor 0 to /model.9/cv1/conv/Conv` |
| 2 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.9/m/MaxPool` |
| 3 | CaskPooling | `/model.9/m/MaxPool` |
| 4 | CaskPooling | `/model.9/m_1/MaxPool` |
| 5 | CaskPooling | `/model.9/m_2/MaxPool` |
| 6 | Reformat | `/model.9/cv1/conv/Conv_output_0 copy` |
| 7 | Reformat | `/model.9/m/MaxPool_output_0 copy` |
| 8 | Reformat | `/model.9/m_1/MaxPool_output_0 copy` |
| 9 | CaskGemmConvolution | `/model.9/cv2/conv/Conv` |
| 10 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.9/cv2/act/Sigmoid), PWN(/model.9/cv2/act/Mul)), PWN(/model.9/Add))` |
| 11 | NoOp | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.9/cv2/act/Sigmoid), PWN(/model.9/cv2/act/Mul)), PWN(/model.9/Add))` |
| 12 | PointWiseV2 | `PWN(PWN(PWN(/model.9/cv2/act/Sigmoid), PWN(/model.9/cv2/act/Mul)), PWN(/model.9/Add))` |

</details>

<details><summary><b>model.10</b> (C2PSA) — .pt leaf 19 → 커널 27개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.10/cv1/conv/Conv + PWN(PWN(/model.10/cv1/act/Sigmoid), PWN(/model.10/cv1/act/Mul))` |
| 1 | CaskConvolution | `/model.10/cv1/conv/Conv + PWN(PWN(/model.10/cv1/act/Sigmoid), PWN(/model.10/cv1/act/Mul))` |
| 2 | NoOp | `Reformatting CopyNode for Output Tensor 0 to /model.10/cv1/conv/Conv + PWN(PWN(/model.10/cv1/act/Sigmoid), PWN(/model.10/cv1/act/Mul))` |
| 3 | Reformat | `/model.10/Split_13` |
| 4 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.10/m/m.0/attn/qkv/conv/Conv` |
| 5 | CaskGemmConvolution | `/model.10/m/m.0/attn/qkv/conv/Conv` |
| 6 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.10/m/m.0/attn/Reshape` |
| 7 | NoOp | `/model.10/m/m.0/attn/Reshape` |
| 8 | Reformat | `/model.10/m/m.0/attn/Split_18` |
| 9 | NoOp | `/model.10/m/m.0/attn/Reshape_2` |
| 10 | CaskConvolution | `/model.10/m/m.0/attn/pe/conv/Conv` |
| 11 | kgen | `__myl_MovSliSliTra_myl87_1` |
| 12 | gemm | `/model_10/m/m_0/attn/MatMul_myl87_2` |
| 13 | kgen | `__myl_MulMaxSubExpSum_myl87_3` |
| 14 | kgen | `__myl_DivMulTra_myl87_4` |
| 15 | gemm | `/model_10/m/m_0/attn/MatMul_1_myl87_5` |
| 16 | kgen | `__myl_Add_myl87_6` |
| 17 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.10/m/m.0/attn/proj/conv/Conv + /model.10/m/m.0/Add` |
| 18 | NoOp | `Reformatting CopyNode for Input Tensor 1 to /model.10/m/m.0/attn/proj/conv/Conv + /model.10/m/m.0/Add` |
| 19 | CaskGemmConvolution | `/model.10/m/m.0/attn/proj/conv/Conv + /model.10/m/m.0/Add` |
| 20 | CaskConvolution | `/model.10/m/m.0/ffn/ffn.0/conv/Conv + PWN(PWN(/model.10/m/m.0/ffn/ffn.0/act/Sigmoid), PWN(/model.10/m/m.0/ffn/ffn.0/act/Mul))` |
| 21 | CaskGemmConvolution | `/model.10/m/m.0/ffn/ffn.1/conv/Conv + /model.10/m/m.0/Add_1` |
| 22 | Reformat | `/model.10/Split_output_0 copy` |
| 23 | Reformat | `/model.10/m/m.0/Add_1_output_0 copy` |
| 24 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.10/cv2/conv/Conv + PWN(PWN(/model.10/cv2/act/Sigmoid), PWN(/model.10/cv2/act/Mul))` |
| 25 | CaskConvolution | `/model.10/cv2/conv/Conv + PWN(PWN(/model.10/cv2/act/Sigmoid), PWN(/model.10/cv2/act/Mul))` |
| 26 | NoOp | `Reformatting CopyNode for Output Tensor 0 to /model.10/cv2/conv/Conv + PWN(PWN(/model.10/cv2/act/Sigmoid), PWN(/model.10/cv2/act/Mul))` |

</details>

<details><summary><b>model.11</b> (Upsample) — .pt leaf 1 → 커널 2개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.11/Resize` |
| 1 | Resize | `/model.11/Resize` |

</details>

<details><summary><b>model.12</b> (Concat) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `/model.11/Resize_output_0 copy` |

</details>

<details><summary><b>model.13</b> (C3k2) — .pt leaf 19 → 커널 13개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.13/cv1/conv/Conv + PWN(PWN(/model.13/cv1/act/Sigmoid), PWN(/model.13/cv1/act/Mul))` |
| 1 | CaskGemmConvolution | `/model.13/m.0/cv1/conv/Conv \|\| /model.13/m.0/cv2/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.13/m.0/cv1/act/Sigmoid), PWN(/model.13/m.0/cv1/act/Mul))` |
| 3 | CaskConvolution | `/model.13/m.0/m/m.0/cv1/conv/Conv + PWN(PWN(/model.13/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.13/m.0/m/m.0/cv1/act/Mul))` |
| 4 | CaskJitConv | `/model.13/m.0/m/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.13/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.13/m.0/m/m.0/cv2/act/Mul)), PWN(/model.13/m.0/m/m.0/Add))` |
| 5 | CaskConvolution | `/model.13/m.0/m/m.1/cv1/conv/Conv + PWN(PWN(/model.13/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.13/m.0/m/m.1/cv1/act/Mul))` |
| 6 | CaskJitConv | `/model.13/m.0/m/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.13/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.13/m.0/m/m.1/cv2/act/Mul)), PWN(/model.13/m.0/m/m.1/Add))` |
| 7 | PointWiseV2 | `PWN(PWN(/model.13/m.0/cv2/act/Sigmoid), PWN(/model.13/m.0/cv2/act/Mul))` |
| 8 | Reformat | `/model.13/m.0/m/m.1/Add_output_0 copy` |
| 9 | CaskConvolution | `/model.13/m.0/cv3/conv/Conv + PWN(PWN(/model.13/m.0/cv3/act/Sigmoid), PWN(/model.13/m.0/cv3/act/Mul))` |
| 10 | Reformat | `/model.13/Split_output_0 copy` |
| 11 | Reformat | `/model.13/Split_output_1 copy` |
| 12 | CaskConvolution | `/model.13/cv2/conv/Conv + PWN(PWN(/model.13/cv2/act/Sigmoid), PWN(/model.13/cv2/act/Mul))` |

</details>

<details><summary><b>model.14</b> (Upsample) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Resize | `/model.14/Resize` |

</details>

<details><summary><b>model.15</b> (Concat) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `/model.14/Resize_output_0 copy` |

</details>

<details><summary><b>model.16</b> (C3k2) — .pt leaf 19 → 커널 13개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.16/cv1/conv/Conv + PWN(PWN(/model.16/cv1/act/Sigmoid), PWN(/model.16/cv1/act/Mul))` |
| 1 | CaskConvolution | `/model.16/m.0/cv1/conv/Conv \|\| /model.16/m.0/cv2/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.16/m.0/cv1/act/Sigmoid), PWN(/model.16/m.0/cv1/act/Mul))` |
| 3 | CaskConvolution | `/model.16/m.0/m/m.0/cv1/conv/Conv + PWN(PWN(/model.16/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.16/m.0/m/m.0/cv1/act/Mul))` |
| 4 | CaskConvolution | `/model.16/m.0/m/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.16/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.16/m.0/m/m.0/cv2/act/Mul)), PWN(/model.16/m.0/m/m.0/Add))` |
| 5 | CaskConvolution | `/model.16/m.0/m/m.1/cv1/conv/Conv + PWN(PWN(/model.16/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.16/m.0/m/m.1/cv1/act/Mul))` |
| 6 | CaskConvolution | `/model.16/m.0/m/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.16/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.16/m.0/m/m.1/cv2/act/Mul)), PWN(/model.16/m.0/m/m.1/Add))` |
| 7 | PointWiseV2 | `PWN(PWN(/model.16/m.0/cv2/act/Sigmoid), PWN(/model.16/m.0/cv2/act/Mul))` |
| 8 | Reformat | `/model.16/m.0/m/m.1/Add_output_0 copy` |
| 9 | CaskConvolution | `/model.16/m.0/cv3/conv/Conv + PWN(PWN(/model.16/m.0/cv3/act/Sigmoid), PWN(/model.16/m.0/cv3/act/Mul))` |
| 10 | Reformat | `/model.16/Split_output_0 copy` |
| 11 | Reformat | `/model.16/Split_output_1 copy` |
| 12 | CaskConvolution | `/model.16/cv2/conv/Conv + PWN(PWN(/model.16/cv2/act/Sigmoid), PWN(/model.16/cv2/act/Mul))` |

</details>

<details><summary><b>model.17</b> (Conv) — .pt leaf 3 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.17/conv/Conv + PWN(PWN(/model.17/act/Sigmoid), PWN(/model.17/act/Mul))` |

</details>

<details><summary><b>model.18</b> (Concat) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `/model.13/cv2/act/Mul_output_0 copy` |

</details>

<details><summary><b>model.19</b> (C3k2) — .pt leaf 19 → 커널 13개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.19/cv1/conv/Conv + PWN(PWN(/model.19/cv1/act/Sigmoid), PWN(/model.19/cv1/act/Mul))` |
| 1 | CaskGemmConvolution | `/model.19/m.0/cv1/conv/Conv \|\| /model.19/m.0/cv2/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.19/m.0/cv1/act/Sigmoid), PWN(/model.19/m.0/cv1/act/Mul))` |
| 3 | CaskConvolution | `/model.19/m.0/m/m.0/cv1/conv/Conv + PWN(PWN(/model.19/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.19/m.0/m/m.0/cv1/act/Mul))` |
| 4 | CaskJitConv | `/model.19/m.0/m/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.19/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.19/m.0/m/m.0/cv2/act/Mul)), PWN(/model.19/m.0/m/m.0/Add))` |
| 5 | CaskConvolution | `/model.19/m.0/m/m.1/cv1/conv/Conv + PWN(PWN(/model.19/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.19/m.0/m/m.1/cv1/act/Mul))` |
| 6 | CaskJitConv | `/model.19/m.0/m/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.19/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.19/m.0/m/m.1/cv2/act/Mul)), PWN(/model.19/m.0/m/m.1/Add))` |
| 7 | PointWiseV2 | `PWN(PWN(/model.19/m.0/cv2/act/Sigmoid), PWN(/model.19/m.0/cv2/act/Mul))` |
| 8 | Reformat | `/model.19/m.0/m/m.1/Add_output_0 copy` |
| 9 | CaskConvolution | `/model.19/m.0/cv3/conv/Conv + PWN(PWN(/model.19/m.0/cv3/act/Sigmoid), PWN(/model.19/m.0/cv3/act/Mul))` |
| 10 | Reformat | `/model.19/Split_output_0 copy` |
| 11 | Reformat | `/model.19/Split_output_1 copy` |
| 12 | CaskConvolution | `/model.19/cv2/conv/Conv + PWN(PWN(/model.19/cv2/act/Sigmoid), PWN(/model.19/cv2/act/Mul))` |

</details>

<details><summary><b>model.20</b> (Conv) — .pt leaf 3 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.20/conv/Conv + PWN(PWN(/model.20/act/Sigmoid), PWN(/model.20/act/Mul))` |

</details>

<details><summary><b>model.21</b> (Concat) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `/model.10/cv2/act/Mul_output_0 copy` |

</details>

<details><summary><b>model.22</b> (C3k2) — .pt leaf 23 → 커널 24개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.22/cv1/conv/Conv + PWN(PWN(/model.22/cv1/act/Sigmoid), PWN(/model.22/cv1/act/Mul))` |
| 1 | CaskConvolution | `/model.22/m.0/m.0.0/cv1/conv/Conv + PWN(PWN(/model.22/m.0/m.0.0/cv1/act/Sigmoid), PWN(/model.22/m.0/m.0.0/cv1/act/Mul))` |
| 2 | CaskConvolution | `/model.22/m.0/m.0.0/cv2/conv/Conv` |
| 3 | PointWiseV2 | `PWN(PWN(PWN(/model.22/m.0/m.0.0/cv2/act/Sigmoid), PWN(/model.22/m.0/m.0.0/cv2/act/Mul)), PWN(/model.22/m.0/m.0.0/Add))` |
| 4 | CaskGemmConvolution | `/model.22/m.0/m.0.1/attn/qkv/conv/Conv` |
| 5 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.22/m.0/m.0.1/attn/Reshape` |
| 6 | NoOp | `/model.22/m.0/m.0.1/attn/Reshape` |
| 7 | Reformat | `/model.22/m.0/m.0.1/attn/Split_40` |
| 8 | NoOp | `/model.22/m.0/m.0.1/attn/Reshape_2` |
| 9 | CaskConvolution | `/model.22/m.0/m.0.1/attn/pe/conv/Conv` |
| 10 | kgen | `__myl_MovSliSliTra_myl156_1` |
| 11 | gemm | `/model_22/m_0/m_0_1/attn/MatMul_myl156_2` |
| 12 | kgen | `__myl_MulMaxSubExpSum_myl156_3` |
| 13 | kgen | `__myl_DivMulTra_myl156_4` |
| 14 | gemm | `/model_22/m_0/m_0_1/attn/MatMul_1_myl156_5` |
| 15 | kgen | `__myl_Add_myl156_6` |
| 16 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.22/m.0/m.0.1/attn/proj/conv/Conv + /model.22/m.0/m.0.1/Add` |
| 17 | CaskGemmConvolution | `/model.22/m.0/m.0.1/attn/proj/conv/Conv + /model.22/m.0/m.0.1/Add` |
| 18 | CaskConvolution | `/model.22/m.0/m.0.1/ffn/ffn.0/conv/Conv + PWN(PWN(/model.22/m.0/m.0.1/ffn/ffn.0/act/Sigmoid), PWN(/model.22/m.0/m.0.1/ffn/ffn.0/act/Mul))` |
| 19 | CaskGemmConvolution | `/model.22/m.0/m.0.1/ffn/ffn.1/conv/Conv + /model.22/m.0/m.0.1/Add_1` |
| 20 | Reformat | `/model.22/Split_output_0 copy` |
| 21 | Reformat | `/model.22/Split_output_1 copy` |
| 22 | Reformat | `/model.22/m.0/m.0.1/Add_1_output_0 copy` |
| 23 | CaskConvolution | `/model.22/cv2/conv/Conv + PWN(PWN(/model.22/cv2/act/Sigmoid), PWN(/model.22/cv2/act/Mul))` |

</details>

<details><summary><b>model.23</b> (Detect) — .pt leaf 88 → 커널 43개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv + PWN(PWN(/model.23/one2one_cv2.2/one2one_cv2.2.0/act/Sigmoid), PWN(/model.23/one2one_cv2.2/one2one_cv2.2.0/act/Mul))` |
| 1 | CaskConvolution | `/model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv + PWN(PWN(/model.23/one2one_cv2.2/one2one_cv2.2.1/act/Sigmoid), PWN(/model.23/one2one_cv2.2/one2one_cv2.2.1/act/Mul))` |
| 2 | CaskConvolution | `/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv` |
| 3 | CaskConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv` |
| 4 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/act/Mul))` |
| 5 | CaskConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/act/Mul))` |
| 6 | CaskConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv` |
| 7 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Mul))` |
| 8 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Mul))` |
| 9 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul))` |
| 10 | CaskConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul))` |
| 11 | CaskGemmConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv` |
| 12 | CaskConvolution | `/model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv + PWN(PWN(/model.23/one2one_cv2.1/one2one_cv2.1.0/act/Sigmoid), PWN(/model.23/one2one_cv2.1/one2one_cv2.1.0/act/Mul))` |
| 13 | CaskConvolution | `/model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv + PWN(PWN(/model.23/one2one_cv2.1/one2one_cv2.1.1/act/Sigmoid), PWN(/model.23/one2one_cv2.1/one2one_cv2.1.1/act/Mul))` |
| 14 | CaskConvolution | `/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv` |
| 15 | CaskConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv` |
| 16 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/act/Mul))` |
| 17 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/act/Mul))` |
| 18 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/act/Mul))` |
| 19 | CaskConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/act/Mul))` |
| 20 | CaskConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv` |
| 21 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/act/Mul))` |
| 22 | CaskConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul))` |
| 23 | CaskConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv` |
| 24 | CaskConvolution | `/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv + PWN(PWN(/model.23/one2one_cv2.0/one2one_cv2.0.0/act/Sigmoid), PWN(/model.23/one2one_cv2.0/one2one_cv2.0.0/act/Mul))` |
| 25 | CaskConvolution | `/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv + PWN(PWN(/model.23/one2one_cv2.0/one2one_cv2.0.1/act/Sigmoid), PWN(/model.23/one2one_cv2.0/one2one_cv2.0.1/act/Mul))` |
| 26 | CaskConvolution | `/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv` |
| 27 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv` |
| 28 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/act/Mul))` |
| 29 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/act/Mul))` |
| 30 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/act/Mul))` |
| 31 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/act/Mul))` |
| 32 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv` |
| 33 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/act/Mul))` |
| 34 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/act/Mul))` |
| 35 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul))` |
| 36 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv + PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul))` |
| 37 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv` |
| 38 | kgen | `__myl_ResResResConNegExpAddDivResResResConSliSubSliAddConMulConTraSliMaxSli_myl203_1` |
| 39 | kgen | `__myl_Top_myl203_2` |
| 40 | kgen | `__myl_CasResCasGatRes_myl203_3` |
| 41 | kgen | `__myl_Top_myl203_4` |
| 42 | kgen | `__myl_ResCasCasDivFloCasSubResCasDivGatCasRepGatConCas_myl203_5` |

</details>

<details><summary><b>post-process</b> — 커널 3개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | shape_call | `dummy_shape_call__mye4273_0_myl87_0` |
| 1 | shape_call | `dummy_shape_call__mye4273_0_myl156_0` |
| 2 | shape_call | `dummy_shape_call__mye23420_0_myl203_0` |

</details>

## 읽는 법

- **`레이어 수 (.pt→TRT)`** — 왼쪽은 `.pt` 모듈 안의 leaf 서브모듈 수(정확한 연산 수 아님, 어림값), 오른쪽은 그 단계로 매핑된 TRT 커널 수. `--onnx model/best.onnx` 를 주면 가운데에 ONNX 노드 수도 표시된다. ONNX 없이도 `.pt→TRT` 는 그대로 나온다.
- **커널 타입** 은 LayerType 계열명: `conv`(CaskConvolution) / `gemm`(MatMul·FullyConnected) / `pointwise`(활성화·elementwise) / `pool` / `reformat`·`noop`(레이아웃 변환) / `resize` / `post`(slice·concat·topk·nms).
- `.pt` 의 `Conv` = Conv2d+BN+SiLU. TRT 에서 BN 은 conv 가중치에 접히고(fold), SiLU(Sigmoid+Mul)는 `PointWiseV2` 1개로 fusion 된다.
- `Concat`(model.12/15/18/21) 은 대부분 zero-copy 로 처리돼 독립 커널이 거의 없다 → `1 → 1` (reformat 1개).
- `Detect`(model.23) 는 학습용 구조가 사라지고 추론 경로 + DFL + box decode + TopK/NMS 가 그래프로 펼쳐져 커널 수가 가장 많다.
