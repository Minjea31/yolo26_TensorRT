# `.pt` vs TensorRT 구조 비교

원본 `best.pt` 의 nn.Module(`model.N`) 이 TensorRT 엔진에서 커널 몇 개로 바뀌었는지 `.pt` 기준으로 나란히 본 문서. (`TensorRT/compare_pt_trt.py` 자동 생성)

- 원본 가중치: `model/best.pt`
- 엔진 레이어 정보: `model/best.engine.layers.json`
- 엔진 IO: `['images', 'output0']`
- 생성일: 2026-09-09

## 요약

| | 값 |
|---|---|
| `.pt` 모듈 | 24개 · 내부 leaf 레이어 277개 · params 2.5M |
| TRT 커널 | 316개 (매핑 313 + post 3) |
| 전체 흐름 | `.pt` leaf 277 → TRT 커널 313 |
| 레이아웃 변환만 남은 모듈 | 4개 (model.12, model.15, model.18, model.21) — Concat 은 zero-copy, reformat 1개씩만 |

> 커널이 가장 많은 단계: **model.23** (Detect, 66개 — end2end decode/NMS 까지 펼쳐짐).

## 모듈별 대응표

`레이어 수 (.pt→TRT)` = 그 `model.N` 의 **`.pt` 내부 leaf 서브모듈 수** → **TRT 커널 수**. `.pt` 모듈 1개엔 '커널 수'가 없으므로 안에 든 Conv2d·BN·SiLU·MaxPool 등을 센다 (SiLU 는 공유돼 1로 세짐, `torch.cat`·`+` 같은 함수형 연산은 안 세짐 — 어림값). `.pt` 출력 shape 는 forward 관측값, `TRT 출력` 은 그 단계 마지막 커널의 출력.

| 단계 | `.pt` 타입 | params | `.pt` 출력 | 레이어 수 (.pt→TRT) | TRT 커널 타입 | TRT 출력 | 비고 |
|---|---|--:|---|:--:|---|---|---|
| **model.0** (backbone) | Conv | 464 | 1x16x320x320 | 3 → 3 | reformat×1 conv×1 pointwise×1 | 1x16x320x320 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 1개 삽입 |
| **model.1** (backbone) | Conv | 4.7K | 1x32x160x160 | 3 → 3 | noop×1 conv×1 pointwise×1 | 1x32x160x160 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 1개 삽입 |
| **model.2** (backbone) | C3k2 | 6.6K | 1x64x160x160 | 9 → 10 | conv×4 pointwise×4 reformat×2 | 1x64x160x160 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 2개 삽입 |
| **model.3** (backbone) | Conv | 37.0K | 1x64x80x80 | 3 → 2 | conv×1 pointwise×1 | 1x64x80x80 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.4** (backbone) | C3k2 | 26.1K | 1x128x80x80 | 9 → 10 | pointwise×4 conv×3 reformat×2 gemm×1 | 1x128x80x80 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 2개 삽입 |
| **model.5** (backbone) | Conv | 147.7K | 1x128x40x40 | 3 → 2 | conv×1 pointwise×1 | 1x128x40x40 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.6** (backbone) | C3k2 | 87.0K | 1x128x40x40 | 19 → 26 | pointwise×9 conv×7 noop×6 reformat×3 gemm×1 | 1x128x40x40 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 9개 삽입 |
| **model.7** (backbone) | Conv | 295.4K | 1x256x20x20 | 3 → 3 | conv×1 noop×1 pointwise×1 | 1x256x20x20 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 1개 삽입 |
| **model.8** (backbone) | C3k2 | 346.1K | 1x256x20x20 | 19 → 25 | pointwise×9 noop×6 gemm×4 conv×4 reformat×2 | 1x256x20x20 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 8개 삽입 |
| **model.9** (backbone) | SPPF | 164.6K | 1x256x20x20 | 7 → 13 | noop×4 pool×3 reformat×3 gemm×2 pointwise×1 | 1x256x20x20 · f32 | 레이아웃 변환 7개 삽입 |
| **model.10** (backbone) | C2PSA | 249.7K | 1x256x20x20 | 19 → 30 | gemm×8 noop×7 reformat×7 post×4 pointwise×3 conv×1 | 1x256x20x20 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 14개 삽입 |
| **model.11** (head) | Upsample | 0 | 1x256x40x40 | 1 → 1 | resize×1 | 1x256x40x40 · f32 |  |
| **model.12** (head) | Concat | 0 | 1x384x40x40 | 1 → 1 | reformat×1 | 1x256x40x40 · f32 | 일부만 커널로 남음 (대부분 zero-copy) |
| **model.13** (head) | C3k2 | 119.8K | 1x128x40x40 | 19 → 26 | pointwise×9 conv×7 noop×6 reformat×3 gemm×1 | 1x128x40x40 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 9개 삽입 |
| **model.14** (head) | Upsample | 0 | 1x128x80x80 | 1 → 1 | resize×1 | 1x128x80x80 · f32 |  |
| **model.15** (head) | Concat | 0 | 1x256x80x80 | 1 → 1 | reformat×1 | 1x128x80x80 · f32 | 일부만 커널로 남음 (대부분 zero-copy) |
| **model.16** (head) | C3k2 | 34.3K | 1x64x80x80 | 19 → 24 | pointwise×9 conv×6 noop×4 reformat×3 gemm×2 | 1x64x80x80 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 7개 삽입 |
| **model.17** (head) | Conv | 37.0K | 1x64x40x40 | 3 → 2 | conv×1 pointwise×1 | 1x64x40x40 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.18** (head) | Concat | 0 | 1x192x40x40 | 1 → 1 | reformat×1 | 1x128x40x40 · f32 | 일부만 커널로 남음 (대부분 zero-copy) |
| **model.19** (head) | C3k2 | 95.2K | 1x128x40x40 | 19 → 25 | pointwise×9 conv×7 noop×5 reformat×3 gemm×1 | 1x128x40x40 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 8개 삽입 |
| **model.20** (head) | Conv | 147.7K | 1x128x20x20 | 3 → 2 | conv×1 pointwise×1 | 1x128x20x20 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise |
| **model.21** (head) | Concat | 0 | 1x384x20x20 | 1 → 1 | reformat×1 | 1x256x20x20 · f32 | 일부만 커널로 남음 (대부분 zero-copy) |
| **model.22** (head) | C3k2 | 463.1K | 1x256x20x20 | 23 → 35 | gemm×8 reformat×8 noop×7 pointwise×5 post×4 conv×3 | 1x256x20x20 · f32 | Conv+BN→conv 커널, SiLU(Sigmoid+Mul)→pointwise · 레이아웃 변환 15개 삽입 |
| **model.23** (detect) | Detect | 241.6K | 1x300x6 | 88 → 66 | conv×19 pointwise×18 noop×14 post×7 gemm×5 reformat×3 | 1x300x4 · f32/i64 | end2end decode+NMS 펼침 (`i64` 텐서 등장) |
| _post-process_ | — | — | — | — → 3 | shape_call×3 |  | end2end 후처리 / 이름 매핑 안 되는 내부 커널 |

## 단계별 TRT 커널 목록

<details><summary><b>model.0</b> (Conv) — .pt leaf 3 → 커널 3개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.0/conv/Conv` |
| 1 | CaskConvolution | `/model.0/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.0/act/Sigmoid), PWN(/model.0/act/Mul))` |

</details>

<details><summary><b>model.1</b> (Conv) — .pt leaf 3 → 커널 3개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.1/conv/Conv` |
| 1 | CaskConvolution | `/model.1/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.1/act/Sigmoid), PWN(/model.1/act/Mul))` |

</details>

<details><summary><b>model.2</b> (C3k2) — .pt leaf 9 → 커널 10개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.2/cv1/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.2/cv1/act/Sigmoid), PWN(/model.2/cv1/act/Mul))` |
| 2 | CaskConvolution | `/model.2/m.0/cv1/conv/Conv` |
| 3 | PointWiseV2 | `PWN(PWN(/model.2/m.0/cv1/act/Sigmoid), PWN(/model.2/m.0/cv1/act/Mul))` |
| 4 | CaskConvolution | `/model.2/m.0/cv2/conv/Conv` |
| 5 | PointWiseV2 | `PWN(PWN(PWN(/model.2/m.0/cv2/act/Sigmoid), PWN(/model.2/m.0/cv2/act/Mul)), PWN(/model.2/m.0/Add))` |
| 6 | Reformat | `/model.2/Split_output_0 copy` |
| 7 | Reformat | `/model.2/Split_output_1 copy` |
| 8 | CaskConvolution | `/model.2/cv2/conv/Conv` |
| 9 | PointWiseV2 | `PWN(PWN(/model.2/cv2/act/Sigmoid), PWN(/model.2/cv2/act/Mul))` |

</details>

<details><summary><b>model.3</b> (Conv) — .pt leaf 3 → 커널 2개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.3/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.3/act/Sigmoid), PWN(/model.3/act/Mul))` |

</details>

<details><summary><b>model.4</b> (C3k2) — .pt leaf 9 → 커널 10개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.4/cv1/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.4/cv1/act/Sigmoid), PWN(/model.4/cv1/act/Mul))` |
| 2 | CaskConvolution | `/model.4/m.0/cv1/conv/Conv` |
| 3 | PointWiseV2 | `PWN(PWN(/model.4/m.0/cv1/act/Sigmoid), PWN(/model.4/m.0/cv1/act/Mul))` |
| 4 | CaskConvolution | `/model.4/m.0/cv2/conv/Conv` |
| 5 | PointWiseV2 | `PWN(PWN(PWN(/model.4/m.0/cv2/act/Sigmoid), PWN(/model.4/m.0/cv2/act/Mul)), PWN(/model.4/m.0/Add))` |
| 6 | Reformat | `/model.4/Split_output_0 copy` |
| 7 | Reformat | `/model.4/Split_output_1 copy` |
| 8 | CaskGemmConvolution | `/model.4/cv2/conv/Conv` |
| 9 | PointWiseV2 | `PWN(PWN(/model.4/cv2/act/Sigmoid), PWN(/model.4/cv2/act/Mul))` |

</details>

<details><summary><b>model.5</b> (Conv) — .pt leaf 3 → 커널 2개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.5/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.5/act/Sigmoid), PWN(/model.5/act/Mul))` |

</details>

<details><summary><b>model.6</b> (C3k2) — .pt leaf 19 → 커널 26개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.6/cv1/conv/Conv` |
| 1 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.6/cv1/act/Sigmoid), PWN(/model.6/cv1/act/Mul))` |
| 2 | PointWiseV2 | `PWN(PWN(/model.6/cv1/act/Sigmoid), PWN(/model.6/cv1/act/Mul))` |
| 3 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(/model.6/cv1/act/Sigmoid), PWN(/model.6/cv1/act/Mul))` |
| 4 | CaskConvolution | `/model.6/m.0/cv2/conv/Conv \|\| /model.6/m.0/cv1/conv/Conv` |
| 5 | PointWiseV2 | `PWN(PWN(/model.6/m.0/cv1/act/Sigmoid), PWN(/model.6/m.0/cv1/act/Mul))` |
| 6 | CaskConvolution | `/model.6/m.0/m/m.0/cv1/conv/Conv` |
| 7 | PointWiseV2 | `PWN(PWN(/model.6/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.6/m.0/m/m.0/cv1/act/Mul))` |
| 8 | CaskConvolution | `/model.6/m.0/m/m.0/cv2/conv/Conv` |
| 9 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.6/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.6/m.0/m/m.0/cv2/act/Mul)), PWN(/model.6/m.0/m/m.0/Add))` |
| 10 | NoOp | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.6/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.6/m.0/m/m.0/cv2/act/Mul)), PWN(/model.6/m.0/m/m.0/Add))` |
| 11 | PointWiseV2 | `PWN(PWN(PWN(/model.6/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.6/m.0/m/m.0/cv2/act/Mul)), PWN(/model.6/m.0/m/m.0/Add))` |
| 12 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.6/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.6/m.0/m/m.0/cv2/act/Mul)), PWN(/model.6/m.0/m/m.0/Add))` |
| 13 | CaskConvolution | `/model.6/m.0/m/m.1/cv1/conv/Conv` |
| 14 | PointWiseV2 | `PWN(PWN(/model.6/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.6/m.0/m/m.1/cv1/act/Mul))` |
| 15 | CaskConvolution | `/model.6/m.0/m/m.1/cv2/conv/Conv` |
| 16 | PointWiseV2 | `PWN(PWN(/model.6/m.0/cv2/act/Sigmoid), PWN(/model.6/m.0/cv2/act/Mul))` |
| 17 | PointWiseV2 | `PWN(PWN(PWN(/model.6/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.6/m.0/m/m.1/cv2/act/Mul)), PWN(/model.6/m.0/m/m.1/Add))` |
| 18 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.6/m.0/cv3/conv/Conv` |
| 19 | CaskConvolution | `/model.6/m.0/cv3/conv/Conv` |
| 20 | PointWiseV2 | `PWN(PWN(/model.6/m.0/cv3/act/Sigmoid), PWN(/model.6/m.0/cv3/act/Mul))` |
| 21 | Reformat | `/model.6/Split_output_0 copy` |
| 22 | Reformat | `/model.6/Split_output_1 copy` |
| 23 | CaskGemmConvolution | `/model.6/cv2/conv/Conv` |
| 24 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.6/cv2/act/Sigmoid), PWN(/model.6/cv2/act/Mul))` |
| 25 | PointWiseV2 | `PWN(PWN(/model.6/cv2/act/Sigmoid), PWN(/model.6/cv2/act/Mul))` |

</details>

<details><summary><b>model.7</b> (Conv) — .pt leaf 3 → 커널 3개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.7/conv/Conv` |
| 1 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.7/act/Sigmoid), PWN(/model.7/act/Mul))` |
| 2 | PointWiseV2 | `PWN(PWN(/model.7/act/Sigmoid), PWN(/model.7/act/Mul))` |

</details>

<details><summary><b>model.8</b> (C3k2) — .pt leaf 19 → 커널 25개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskGemmConvolution | `/model.8/cv1/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.8/cv1/act/Sigmoid), PWN(/model.8/cv1/act/Mul))` |
| 2 | CaskGemmConvolution | `/model.8/m.0/cv2/conv/Conv \|\| /model.8/m.0/cv1/conv/Conv` |
| 3 | PointWiseV2 | `PWN(PWN(/model.8/m.0/cv1/act/Sigmoid), PWN(/model.8/m.0/cv1/act/Mul))` |
| 4 | CaskConvolution | `/model.8/m.0/m/m.0/cv1/conv/Conv` |
| 5 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.8/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.8/m.0/m/m.0/cv1/act/Mul))` |
| 6 | PointWiseV2 | `PWN(PWN(/model.8/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.8/m.0/m/m.0/cv1/act/Mul))` |
| 7 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.8/m.0/m/m.0/cv2/conv/Conv` |
| 8 | CaskConvolution | `/model.8/m.0/m/m.0/cv2/conv/Conv` |
| 9 | PointWiseV2 | `PWN(PWN(PWN(/model.8/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.8/m.0/m/m.0/cv2/act/Mul)), PWN(/model.8/m.0/m/m.0/Add))` |
| 10 | CaskConvolution | `/model.8/m.0/m/m.1/cv1/conv/Conv` |
| 11 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.8/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.8/m.0/m/m.1/cv1/act/Mul))` |
| 12 | PointWiseV2 | `PWN(PWN(/model.8/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.8/m.0/m/m.1/cv1/act/Mul))` |
| 13 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.8/m.0/m/m.1/cv2/conv/Conv` |
| 14 | CaskConvolution | `/model.8/m.0/m/m.1/cv2/conv/Conv` |
| 15 | PointWiseV2 | `PWN(PWN(/model.8/m.0/cv2/act/Sigmoid), PWN(/model.8/m.0/cv2/act/Mul))` |
| 16 | PointWiseV2 | `PWN(PWN(PWN(/model.8/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.8/m.0/m/m.1/cv2/act/Mul)), PWN(/model.8/m.0/m/m.1/Add))` |
| 17 | CaskGemmConvolution | `/model.8/m.0/cv3/conv/Conv` |
| 18 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.8/m.0/cv3/act/Sigmoid), PWN(/model.8/m.0/cv3/act/Mul))` |
| 19 | PointWiseV2 | `PWN(PWN(/model.8/m.0/cv3/act/Sigmoid), PWN(/model.8/m.0/cv3/act/Mul))` |
| 20 | Reformat | `/model.8/Split_output_0 copy` |
| 21 | Reformat | `/model.8/Split_output_1 copy` |
| 22 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.8/cv2/conv/Conv` |
| 23 | CaskGemmConvolution | `/model.8/cv2/conv/Conv` |
| 24 | PointWiseV2 | `PWN(PWN(/model.8/cv2/act/Sigmoid), PWN(/model.8/cv2/act/Mul))` |

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

<details><summary><b>model.10</b> (C2PSA) — .pt leaf 19 → 커널 30개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.10/cv1/conv/Conv` |
| 1 | CaskGemmConvolution | `/model.10/cv1/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.10/cv1/act/Sigmoid), PWN(/model.10/cv1/act/Mul))` |
| 3 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(/model.10/cv1/act/Sigmoid), PWN(/model.10/cv1/act/Mul))` |
| 4 | Reformat | `/model.10/Split_13` |
| 5 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.10/m/m.0/attn/qkv/conv/Conv` |
| 6 | CaskGemmConvolution | `/model.10/m/m.0/attn/qkv/conv/Conv` |
| 7 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.10/m/m.0/attn/Reshape` |
| 8 | NoOp | `/model.10/m/m.0/attn/Reshape` |
| 9 | Reformat | `/model.10/m/m.0/attn/Split_18` |
| 10 | NoOp | `/model.10/m/m.0/attn/Reshape_2` |
| 11 | CaskConvolution | `/model.10/m/m.0/attn/pe/conv/Conv` |
| 12 | kgen | `__myl_MovSliSliTra_myl109_1` |
| 13 | gemm | `/model_10/m/m_0/attn/MatMul_myl109_2` |
| 14 | kgen | `__myl_MulMaxSubExpSum_myl109_3` |
| 15 | kgen | `__myl_DivMulTra_myl109_4` |
| 16 | gemm | `/model_10/m/m_0/attn/MatMul_1_myl109_5` |
| 17 | kgen | `__myl_Add_myl109_6` |
| 18 | Reformat | `Reformatting CopyNode for Output Tensor 0 to {ForeignNode[/model.10/m/m.0/attn/Split_16.../model.10/m/m.0/attn/Add]}` |
| 19 | NoOp | `Reformatting CopyNode for Input Tensor 1 to /model.10/m/m.0/attn/proj/conv/Conv + /model.10/m/m.0/Add` |
| 20 | CaskGemmConvolution | `/model.10/m/m.0/attn/proj/conv/Conv + /model.10/m/m.0/Add` |
| 21 | CaskGemmConvolution | `/model.10/m/m.0/ffn/ffn.0/conv/Conv` |
| 22 | PointWiseV2 | `PWN(PWN(/model.10/m/m.0/ffn/ffn.0/act/Sigmoid), PWN(/model.10/m/m.0/ffn/ffn.0/act/Mul))` |
| 23 | CaskGemmConvolution | `/model.10/m/m.0/ffn/ffn.1/conv/Conv + /model.10/m/m.0/Add_1` |
| 24 | Reformat | `/model.10/Split_output_0 copy` |
| 25 | Reformat | `/model.10/m/m.0/Add_1_output_0 copy` |
| 26 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.10/cv2/conv/Conv` |
| 27 | CaskGemmConvolution | `/model.10/cv2/conv/Conv` |
| 28 | PointWiseV2 | `PWN(PWN(/model.10/cv2/act/Sigmoid), PWN(/model.10/cv2/act/Mul))` |
| 29 | Reformat | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(/model.10/cv2/act/Sigmoid), PWN(/model.10/cv2/act/Mul))` |

</details>

<details><summary><b>model.11</b> (Upsample) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Resize | `/model.11/Resize` |

</details>

<details><summary><b>model.12</b> (Concat) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `/model.11/Resize_output_0 copy` |

</details>

<details><summary><b>model.13</b> (C3k2) — .pt leaf 19 → 커널 26개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.13/cv1/conv/Conv` |
| 1 | CaskGemmConvolution | `/model.13/cv1/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.13/cv1/act/Sigmoid), PWN(/model.13/cv1/act/Mul))` |
| 3 | CaskConvolution | `/model.13/m.0/cv2/conv/Conv \|\| /model.13/m.0/cv1/conv/Conv` |
| 4 | Reformat | `Reformatting CopyNode for Output Tensor 0 to /model.13/m.0/cv2/conv/Conv \|\| /model.13/m.0/cv1/conv/Conv` |
| 5 | PointWiseV2 | `PWN(PWN(/model.13/m.0/cv1/act/Sigmoid), PWN(/model.13/m.0/cv1/act/Mul))` |
| 6 | CaskConvolution | `/model.13/m.0/m/m.0/cv1/conv/Conv` |
| 7 | PointWiseV2 | `PWN(PWN(/model.13/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.13/m.0/m/m.0/cv1/act/Mul))` |
| 8 | CaskConvolution | `/model.13/m.0/m/m.0/cv2/conv/Conv` |
| 9 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.13/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.13/m.0/m/m.0/cv2/act/Mul)), PWN(/model.13/m.0/m/m.0/Add))` |
| 10 | NoOp | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.13/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.13/m.0/m/m.0/cv2/act/Mul)), PWN(/model.13/m.0/m/m.0/Add))` |
| 11 | PointWiseV2 | `PWN(PWN(PWN(/model.13/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.13/m.0/m/m.0/cv2/act/Mul)), PWN(/model.13/m.0/m/m.0/Add))` |
| 12 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.13/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.13/m.0/m/m.0/cv2/act/Mul)), PWN(/model.13/m.0/m/m.0/Add))` |
| 13 | CaskConvolution | `/model.13/m.0/m/m.1/cv1/conv/Conv` |
| 14 | PointWiseV2 | `PWN(PWN(/model.13/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.13/m.0/m/m.1/cv1/act/Mul))` |
| 15 | CaskConvolution | `/model.13/m.0/m/m.1/cv2/conv/Conv` |
| 16 | PointWiseV2 | `PWN(PWN(/model.13/m.0/cv2/act/Sigmoid), PWN(/model.13/m.0/cv2/act/Mul))` |
| 17 | PointWiseV2 | `PWN(PWN(PWN(/model.13/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.13/m.0/m/m.1/cv2/act/Mul)), PWN(/model.13/m.0/m/m.1/Add))` |
| 18 | CaskConvolution | `/model.13/m.0/cv3/conv/Conv` |
| 19 | PointWiseV2 | `PWN(PWN(/model.13/m.0/cv3/act/Sigmoid), PWN(/model.13/m.0/cv3/act/Mul))` |
| 20 | Reformat | `/model.13/Split_output_0 copy` |
| 21 | Reformat | `/model.13/Split_output_1 copy` |
| 22 | CaskConvolution | `/model.13/cv2/conv/Conv` |
| 23 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.13/cv2/act/Sigmoid), PWN(/model.13/cv2/act/Mul))` |
| 24 | PointWiseV2 | `PWN(PWN(/model.13/cv2/act/Sigmoid), PWN(/model.13/cv2/act/Mul))` |
| 25 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(/model.13/cv2/act/Sigmoid), PWN(/model.13/cv2/act/Mul))` |

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

<details><summary><b>model.16</b> (C3k2) — .pt leaf 19 → 커널 24개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskGemmConvolution | `/model.16/cv1/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.16/cv1/act/Sigmoid), PWN(/model.16/cv1/act/Mul))` |
| 2 | CaskConvolution | `/model.16/m.0/cv2/conv/Conv \|\| /model.16/m.0/cv1/conv/Conv` |
| 3 | PointWiseV2 | `PWN(PWN(/model.16/m.0/cv1/act/Sigmoid), PWN(/model.16/m.0/cv1/act/Mul))` |
| 4 | CaskConvolution | `/model.16/m.0/m/m.0/cv1/conv/Conv` |
| 5 | PointWiseV2 | `PWN(PWN(/model.16/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.16/m.0/m/m.0/cv1/act/Mul))` |
| 6 | CaskConvolution | `/model.16/m.0/m/m.0/cv2/conv/Conv` |
| 7 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.16/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.16/m.0/m/m.0/cv2/act/Mul)), PWN(/model.16/m.0/m/m.0/Add))` |
| 8 | NoOp | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.16/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.16/m.0/m/m.0/cv2/act/Mul)), PWN(/model.16/m.0/m/m.0/Add))` |
| 9 | PointWiseV2 | `PWN(PWN(PWN(/model.16/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.16/m.0/m/m.0/cv2/act/Mul)), PWN(/model.16/m.0/m/m.0/Add))` |
| 10 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.16/m.0/m/m.1/cv1/conv/Conv` |
| 11 | CaskConvolution | `/model.16/m.0/m/m.1/cv1/conv/Conv` |
| 12 | PointWiseV2 | `PWN(PWN(/model.16/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.16/m.0/m/m.1/cv1/act/Mul))` |
| 13 | CaskConvolution | `/model.16/m.0/m/m.1/cv2/conv/Conv` |
| 14 | PointWiseV2 | `PWN(PWN(/model.16/m.0/cv2/act/Sigmoid), PWN(/model.16/m.0/cv2/act/Mul))` |
| 15 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.16/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.16/m.0/m/m.1/cv2/act/Mul)), PWN(/model.16/m.0/m/m.1/Add))` |
| 16 | PointWiseV2 | `PWN(PWN(PWN(/model.16/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.16/m.0/m/m.1/cv2/act/Mul)), PWN(/model.16/m.0/m/m.1/Add))` |
| 17 | Reformat | `/model.16/m.0/m/m.1/Add_output_0 copy` |
| 18 | CaskConvolution | `/model.16/m.0/cv3/conv/Conv` |
| 19 | PointWiseV2 | `PWN(PWN(/model.16/m.0/cv3/act/Sigmoid), PWN(/model.16/m.0/cv3/act/Mul))` |
| 20 | Reformat | `/model.16/Split_output_0 copy` |
| 21 | Reformat | `/model.16/Split_output_1 copy` |
| 22 | CaskGemmConvolution | `/model.16/cv2/conv/Conv` |
| 23 | PointWiseV2 | `PWN(PWN(/model.16/cv2/act/Sigmoid), PWN(/model.16/cv2/act/Mul))` |

</details>

<details><summary><b>model.17</b> (Conv) — .pt leaf 3 → 커널 2개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.17/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.17/act/Sigmoid), PWN(/model.17/act/Mul))` |

</details>

<details><summary><b>model.18</b> (Concat) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `/model.13/cv2/act/Mul_output_0 copy` |

</details>

<details><summary><b>model.19</b> (C3k2) — .pt leaf 19 → 커널 25개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.19/cv1/conv/Conv` |
| 1 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.19/cv1/act/Sigmoid), PWN(/model.19/cv1/act/Mul))` |
| 2 | PointWiseV2 | `PWN(PWN(/model.19/cv1/act/Sigmoid), PWN(/model.19/cv1/act/Mul))` |
| 3 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(/model.19/cv1/act/Sigmoid), PWN(/model.19/cv1/act/Mul))` |
| 4 | CaskConvolution | `/model.19/m.0/cv2/conv/Conv \|\| /model.19/m.0/cv1/conv/Conv` |
| 5 | PointWiseV2 | `PWN(PWN(/model.19/m.0/cv1/act/Sigmoid), PWN(/model.19/m.0/cv1/act/Mul))` |
| 6 | CaskConvolution | `/model.19/m.0/m/m.0/cv1/conv/Conv` |
| 7 | PointWiseV2 | `PWN(PWN(/model.19/m.0/m/m.0/cv1/act/Sigmoid), PWN(/model.19/m.0/m/m.0/cv1/act/Mul))` |
| 8 | CaskConvolution | `/model.19/m.0/m/m.0/cv2/conv/Conv` |
| 9 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.19/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.19/m.0/m/m.0/cv2/act/Mul)), PWN(/model.19/m.0/m/m.0/Add))` |
| 10 | NoOp | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.19/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.19/m.0/m/m.0/cv2/act/Mul)), PWN(/model.19/m.0/m/m.0/Add))` |
| 11 | PointWiseV2 | `PWN(PWN(PWN(/model.19/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.19/m.0/m/m.0/cv2/act/Mul)), PWN(/model.19/m.0/m/m.0/Add))` |
| 12 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.19/m.0/m/m.0/cv2/act/Sigmoid), PWN(/model.19/m.0/m/m.0/cv2/act/Mul)), PWN(/model.19/m.0/m/m.0/Add))` |
| 13 | CaskConvolution | `/model.19/m.0/m/m.1/cv1/conv/Conv` |
| 14 | PointWiseV2 | `PWN(PWN(/model.19/m.0/m/m.1/cv1/act/Sigmoid), PWN(/model.19/m.0/m/m.1/cv1/act/Mul))` |
| 15 | CaskConvolution | `/model.19/m.0/m/m.1/cv2/conv/Conv` |
| 16 | PointWiseV2 | `PWN(PWN(/model.19/m.0/cv2/act/Sigmoid), PWN(/model.19/m.0/cv2/act/Mul))` |
| 17 | PointWiseV2 | `PWN(PWN(PWN(/model.19/m.0/m/m.1/cv2/act/Sigmoid), PWN(/model.19/m.0/m/m.1/cv2/act/Mul)), PWN(/model.19/m.0/m/m.1/Add))` |
| 18 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.19/m.0/cv3/conv/Conv` |
| 19 | CaskConvolution | `/model.19/m.0/cv3/conv/Conv` |
| 20 | PointWiseV2 | `PWN(PWN(/model.19/m.0/cv3/act/Sigmoid), PWN(/model.19/m.0/cv3/act/Mul))` |
| 21 | Reformat | `/model.19/Split_output_0 copy` |
| 22 | Reformat | `/model.19/Split_output_1 copy` |
| 23 | CaskGemmConvolution | `/model.19/cv2/conv/Conv` |
| 24 | PointWiseV2 | `PWN(PWN(/model.19/cv2/act/Sigmoid), PWN(/model.19/cv2/act/Mul))` |

</details>

<details><summary><b>model.20</b> (Conv) — .pt leaf 3 → 커널 2개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.20/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.20/act/Sigmoid), PWN(/model.20/act/Mul))` |

</details>

<details><summary><b>model.21</b> (Concat) — .pt leaf 1 → 커널 1개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | Reformat | `/model.10/cv2/act/Mul_output_0 copy` |

</details>

<details><summary><b>model.22</b> (C3k2) — .pt leaf 23 → 커널 35개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskGemmConvolution | `/model.22/cv1/conv/Conv` |
| 1 | PointWiseV2 | `PWN(PWN(/model.22/cv1/act/Sigmoid), PWN(/model.22/cv1/act/Mul))` |
| 2 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(/model.22/cv1/act/Sigmoid), PWN(/model.22/cv1/act/Mul))` |
| 3 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.22/m.0/m.0.0/cv1/conv/Conv` |
| 4 | CaskConvolution | `/model.22/m.0/m.0.0/cv1/conv/Conv` |
| 5 | Reformat | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.22/m.0/m.0.0/cv1/act/Sigmoid), PWN(/model.22/m.0/m.0.0/cv1/act/Mul))` |
| 6 | PointWiseV2 | `PWN(PWN(/model.22/m.0/m.0.0/cv1/act/Sigmoid), PWN(/model.22/m.0/m.0.0/cv1/act/Mul))` |
| 7 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.22/m.0/m.0.0/cv2/conv/Conv` |
| 8 | CaskConvolution | `/model.22/m.0/m.0.0/cv2/conv/Conv` |
| 9 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.22/m.0/m.0.0/cv2/act/Sigmoid), PWN(/model.22/m.0/m.0.0/cv2/act/Mul)), PWN(/model.22/m.0/m.0.0/Add))` |
| 10 | PointWiseV2 | `PWN(PWN(PWN(/model.22/m.0/m.0.0/cv2/act/Sigmoid), PWN(/model.22/m.0/m.0.0/cv2/act/Mul)), PWN(/model.22/m.0/m.0.0/Add))` |
| 11 | NoOp | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.22/m.0/m.0.0/cv2/act/Sigmoid), PWN(/model.22/m.0/m.0.0/cv2/act/Mul)), PWN(/model.22/m.0/m.0.0/Add))` |
| 12 | CaskGemmConvolution | `/model.22/m.0/m.0.1/attn/qkv/conv/Conv` |
| 13 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.22/m.0/m.0.1/attn/Reshape` |
| 14 | NoOp | `/model.22/m.0/m.0.1/attn/Reshape` |
| 15 | Reformat | `/model.22/m.0/m.0.1/attn/Split_40` |
| 16 | NoOp | `/model.22/m.0/m.0.1/attn/Reshape_2` |
| 17 | CaskConvolution | `/model.22/m.0/m.0.1/attn/pe/conv/Conv` |
| 18 | kgen | `__myl_MovSliSliTra_myl225_1` |
| 19 | gemm | `/model_22/m_0/m_0_1/attn/MatMul_myl225_2` |
| 20 | kgen | `__myl_MulMaxSubExpSum_myl225_3` |
| 21 | kgen | `__myl_DivMulTra_myl225_4` |
| 22 | gemm | `/model_22/m_0/m_0_1/attn/MatMul_1_myl225_5` |
| 23 | kgen | `__myl_Add_myl225_6` |
| 24 | Reformat | `Reformatting CopyNode for Output Tensor 0 to {ForeignNode[/model.22/m.0/m.0.1/attn/Split_38.../model.22/m.0/m.0.1/attn/Add]}` |
| 25 | CaskGemmConvolution | `/model.22/m.0/m.0.1/attn/proj/conv/Conv + /model.22/m.0/m.0.1/Add` |
| 26 | CaskGemmConvolution | `/model.22/m.0/m.0.1/ffn/ffn.0/conv/Conv` |
| 27 | PointWiseV2 | `PWN(PWN(/model.22/m.0/m.0.1/ffn/ffn.0/act/Sigmoid), PWN(/model.22/m.0/m.0.1/ffn/ffn.0/act/Mul))` |
| 28 | CaskGemmConvolution | `/model.22/m.0/m.0.1/ffn/ffn.1/conv/Conv + /model.22/m.0/m.0.1/Add_1` |
| 29 | Reformat | `/model.22/Split_output_0 copy` |
| 30 | Reformat | `/model.22/Split_output_1 copy` |
| 31 | Reformat | `/model.22/m.0/m.0.1/Add_1_output_0 copy` |
| 32 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.22/cv2/conv/Conv` |
| 33 | CaskGemmConvolution | `/model.22/cv2/conv/Conv` |
| 34 | PointWiseV2 | `PWN(PWN(/model.22/cv2/act/Sigmoid), PWN(/model.22/cv2/act/Mul))` |

</details>

<details><summary><b>model.23</b> (Detect) — .pt leaf 88 → 커널 66개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv` |
| 1 | CaskConvolution | `/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv` |
| 2 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/act/Mul))` |
| 3 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv2.0/one2one_cv2.0.0/act/Sigmoid), PWN(/model.23/one2one_cv2.0/one2one_cv2.0.0/act/Mul))` |
| 4 | CaskConvolution | `/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv` |
| 5 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv` |
| 6 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv2.0/one2one_cv2.0.1/act/Sigmoid), PWN(/model.23/one2one_cv2.0/one2one_cv2.0.1/act/Mul))` |
| 7 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/act/Mul))` |
| 8 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv` |
| 9 | CaskConvolution | `/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv` |
| 10 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/act/Mul))` |
| 11 | CaskConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv` |
| 12 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul))` |
| 13 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv3.0/one2one_cv3.0.2/Conv` |
| 14 | CaskGemmConvolution | `/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv` |
| 15 | CaskConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv` |
| 16 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv` |
| 17 | CaskConvolution | `/model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv` |
| 18 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/act/Mul))` |
| 19 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv2.1/one2one_cv2.1.0/act/Sigmoid), PWN(/model.23/one2one_cv2.1/one2one_cv2.1.0/act/Mul))` |
| 20 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv2.1/one2one_cv2.1.0/act/Sigmoid), PWN(/model.23/one2one_cv2.1/one2one_cv2.1.0/act/Mul))` |
| 21 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv` |
| 22 | CaskConvolution | `/model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv` |
| 23 | CaskGemmConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv` |
| 24 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv2.1/one2one_cv2.1.1/act/Sigmoid), PWN(/model.23/one2one_cv2.1/one2one_cv2.1.1/act/Mul))` |
| 25 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv2.1/one2one_cv2.1.1/act/Sigmoid), PWN(/model.23/one2one_cv2.1/one2one_cv2.1.1/act/Mul))` |
| 26 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/act/Mul))` |
| 27 | CaskConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv` |
| 28 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv2.1/one2one_cv2.1.2/Conv` |
| 29 | CaskConvolution | `/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv` |
| 30 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/act/Mul))` |
| 31 | CaskConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv` |
| 32 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul))` |
| 33 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv3.1/one2one_cv3.1.2/Conv` |
| 34 | CaskGemmConvolution | `/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv` |
| 35 | CaskConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv` |
| 36 | Reformat | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv` |
| 37 | CaskConvolution | `/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv` |
| 38 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/act/Mul))` |
| 39 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv2.2/one2one_cv2.2.0/act/Sigmoid), PWN(/model.23/one2one_cv2.2/one2one_cv2.2.0/act/Mul))` |
| 40 | CaskConvolution | `/model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv` |
| 41 | CaskGemmConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv` |
| 42 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv2.2/one2one_cv2.2.1/act/Sigmoid), PWN(/model.23/one2one_cv2.2/one2one_cv2.2.1/act/Mul))` |
| 43 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/act/Mul))` |
| 44 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/act/Mul))` |
| 45 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv` |
| 46 | CaskConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv` |
| 47 | CaskConvolution | `/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv` |
| 48 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Mul))` |
| 49 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Mul))` |
| 50 | NoOp | `Reformatting CopyNode for Input Tensor 0 to /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv` |
| 51 | CaskConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv` |
| 52 | NoOp | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul))` |
| 53 | PointWiseV2 | `PWN(PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Sigmoid), PWN(/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul))` |
| 54 | CaskGemmConvolution | `/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv` |
| 55 | NoOp | `Reformatting CopyNode for Input Tensor 0 to {ForeignNode[/model.23/Gather_1_output_0[Constant] + ONNXTRT_Broadcast_141.../model.23/Concat_6]}` |
| 56 | NoOp | `Reformatting CopyNode for Input Tensor 1 to {ForeignNode[/model.23/Gather_1_output_0[Constant] + ONNXTRT_Broadcast_141.../model.23/Concat_6]}` |
| 57 | NoOp | `Reformatting CopyNode for Input Tensor 2 to {ForeignNode[/model.23/Gather_1_output_0[Constant] + ONNXTRT_Broadcast_141.../model.23/Concat_6]}` |
| 58 | Reformat | `Reformatting CopyNode for Input Tensor 3 to {ForeignNode[/model.23/Gather_1_output_0[Constant] + ONNXTRT_Broadcast_141.../model.23/Concat_6]}` |
| 59 | kgen | `__myl_ConNegExpAddDivConSliSubSliAddConMulConTraSliMaxSli_myl296_1` |
| 60 | kgen | `__myl_Top_myl296_2` |
| 61 | kgen | `__myl_CasResCasGatRes_myl296_3` |
| 62 | kgen | `__myl_Top_myl296_4` |
| 63 | kgen | `__myl_Mov_myl296_5` |
| 64 | kgen | `__myl_CasCasDivFloCasSubResCasMov_myl296_6` |
| 65 | kgen | `__myl_ResDivGatCasRepGatMov_myl296_7` |

</details>

<details><summary><b>post-process</b> — 커널 3개</summary>

| # | 타입 | 커널 이름 |
|--:|---|---|
| 0 | shape_call | `dummy_shape_call__mye4273_0_myl109_0` |
| 1 | shape_call | `dummy_shape_call__mye4273_0_myl225_0` |
| 2 | shape_call | `dummy_shape_call__mye23516_0_myl296_0` |

</details>

## 읽는 법

- **`레이어 수 (.pt→TRT)`** — 왼쪽은 `.pt` 모듈 안의 leaf 서브모듈 수(정확한 연산 수 아님, 어림값), 오른쪽은 그 단계로 매핑된 TRT 커널 수. `--onnx model/best.onnx` 를 주면 가운데에 ONNX 노드 수도 표시된다. ONNX 없이도 `.pt→TRT` 는 그대로 나온다.
- **커널 타입** 은 LayerType 계열명: `conv`(CaskConvolution) / `gemm`(MatMul·FullyConnected) / `pointwise`(활성화·elementwise) / `pool` / `reformat`·`noop`(레이아웃 변환) / `resize` / `post`(slice·concat·topk·nms).
- `.pt` 의 `Conv` = Conv2d+BN+SiLU. TRT 에서 BN 은 conv 가중치에 접히고(fold), SiLU(Sigmoid+Mul)는 `PointWiseV2` 1개로 fusion 된다.
- `Concat`(model.12/15/18/21) 은 대부분 zero-copy 로 처리돼 독립 커널이 거의 없다 → `1 → 1` (reformat 1개).
- `Detect`(model.23) 는 학습용 구조가 사라지고 추론 경로 + DFL + box decode + TopK/NMS 가 그래프로 펼쳐져 커널 수가 가장 많다.
