# TensorRT/analysis.py

**[TensorRT 레이어 변환 분석 → Markdown]**
`best.engine.layers.json` (+ `best.onnx`) 을 분석해 **원래 레이어가 어떻게 합쳐지고(fusion)
어떤 이름으로 바뀌었는지** 를 `.md` 문서로 저장한다.

분석 로직은 [`TRT_layer_print.py`](TRT_layer_print.md) 의 `analyze()` 를 그대로 import 해서 쓴다
(터미널 출력 = `TRT_layer_print.py`, 문서 정리 = 이 스크립트).

## 입력 / 출력

| | |
|---|---|
| 입력 | `model/best.engine.layers.json`, `model/best.onnx` (선택) |
| 출력 | `model/TRT_layer_analysis.md` (`--out` 으로 변경) |

## 생성되는 .md 구성

| 섹션 | 내용 |
|---|---|
| 요약 | keep / rename / fuse / new-io / new-int / (흡수·제거) 개수 표 |
| 1. Fusion | 원본 `model.N` 단계별로, 합쳐진 원본 op 들(`A + B + …`) → TRT 커널 이름·타입 |
| 2. rename | 원본 ONNX 이름 → 바뀐 TRT 커널 이름 (1:1) |
| 3. TRT 신규 삽입 | `Reformat`/`NoOp`(new-io), 내부 커널(new-int). `<details>` 접힘 |
| 4. 사라진 노드 | `best.onnx` 에 있었지만 엔진에 없는 것 (상수폴딩/흡수). `<details>` 접힘 |
| 5. keep | 이름 유지된 레이어 목록. `<details>` 접힘 |

## 사용법

```bash
conda activate yolo
python TensorRT/analysis.py
python TensorRT/analysis.py --engine-json model/best.engine.layers.json \
    --onnx model/best.onnx --out model/TRT_layer_analysis.md
```

## 동작 원리

| 함수 | 역할 |
|---|---|
| `from TRT_layer_print import analyze, summary_counts, norm` | 분석 로직 재사용 (중복 구현 안 함) |
| `code()` | 마크다운 표 셀용 — `\|` 이스케이프 + 백틱 |
| `to_markdown(res)` | `analyze()` 결과 dict → 위 6개 섹션 Markdown 문자열 |
| `main()` | `os.chdir(repo 루트)` → 분석 → `--out` 에 파일 저장 |

## 참고

- Fusion 은 대부분 **SiLU 활성화**(`Sigmoid`+`Mul`, residual 있으면 +`Add`) → `PointWiseV2` 1개,
  그리고 일부 **conv 쌍**(`cv1`+`cv2`) → `CaskConvolution` 1개.
- `rename` 의 상당수는 attention 블록의 `MatMul`/`Add`/`Split` 이 myelin(`_mylNNN`) 접미사를 달고
  이름이 바뀐 것, 그리고 `Concat` 을 위한 `Reformat` copy.
- `/model.23/*` (end2end decode/NMS) 는 대부분 상수폴딩되어 섹션 4 로 빠진다.
