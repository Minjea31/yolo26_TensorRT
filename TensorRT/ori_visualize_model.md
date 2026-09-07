# TensorRT/ori_visualize_model.py

**[원본 모델 그래프]** `model/best.pt` (Ultralytics YOLO 계열) 체크포인트의
**레이어 구조를 세로 방향 그래프로** 그려서 이미지(PNG/SVG/PDF)로 저장한다.
(`.pt` → ONNX → TensorRT 엔진 파이프라인의 사전 모델 검사용 → `TensorRT/` 에서 관리)

TensorRT 적용 **후** 구조는 → [`TRT_visualize_model.md`](TRT_visualize_model.md)

## 무엇을 하나

1. `best.pt` 를 로드해 실제 `nn.Module` 을 꺼낸다.
2. `net.model`(레이어들의 `nn.Sequential`)을 순회하며 레이어별로
   인덱스 · 입력 연결(`from`) · 모듈 타입 · 파라미터 수를 수집한다.
3. (옵션) 더미 입력을 한 번 흘려 **forward hook** 으로 레이어별 출력 텐서 shape 를 캡처한다.
4. Graphviz **DOT** 문자열을 만들어 이미지로 렌더한다.
5. 콘솔에 레이어 요약표를 출력한다.

## 요구 사항

| 항목 | 비고 |
|---|---|
| Python 환경 | **torch 2.x + ultralytics 포크** 가 설치된 conda 환경. 예: `conda activate yolo` |
| ultralytics 포크 | `pip install -e yolo26/` 로 설치돼 있으면 됨 (이미 `yolo` env 에 설치 완료). 미설치여도 스크립트가 repo 안 `yolo26/` 를 자동으로 찾아 `sys.path` 에 넣음 |
| Graphviz | 시스템 `dot` 실행파일 (`sudo apt-get install graphviz`) + python `graphviz` 패키지. python 패키지가 있으면 그것을 우선 사용, 없으면 `dot` 직접 호출 |

전체 목록은 `yolo26/requirements.txt` 참고. 요약:

```bash
conda activate yolo
cd yolo26 && pip install -r requirements.txt   # 포크(-e .) 포함
sudo apt-get install -y graphviz
```

> ⚠️ 기본 `python`(anaconda base)은 torch 1.12 + torchvision 없음 → 로드 실패. 반드시 `yolo` 등 torch 2.x 환경에서 실행할 것.
> ✅ 실행 위치 무관 — `find_repo_root()` 로 repo 루트를 찾아 `os.chdir()` 한 뒤 동작한다.
> 모든 경로(`--weights` / `--out` 기본값, 로그 출력)는 **repo 루트 기준 상대경로** (`model/best.pt`, `model/ori_structure.png` …). 절대경로 안 씀.

## 사용법

```bash
conda activate yolo

# repo 루트에서
python TensorRT/ori_visualize_model.py

# 또는 TensorRT/ 안에서
cd TensorRT && python ori_visualize_model.py

# 옵션 예시
python TensorRT/ori_visualize_model.py --weights model/prune.pt --out model/ori_prune.png
python TensorRT/ori_visualize_model.py --rankdir LR --no-shapes
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `model/best.pt` | 가중치 파일 경로 (`.pt`) |
| `--out` | `model/ori_structure.png` | 출력 이미지 경로. 확장자(`.png`/`.svg`/`.pdf`)로 포맷 결정 |
| `--imgsz` | `640` | shape 추론용 더미 입력 크기 |
| `--no-shapes` | (꺼짐) | forward 패스를 건너뛰고 shape 표기 생략 (더 빠름) |
| `--rankdir` | `TB` | 그래프 방향. `TB`=위→아래, `LR`=왼→오른 |
| `--dpi` | `150` | PNG 해상도 |

## 출력물

- `<out>` — 구조 그래프 이미지
- `<out>.dot` — Graphviz DOT 소스(텍스트). `dot -Tsvg ori_structure.dot -o x.svg` 로 재렌더 가능
- stdout — 레이어별 요약표 (`idx / type / from / params / shape`) + 전체 파라미터 합계

## 그래프 읽는 법

| 색 | 의미 |
|---|---|
| 🟦 파랑 `#d0ebff` | backbone (`i < len(yaml['backbone'])`) |
| 🟩 연두 `#d3f9d8` | neck / head |
| 🟪 라벤더 `#e5dbff` | `Concat` / `Upsample` |
| 🟧 주황 `#ffd8a8` | 마지막 `Detect` head |
| 주황 굵은 화살표 | 입력이 여러 개인 연결 (skip / concat) |

노드 라벨: `인덱스  타입` / `params <수>` / `<출력 shape>`

## 동작 원리 (함수별)

| 함수 | 역할 |
|---|---|
| `find_repo_root()` | 스크립트 위치·현재 폴더에서 위로 올라가며 `yolo26/ultralytics/__init__.py` 를 가진 폴더(repo 루트) 탐색. `main()` 이 그 경로로 `os.chdir()` → 이후 모든 상대경로가 repo 루트 기준 |
| `add_local_fork_to_path()` | repo 안 커스텀 `yolo26/ultralytics` 를 우선 import 하도록 `sys.path` 조정 (pickle 클래스 정합성). 포크가 `pip install -e` 로 설치돼 있으면 없어도 됨 |
| `load_model()` | `YOLO(weights).model` → `nn.Module`, `.eval().float()` |
| `collect_layers()` | ultralytics 가 레이어에 붙여둔 `.i` `.f` `.type` `.np` 수집 + (옵션) forward hook 으로 shape 캡처. hook 은 `finally` 에서 `remove()` |
| `human()` / `shape_str()` | 숫자·shape 를 보기 좋은 문자열로 |
| `build_dot()` | 노드/엣지 + 색상으로 DOT 문자열 생성. `f == -1` → `i-1`, `f` 가 리스트 → skip 강조 |
| `render()` | ① python `graphviz` → ② 시스템 `dot` → ③ 실패 시 `.dot` 만 저장 |
| `print_table()` | 콘솔 요약표 |

## 예시 결과 (현재 `best.pt`)

- YOLO11n 계열, `task=detect`, 클래스 1개 `['CAR']`, **24 레이어**, 약 **2.5M 파라미터**
- backbone(0–10): `Conv → C3k2 → SPPF → C2PSA`
- head(11–23): PAN-FPN (`Upsample`/`Concat`/`C3k2`) + `Detect`([16, 19, 22] 입력, end2end 출력 `1x300x6`)
