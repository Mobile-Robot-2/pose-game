# pose-game 서버 실행 방법

이 프로젝트는 Temi 카메라 영상을 MediaPipe로 분석하는 FastAPI 서버입니다. 현재 `eunji` 브랜치에서는 낙상 감지와 리듬게임 동작을 확인하는 **테스트 서버**를 별도로 실행할 수 있습니다.

## 1. VS Code 터미널 열기

VS Code에서 **파일 → 폴더 열기**로 각자의 컴퓨터에 저장된 `pose-game` 폴더를 연 뒤 **터미널 → 새 터미널**을 선택합니다. 터미널 오른쪽 위의 `+` 옆 화살표에서 **Command Prompt(명령 프롬프트)**를 선택하세요. 아래 명령은 Windows 명령 프롬프트 기준입니다.

터미널이 `pose-game` 폴더에서 열렸는지 확인하고 Python 버전을 확인합니다.

```cmd
cd
python --version
```

`cd` 결과가 `pose-game` 폴더가 아니라면, 각자 프로젝트를 저장한 경로로 이동하세요. 경로에 공백이 있어도 따옴표로 감싸면 됩니다.

```cmd
cd /d "각자의 pose-game 폴더 경로"
```

Python 버전이 표시되지 않으면 VS Code에서 사용할 Python 환경을 먼저 선택하거나 Python을 설치해야 합니다. 기존에 사용하던 가상환경이 있다면 아래 새 환경 생성은 건너뛰고 그 환경을 활성화하세요.

## 2. VS Code에서 가상환경과 패키지 준비

VS Code에서 연 **Command Prompt** 터미널의 현재 위치가 `pose-game` 폴더인지 확인합니다. 처음 한 번, 같은 터미널에서 프로젝트 폴더에 `.venv` 가상환경을 만듭니다.

```cmd
python -m venv .venv
```

Command Prompt 터미널을 새로 열 때마다 다음 명령으로 가상환경을 활성화합니다.

```cmd
.venv\Scripts\activate.bat
```

성공하면 터미널 입력줄 앞에 보통 `(.venv)`가 표시됩니다. 위 명령은 Command Prompt용입니다. PowerShell을 사용할 경우 다음 명령으로 활성화해야 합니다.

```powershell
& .\.venv\Scripts\Activate.ps1
```

PowerShell에서 스크립트 실행이 차단되면 Command Prompt를 사용하거나, 현재 PowerShell 터미널에만 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`를 실행한 뒤 활성화 명령을 다시 입력하세요.

처음 한 번 또는 `requirements.txt`가 변경되었을 때 패키지를 설치합니다.

```powershell
python -m pip install -r requirements.txt
```

`models/pose_landmarker_lite.task` 파일이 있어야 영상 분석이 가능합니다.

## 3. 테스트 서버 실행

가상환경을 활성화한 VS Code 터미널에서 다음 명령을 실행합니다.

```powershell
python -m uvicorn test_server:app --host 0.0.0.0 --port 8000
```

서버가 실행되면 노트북 브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)를 열어 테스트 API를 확인할 수 있습니다. 서버를 종료하려면 실행 중인 터미널에서 `Ctrl+C`를 누릅니다.

테스트 서버는 Firebase에 결과를 저장하거나 보호자에게 연락하지 않습니다. 한 번에 한 기기로 테스트하는 용도이며, 화면을 나갔다 다시 열면 새 세션이 시작됩니다. 기존 서버가 8000번 포트를 사용하고 있다면 먼저 종료하세요.

## 4. Temi 앱 연결

1. 노트북과 Temi가 서로 통신할 수 있는 네트워크에 연결되어 있는지 확인합니다.
2. 노트북의 새 터미널에서 `ipconfig`를 실행하고 현재 Wi-Fi 연결의 **IPv4 주소**를 확인합니다.
3. Android 프로젝트의 `app/src/main/java/com/temi/rhythmgame/ServerConfig.java`에서 `BASE_URL`을 `http://노트북의_IPv4_주소:8000`으로 설정합니다. `127.0.0.1`은 Temi에서 노트북을 가리키지 않습니다.
4. Android 앱을 다시 빌드해 Temi에 설치하고, **낙상 감지 테스트** 또는 **리듬게임 테스트** 버튼을 누릅니다.

연결되지 않으면 서버 실행 상태, IP 주소, 포트와 Windows 방화벽을 확인합니다. 같은 Wi-Fi에 연결되어도 네트워크 설정에 따라 기기 간 통신이 차단될 수 있습니다.

## 기존 서비스 서버

기존 게임·순찰 기능을 사용할 때는 테스트 서버 대신 아래 명령을 실행합니다.

```powershell
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

기존 서버는 실행 시 프로젝트 루트의 `firebase-key.json`을 읽으며 Firebase에 접근합니다. 이 파일은 비밀 키이므로 공유하거나 Git에 올리지 마세요. 기존 서버와 테스트 서버는 같은 포트에서 동시에 실행할 수 없습니다.

테스트 화면의 동작과 제한은 [TESTING.md](TESTING.md)를 참고하세요.

## Temi 낙상 감지 사진과 로그 저장

Temi 카메라가 서버로 전송한 낙상 분석 프레임을 사진과 CSV로 저장하려면, 가상환경이 활성화된 **PowerShell**에서 `pose-game` 폴더로 이동한 뒤 다음 명령을 실행하세요.

```powershell
$env:FALL_DEBUG = "frames"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Temi 앱이 이 서버의 `/analyze` 또는 `/analyze_frame`으로 카메라 프레임을 보내면 `data/fall_debug/<실행별 폴더>/`에 다음 파일이 저장됩니다. 위 서비스 서버를 사용하려면 `firebase-key.json`이 필요합니다.

- `frame-000001.jpg` 등 — 왼쪽에 서버가 받은 원본 프레임, 오른쪽에 관절과 신체 중심을 표시한 비교 사진이 저장됩니다. 아래쪽에는 낙상 상태와 판정 수치가 표시됩니다.
- `frames.csv` — 프레임별 관절 좌표, 하강량, 기울기, 낙상 판정과 관측 초기화 사유가 기록됩니다.

낙상 감지 테스트 화면의 `/test/fall` 요청을 기록하려면 두 번째 명령을 `python -m uvicorn test_server:app --host 0.0.0.0 --port 8000`으로 바꾸세요. 테스트 서버는 Firebase 키 없이 사용할 수 있습니다.

사진은 낙상 확정 여부와 관계없이 수신·분석한 프레임에서 저장되며, 저장 폴더는 서버를 실행할 때마다 새로 만들어집니다. 사진 없이 CSV만 저장하려면 `$env:FALL_DEBUG = "csv"`로 설정한 뒤 서버를 실행하세요. `data/`는 Git에서 제외됩니다.

## 웹 실험 관리 대시보드

가상환경이 활성화된 PowerShell에서 다음 명령으로 테스트 서버를 실행하고, 노트북 브라우저에서 [http://127.0.0.1:8000/experiment](http://127.0.0.1:8000/experiment)에 접속하세요.

```powershell
$env:FALL_DEBUG = "frames"
python -m uvicorn test_server:app --host 0.0.0.0 --port 8000
```

Temi에서는 기존 **낙상 감지 테스트** 화면을 열어 두고, 노트북에서 아래 순서로 진행합니다. 한 서버 프로세스와 한 Temi로 실험하세요.

1. 참가자 ID, 낙상/비낙상, 동작, 반복 세트와 관측 시간을 설정하고 **실험 시작**을 누릅니다. 기본값은 전방 → 좌측 → 우측 → 후방을 5세트, 총 20시행이며 관측 시간은 시행당 10초입니다. 비낙상은 앉기, 의도적으로 눕기, 물건 줍기, 쪼그려 앉기를 선택할 수 있습니다.
2. 참가자가 직립 자세로 준비되면 **시행 시작**을 누릅니다. 이때 기존 FSM 상태와 이력을 초기화합니다. 준비 중 수신된 프레임은 시행 평가에 넣지 않습니다.
3. 동작을 수행합니다. 낙상이 확정돼도 남은 관측 시간 동안 계속 기록합니다. 시간이 끝나면 자동 종료되며 **시행 종료**로 일찍 끝낼 수도 있습니다.
4. **시행 검토**에서 실제 행동, 실제 낙상 여부와 시행 유효성을 입력하고 저장합니다. **다음 시행**을 눌러 다음 방향으로 진행하고, 마지막에는 **실험 완료**를 누릅니다. 실험을 중간에 마치려면 **실험 종료**를 누릅니다.
5. **저장된 실험**에서 과거 결과를 다시 검토할 수 있습니다. 브라우저 새로고침은 진행 중인 시행을 초기화하지 않습니다. 서버를 재시작하면 이전에 기록 중이던 시행은 중단으로 보존됩니다.

저장 위치는 다음과 같습니다.

- `data/experiments/<참가자 ID>/<실험 ID>/trials.csv` — 준비·기록·종료 상태, 계획 동작과 실제 행동, 관측 시간, 최초 확정 시각, 수신/유효 관절/UNKNOWN 프레임 수, 연결 공백 및 검토 결과.
- `data/experiments/<참가자 ID>/<실험 ID>/trial-001/<실행별 폴더>/frames.csv` — 시행 ID가 연결된 기존 관절·신뢰도·각도·하강량·FSM 진단 기록.
- 같은 시행 폴더의 `frame-*.jpg` — 감지 성공 여부와 관계없이 기록한 원본·관절 비교 사진. `FALL_DEBUG=frames`일 때 저장되며 대시보드에 최신 사진이 표시됩니다.
- `data/experiments/experiments.sqlite3` — 서버 종료 후에도 유지되는 실험 원본 기록. CSV를 Excel에서 열어 파일이 잠겨도 이 기록을 기준으로 대시보드에서 전체 결과를 내려받을 수 있습니다.

기존 `data/fall_debug` 기록도 함께 유지됩니다. 사진은 로컬에 저장됩니다. 실험 중 Temi 세션이 바뀌면 해당 프레임을 평가에서 제외하고 경고하며, 새 세션으로 다시 준비한 뒤 다음 시행을 시작하세요. 2초가 넘는 수신 공백과 수신 없는 시행도 기록합니다. 진단 저장 대기열이 가득 차면 사진·프레임 로그 일부가 빠질 수 있으며 누락 수와 저장 오류 수를 시행 결과에 표시합니다.

방향별 표에는 확정 여부를 집계합니다. TP, FN, FP, TN과 Precision, Recall, Specificity, F1은 **종료되고, 실험자가 유효성과 실제 정답을 확인했으며, 유효 관절 프레임이 있는 시행**만 계산합니다. 미검토·무효·중단·전부 인식 불가인 시행은 제외 건수로 표시합니다. 최초 확정까지의 시간은 시행 시작 기준입니다. 실제 낙상 발생부터의 지연을 평가하려면 영상으로 실제 발생 시각을 별도 표기해야 합니다.

추가 API는 `/experiment/api/state`, `create`, `start`, `end`, `next`, `finish`, `review`, `select`, `preview/{실험 ID}/{시행 ID}`, `results.csv`입니다. 조작 요청은 현재 시행 ID로 검증하므로 같은 시작·종료·다음 요청을 반복해도 시행이 중복 생성되거나 건너뛰지 않습니다.

## 저장한 영상 분석

Temi 등으로 촬영한 영상 한 개를 노트북에서 분석할 수 있습니다. 예를 들어 `data/videos` 폴더를 만들고 그 안에 영상을 저장하세요. 이 폴더와 분석 결과가 저장되는 `data` 폴더는 Git에 올라가지 않습니다. 다른 위치의 영상도 경로를 직접 지정해 분석할 수 있습니다.

VS Code의 Command Prompt 터미널에서 `pose-game` 폴더를 열고 가상환경을 활성화한 뒤 실행합니다. 영상 파일명은 실제 이름으로 바꾸세요.

```cmd
.venv\Scripts\activate.bat
python analyze_video.py fall "data\videos\fall.mp4"
python analyze_video.py rhythm "data\videos\rhythm.mp4"
```

`fall`은 낙상 감지, `rhythm`은 리듬게임 동작 분석입니다. 명령 한 번에 한 영상씩 분석하며 기존 서버나 Firebase를 실행하지 않습니다. 낙상 영상은 동작 전 서 있는 모습과 넘어진 후 유지되는 모습을 모두 포함해야 합니다.

결과는 `data/video_analysis`의 새로운 CSV 파일에 저장되고 실행이 끝나면 파일 경로가 표시됩니다. CSV에는 분석한 프레임의 영상 시각과 판정이 기록됩니다. 낙상은 인식 불가일 때 `UNKNOWN`을 기록하며, 리듬게임은 여러 동작이 동시에 감지되면 후보를 모두 기록하고 최종 판정은 비워 둡니다. 실험의 정확도를 구하려면 별도로 기록한 정답과 비교해야 합니다.

박수는 두 손목이 가까워지는 프레임에 `CLAP`으로 기록하고, 양손이 다시 벌어져야 다음 박수를 기록합니다. 손목이 가려지거나 화면 밖으로 나가면 박수를 놓칠 수 있습니다. 손목 거리는 손바닥의 실제 접촉 여부를 보장하지 않습니다.

기본 분석 간격은 영상 시간 기준 낙상은 최소 0.2초, 리듬게임은 최소 0.1초입니다. 필요하면 간격이나 저장 경로를 지정할 수 있습니다.

```cmd
python analyze_video.py fall "data\videos\fall.mp4" --interval 0.5 --output "data\video_analysis\fall_result.csv"
```

기존 결과 파일은 덮어쓰지 않습니다. 영상 프레임 번호와 FPS로 영상 시각을 계산하므로 **고정 프레임 속도 영상**을 사용하세요. 가변 프레임 속도 영상은 고정 프레임 속도로 변환한 뒤 분석해야 정확한 시간 비교가 가능합니다. 오프라인 결과는 Temi의 실제 네트워크 지연이나 프레임 누락을 재현하지 않습니다.
