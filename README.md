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
