# InfoQGAN

InfoQGAN은 InfoGAN의 양자 버전입니다. 이 저장소에는 InfoQGAN 모델을 학습하고, 결과를 시각화하는 코드를 포함하고 있습니다.

## 시작하기

시작하기 전에, 루트 디렉터리에 `runs` 폴더를 생성하세요. 이 폴더는 TensorBoard 데이터와 학습 로그를 저장하는 데 사용됩니다.

visualize/tmp/에 생성된 이미지는 임시 저장용으로 Git에서 무시됩니다. 논문이나 보고서에 사용할 이미지는 visualize/로 이동하여 보관하세요.

## 폴더 구조
```bash
INFOQGAN/
├── data/                           # 데이터 관련 파일 저장 폴더
├── modules/                        # 모델 구성 요소 및 유틸리티 코드 폴더
│   ├── Discriminator.py            # 판별자(Discriminator) 모델 코드
│   ├── MINE.py                     # Mutual Information Neural Estimator (MINE) 코드
│   ├── PointGenerator.py           # 2D 분포 만드는 코드
│   ├── QGAN.py                     # QGAN 모델 정의 코드
│   └── utils.py                    # 유틸리티 함수 모음
├── runs/                           # TensorBoard 및 학습 로그 저장 폴더
├── visualize/                      # 이미지 생성 및 시각화 코드 폴더
│   ├── tmp/                        # 학습 중 생성된 임시 이미지 저장 (Git에서 무시됨)
│   ├── disentanglement.ipynb 
│   ├── mode_collapse_box.ipynb 
│   ├── mode_collapse_diamond_timeline.ipynb
├── .gitignore
├── 2D_prepare.ipynb                # 2D 실험 데이터 생성하는 노트북
├── **2D_train.ipynb**              # 2D 실험 학습 노트북
├── 2D_run.ipynb                    # 2D 학습 완료된 모델을 불러와서 분포 생성하는 노트북
├── README.md                       # 프로젝트 설명 및 사용법이 담긴 README 파일

### 요구 사항

- Python 3.9 또는 그 이상
- 필수 패키지는 `requirements.txt`에 정의되어 있습니다.

종속성을 설치하려면 다음 명령어를 실행하세요:
```bash
pip install -r requirements.txt