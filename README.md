# InfoQGAN

InfoQGAN은 InfoGAN의 양자 버전입니다. 이 저장소에는 InfoQGAN 모델을 학습하고, 결과를 시각화하는 코드를 포함하고 있습니다.

## 시작하기

시작하기 전에, 루트 디렉터리에 `runs` 폴더를 생성하세요. 이 폴더는 TensorBoard 데이터와 학습 로그를 저장하는 데 사용되며, git에서 무시됩니다.

visualize/tmp/에 생성된 이미지는 임시 저장용으로 Git에서 무시됩니다. 논문에 넣을 최종적인 이미지는 visualize/로 이동시켜주세요.

## 폴더 구조

```bash
INFOQGAN/
├── data/                           # 데이터 관련 파일 저장 폴더
│   ├── 2D/                         # 2D 학습에 사용되는 데이터
│   ├── MNIST/                      # MNIST 학습에 사용되는 데이터
├── modules/                        # 모델 구성 요소 및 유틸리티 코드 폴더
│   ├── Discriminator.py            # 판별자(Discriminator) 모델 코드
│   ├── MINE.py                     # Mutual Information Neural Estimator (MINE) 코드
│   ├── PointGenerator.py           # 2D 분포 만드는 코드
│   ├── QGAN.py                     # QGAN 모델 정의 코드
│   └── utils.py                    # 유틸리티 함수 모음
├── runs/                           # TensorBoard 및 학습 로그 저장 폴더
├── savepoints/                     # autoencoder 파라미터 저장
├── visualize/                      # 이미지 생성 및 시각화 코드 폴더
│   ├── tmp/                        # 학습 중 생성된 임시 이미지 저장 (Git에서 무시됨)
│   ├── disentanglement.ipynb 
│   ├── mode_collapse_box.ipynb 
│   ├── mode_collapse_diamond_timeline.ipynb
├── .gitignore
├── mnist_train.py                  # (중요) MNIST 학습 수행하는 파이썬 파일
├── mnist_train.ipynb               # MNIST 학습 수행하는 노트북
├── 2D_prepare.ipynb                # 2D 실험 데이터 생성하는 노트북
├── 2D_train.ipynb                  # 2D 실험 학습 노트북
├── 2D_run.ipynb                    # 2D 학습 완료된 모델을 불러와서 분포 생성하는 노트북
├── README.md                       # 프로젝트 설명 및 사용법이 담긴 README 파일
├── requirements.txt                # 파이썬 3.9 기준으로 작성됨.
```

### 요구 사항

- Python 3.9 또는 그 이상 (**이 코드는 파이썬 3.9 기준으로 작성되었습니다.**)
- 필수 패키지는 `requirements.txt`에 정의되어 있습니다.

종속성을 설치하려면 다음 명령어를 실행하세요:
```bash
pip install -r requirements.txt
```

### 실행 방법
InfoQGAN으로 학습하고 싶은 경우
```bash
python mnist_train.py --model_type InfoQGAN --DIGITS_STR 0123456789 --DIGIT 1 --G_lr 0.01 --M_lr 0.0001 --D_lr 0.001 --coeff 0.05 --epochs 300 --latent_dim 16 --num_images_per_class 2000
```

QGAN으로 학습하고 싶은 경우
```bash
python mnist_train.py --model_type QGAN --DIGITS_STR 0123456789 --DIGIT 1 --G_lr 0.01 --M_lr 0.0001 --D_lr 0.001 --coeff 0.05 --epochs 300 --latent_dim 16 --num_images_per_class 2000
```