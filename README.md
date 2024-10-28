# InfoQGAN

InfoQGAN은 InfoGAN의 양자 버전입니다. 이 저장소에는 InfoQGAN 모델을 학습하고, 결과를 시각화하는 코드를 포함하고 있습니다.

## 시작하기

시작하기 전에, 루트 디렉터리에 `runs` 폴더를 생성하세요. 이 폴더는 TensorBoard 데이터와 학습 로그를 저장하는 데 사용됩니다.

visualize/tmp/에 생성된 이미지는 임시 저장용으로 Git에서 무시됩니다. 논문이나 보고서에 사용할 이미지는 visualize/로 이동하여 보관하세요.

## 폴더 구조

md
packages/button
├── lib
│   ├── button.d.ts
│   ├── button.js
│   ├── button.js.map
│   ├── button.stories.d.ts
│   ├── button.stories.js
│   ├── button.stories.js.map
│   ├── index.d.ts
│   ├── index.js
│   └── index.js.map
├── package.json
├── src
│   ├── button.stories.tsx
│   ├── button.tsx
│   └── index.ts
└── tsconfig.json

### 요구 사항

- Python 3.9 또는 그 이상
- 필수 패키지는 `requirements.txt`에 정의되어 있습니다.

종속성을 설치하려면 다음 명령어를 실행하세요:
```bash
pip install -r requirements.txt