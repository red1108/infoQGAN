import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from io import BytesIO
import os
from tqdm import tqdm

# 설정
base_dir = './runs/MNIST0123456789_35_ld16_InfoQGAN_Nov16_14_49_26'
output_video = os.path.join(base_dir, 'params.mp4')  # 저장 경로 변경
epochs = range(1, 201)
frame_duration = 0.5  # 초
frame_size = (800, 400)  # 이미지 크기
dpi = 100

# 동영상 생성 준비
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, 1/frame_duration, frame_size)

# 데이터 시각화 및 동영상에 추가
for epoch in tqdm(epochs):
    # generator 파라미터 로드
    file_path = f'{base_dir}/params/generator_params_epoch{epoch}.pth'
    if not os.path.exists(file_path):
        print(f"파일 {file_path}이(가) 존재하지 않습니다. 건너뜁니다.")
        continue

    generator_initial_params = torch.load(file_path)
    data = generator_initial_params.detach().numpy().squeeze()  # [20, 5, 1] -> [20, 5]
    data_reshaped = data.T  # [5, 20]

    # matplotlib을 이용한 시각화
    plt.figure(figsize=(frame_size[0]/dpi, frame_size[1]/dpi), dpi=dpi)
    plt.imshow(data_reshaped, cmap='viridis', aspect='auto')
    plt.colorbar(label="Parameter Value")
    plt.title(f"epoch={epoch}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    # 이미지를 메모리에 저장
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()

    # OpenCV용 이미지로 변환
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, frame_size)  # 크기 조정

    # 동영상에 프레임 추가
    video.write(img)

# 동영상 저장 종료
video.release()
print(f"동영상 생성 완료: {output_video}")
