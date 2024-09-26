import numpy as np

class PointGenerator:
    def __init__(self, data_num=2000):
        self.data_num = data_num
    
    def generate_spiral(self):
        theta = np.linspace(0, 5*np.pi, self.data_num)  # 각도 범위를 정의합니다.
        radius = np.linspace(0, 0.3, self.data_num)  # 반지름 범위를 정의합니다.
        xx = 0.5 + radius * np.cos(theta) + 0.012 * np.random.randn(self.data_num)  # x 좌표 계산
        yy = 0.5 + radius * np.sin(theta) + 0.012 * np.random.randn(self.data_num)  # y 좌표 계산
        return self._combine_and_shuffle(xx, yy)

    def generate_box(self):
        xx = np.random.uniform(0.3, 0.7, self.data_num) + 0.012 * np.random.randn(self.data_num)  # x 좌표 계산
        yy = np.random.uniform(0.3, 0.7, self.data_num) + 0.012 * np.random.randn(self.data_num)  # y 좌표 계산
        return self._combine_and_shuffle(xx, yy)

    def generate_curve(self):
        xx = np.linspace(0.15, 0.85, self.data_num) + 0.02 * np.random.randn(self.data_num)  # x 좌표 계산
        yy = 3.8 * xx * xx - 3.8 * xx + 1.2 + 0.03 * np.random.randn(self.data_num)  # y 좌표 계산
        return self._combine_and_shuffle(xx, yy)

    def generate_circle(self):
        radius = 0.05
        theta = np.linspace(0, 2 * np.pi, self.data_num)  # 각도 범위를 정의합니다.
        xx = radius * np.cos(theta) + 0.3 + 0.02 * np.random.randn(self.data_num)  # x 좌표 계산
        yy = radius * np.sin(theta) + 0.3 + 0.02 * np.random.randn(self.data_num)  # y 좌표 계산
        return self._combine_and_shuffle(xx, yy)

    def generate_lemniscate(self):
        xx = []
        yy = []
        while len(xx) < self.data_num:
            x = np.random.uniform(0, 0.5)
            y = np.random.uniform(0, 0.5)

            r = (x**2 + y**2)**0.5
            theta = np.arctan2(y, x)
            if r**2 <= 2 * 0.25**2 * np.cos(2 * theta):
                x = x if np.random.uniform(0, 1) < 0.5 else -x
                y = y if np.random.uniform(0, 1) < 0.5 else -y
                xx.append(x + 0.5)
                yy.append(y + 0.5)
        return self._combine_and_shuffle(xx, yy)

    def _combine_and_shuffle(self, xx, yy):
        data = np.column_stack((xx, yy))  # x와 y 좌표를 합쳐서 데이터 생성
        np.random.shuffle(data)
        return data