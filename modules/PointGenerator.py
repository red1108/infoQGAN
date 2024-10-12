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

    def generate_biased_box(self):
        xx = np.random.uniform(0.5, 0.9, self.data_num) # x 좌표 계산
        yy = np.random.uniform(0.5, 0.9, self.data_num)  # y 좌표 계산
        return self._combine_and_shuffle(xx, yy)


    def generate_biased_box2(self):
        xx = np.random.uniform(0.1, 0.8, self.data_num) # x 좌표 계산
        yy = np.random.uniform(0.1, 0.8, self.data_num)  # y 좌표 계산
        return self._combine_and_shuffle(xx, yy)
    
    def generate_biased_donut(self):
        # Generate random angles
        angles = np.random.uniform(0, 2 * np.pi, self.data_num)
        
        # Generate random radii uniformly between inner and outer radius
        radii = np.sqrt(np.random.uniform(0.15**2, 0.3**2, self.data_num))
        
        # Calculate x and y coordinates based on polar coordinates
        xx = 0.4 + radii * np.cos(angles)
        yy = 0.4 + radii * np.sin(angles)

        return self._combine_and_shuffle(xx, yy)

    def generate_biased_curve(self):
        xx = np.linspace(0.1, 0.5, self.data_num)  # 범위 조정
        yy = 7 * (xx - 0.3) * (xx - 0.3) + 0.1 + (0.1 * np.random.rand(self.data_num) - 0.03)  # 중심을 x=0.3으로 이동
        
        return self._combine_and_shuffle(xx, yy)

    def generate_bar(self):
        # 중심 (0.5, 0.5)에서 살짝 벗어나 있는 얇은 막대기 분포 생성
        xx = np.random.uniform(0.3, 0.8, self.data_num) + 0.012 * np.random.randn(self.data_num)  # x 좌표 계산 (좁은 범위)
        yy = np.random.uniform(0.6, 0.7, self.data_num) + 0.012 * np.random.randn(self.data_num)  # y 좌표 계산 (좁은 범위)
        return self._combine_and_shuffle(xx, yy)

    def generate_curve(self):
        xx = np.linspace(0.1, 0.9, self.data_num)# + 0.02 * np.random.randn(self.data_num)  # x 좌표 계산
        yy = 3.8 * xx * xx - 3.8 * xx + 1.2 + (0.06 * np.random.rand(self.data_num)-0.03)  # y 좌표 계산
        return self._combine_and_shuffle(xx, yy)

    def generate_circle(self):
        # 중심 (0.3, 0.3) 반지름 0.2 내부에서 uniform 해야 함
        
        xx = []
        yy = []
        while len(xx) < self.data_num:
            x = np.random.uniform(0, 1)  # 원을 포함하는 사각형 내에서 x 좌표 생성
            y = np.random.uniform(0, 1)  # 원을 포함하는 사각형 내에서 y 좌표 생성
            if (x - 0.3) ** 2 + (y - 0.3) ** 2 <= 0.2 ** 2:
                # 원 내부에 있는 경우에만 추가
                xx.append(x + 0.012 * np.random.rand())
                yy.append(y + 0.012 * np.random.rand())
        xx = np.array(xx)
        yy = np.array(yy)
        return self._combine_and_shuffle(xx, yy)


    def generate_circle2(self):
        # 중심 (0.3, 0.3) 반지름 0.25 내부에서 uniform 해야 함
        
        xx = []
        yy = []
        while len(xx) < self.data_num:
            x = np.random.uniform(0, 1)  # 원을 포함하는 사각형 내에서 x 좌표 생성
            y = np.random.uniform(0, 1)  # 원을 포함하는 사각형 내에서 y 좌표 생성
            if (x - 0.3) ** 2 + (y - 0.3) ** 2 <= 0.25 ** 2:
                # 원 내부에 있는 경우에만 추가
                xx.append(x + 0.012 * np.random.rand())
                yy.append(y + 0.012 * np.random.rand())
        xx = np.array(xx)
        yy = np.array(yy)
        return self._combine_and_shuffle(xx, yy)
    
    def generate_double_circle(self):
        # 원 1: 중심 (0.4, 0.4), 반지름 0.1 내부에서 uniform 해야 함
        # 원 2: 중심 (0.6, 0.6), 반지름 0.1 내부에서 uniform 해야 함
        
        xx = []
        yy = []
        
        while len(xx) < self.data_num // 2:
            x = np.random.uniform(0, 1)  # 원을 포함하는 사각형 내에서 x 좌표 생성
            y = np.random.uniform(0, 1)  # 원을 포함하는 사각형 내에서 y 좌표 생성
            if (x - 0.4) ** 2 + (y - 0.4) ** 2 <= 0.2*0.2/2:
                # 첫 번째 원 내부에 있는 경우에만 추가
                xx.append(x + 0.012 * np.random.rand())
                yy.append(y + 0.012 * np.random.rand())
        
        while len(xx) < self.data_num:
            x = np.random.uniform(0, 1)  # 원을 포함하는 사각형 내에서 x 좌표 생성
            y = np.random.uniform(0, 1)  # 원을 포함하는 사각형 내에서 y 좌표 생성
            if (x - 0.6) ** 2 + (y - 0.6) ** 2 <= 0.2*0.2/2:
                # 두 번째 원 내부에 있는 경우에만 추가
                xx.append(x + 0.012 * np.random.rand())
                yy.append(y + 0.012 * np.random.rand())
        
        xx = np.array(xx)
        yy = np.array(yy)
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