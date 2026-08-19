import numpy as np
import matplotlib.pyplot as plt

#이 코드는 랜덤한 위치와 전하량을 가진 전하들을 생성하여 그에 대한 전기장을 나타낸다
class EC:
    def __init__(self, x, y, z, q):
        self.x = x
        self.y = y
        self.z = z
        self.q = q

EPSILON_0 = 1 #편의상 진공 유전율을 1로 설정한다


def potential(p, neg):
    r  = np.sqrt((X-p.x)**2 + (Y-p.y)**2 + p.z**2)
    return 1/(4*np.pi*EPSILON_0) * p.q / r * neg


#전하는 3차원에 있지만 관측 평면은 z=0으로 설정하였다
gridN = 32 #공간 해상도
x = np.linspace(-2, 2, gridN)
y = np.linspace(-2, 2, gridN)

X, Y = np.meshgrid(x, y)

N = 2

Charges = []

for i in range(N):
    cx = np.random.uniform(-1.5, 1.5)
    cy = np.random.uniform(-1.5, 1.5)
    cz = np.random.uniform(0.1, 1.5)
    cq = np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0)

    Charges.append(EC(cx, cy, cz, cq))

Charges.sort(key=lambda c: (c.x, c.y, c.z)) #회귀 모델이 학습하기 쉽도록 순열 요소를 제거한다

V_p = sum(potential(Charges[i], 1) for i in range(N))
V_n = sum(potential(Charges[i], -1) for i in range(N))

Obs_p = V_p**2
Obs_n = V_n**2

print("V^2 최대 차이:", np.max(np.abs(Obs_p - Obs_n)))

plt.imshow(
    Obs_p,
    extent=[-2, 2, -2, 2],
    origin="lower"
)

plt.colorbar(label="V^2")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Observed field: V^2")

plt.show()