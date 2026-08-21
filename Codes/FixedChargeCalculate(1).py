import numpy as np
import matplotlib.pyplot as plt

#이 코드는 고정된 전하 하나만을 다뤄 V^2를 나타낸다
class EC:
    def __init__(self, x, y, z, q):
        self.x = x
        self.y = y
        self.z = z

EPSILON_0 = 1 #편의상 진공 유전율을 1로 설정한다


def potential(p, q):
    r  = np.sqrt((X-p.x)**2 + (Y-p.y)**2 + p.z**2)
    return q / r


#전하는 3차원에 있지만 관측 평면은 z=0으로 설정하였다
gridN = 32 #공간 해상도
x = np.linspace(-2, 2, gridN)
y = np.linspace(-2, 2, gridN)

X, Y = np.meshgrid(x, y)

p0 = EC(0.3, -0.2, 0.5, 1)

V_p = potential(p0, 1)
V_n = potential(p0, -1)

Obs_p = V_p**2
Obs_n = V_n**2

print("V^2 최대 차이:", np.abs(Obs_p - Obs_n))

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