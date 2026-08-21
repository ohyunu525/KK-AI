import numpy as np
import matplotlib.pyplot as plt

#이 코드는 랜덤한 위치와 전하량을 가진 전하들을 생성하여 그에 대한 V^2를 나타낸다
class EC:
    def __init__(self, x, y, z, q):
        self.x = x
        self.y = y
        self.z = z
        self.q = q

EPSILON_0 = 1 #편의상 진공 유전율을 1로 설정한다


def potential(ox, oy, oz, p, neg):
    r  = np.sqrt((ox-p.x)**2 + (oy-p.y)**2 + (oz-p.z)**2)
    return 1/(4*np.pi*EPSILON_0) * p.q / r * neg


#전하는 3차원에 있지만 관측 평면은 z=0으로 설정하였다
gridN = 32 #공간 해상도
x = np.linspace(-2, 2, gridN)
y = np.linspace(-2, 2, gridN)

X, Y = np.meshgrid(x, y)

N = 2                   # 전하 개수
NUM_SAMPLES = 10000     # 생성할 샘플 수
G05_POINTS = 1          # 샘플당 제공할 G05 관측점 개수


# 전체 데이터셋
dataset_G00 = []
dataset_G05 = []
dataset_target = []


# -----------------------------
# 데이터 생성
# -----------------------------

for sample in range(NUM_SAMPLES):

    Charges = []

    # 랜덤 전하 N개 생성
    for i in range(N):

        cx = np.random.uniform(-1.5, 1.5)
        cy = np.random.uniform(-1.5, 1.5)
        cz = np.random.uniform(0.1, 1.5)

        cq = (
            np.random.choice([-1, 1])
            * np.random.uniform(0.3, 1.0)
        )

        Charges.append(
            EC(cx, cy, cz, cq)
        )


    # 순열 문제 제거를 위해 위치순 정렬
    Charges.sort(
        key=lambda c: (c.x, c.y, c.z)
    )

    # 전체 G00
    V = sum(
        potential(X, Y, 0, Charges[i], 1)
        for i in range(N)
    )

    # 현재 단순화: G00 ~ V^2
    G00 = V**2


    # -----------------------------
    # 일부 G05
    # -----------------------------

    G05_samples = []

    for j in range(G05_POINTS):
        ix = np.random.randint(gridN)
        iy = np.random.randint(gridN)

        gx = x[ix]
        gy = y[iy]

        V_sample = sum(
            potential(gx, gy, 0, Charges[i], 1)
            for i in range(N)
        )

        G05_samples.append([ix, iy, V_sample])


    # -----------------------------
    # 정답 데이터
    # -----------------------------

    target = []

    for charge in Charges:

        target.extend([
            charge.x,
            charge.y,
            charge.z,
            charge.q
        ])


    # -----------------------------
    # 전체 데이터셋에 추가
    # -----------------------------

    dataset_G00.append(G00)
    dataset_G05.append(G05_samples)
    dataset_target.append(target)


# -----------------------------
# NumPy 배열로 변환
# -----------------------------

dataset_G00 = np.array(
    dataset_G00,
    dtype=np.float32
)

dataset_G05 = np.array(
    dataset_G05,
    dtype=np.float32
)

dataset_target = np.array(
    dataset_target,
    dtype=np.float32
)


# -----------------------------
# 데이터 크기 확인
# -----------------------------

print("G00 shape:", dataset_G00.shape)
print("G05 shape:", dataset_G05.shape)
print("Target shape:", dataset_target.shape)


# -----------------------------
# 파일로 저장
# -----------------------------

np.savez_compressed(
    "charge_dataset.npz",
    G00=dataset_G00,
    G05=dataset_G05,
    target=dataset_target
)


print("데이터셋 저장 완료")

idx = 0

ix = int(dataset_G05[idx, 0, 0])
iy = int(dataset_G05[idx, 0, 1])

print("G05:", dataset_G05[idx])
print("Target:", dataset_target[idx])

plt.imshow(
    dataset_G00[idx],
    extent=[-2, 2, -2, 2],
    origin="lower"
)

plt.scatter(
    x[ix],
    y[iy],
    marker="x"
)

plt.colorbar(label="G00")
plt.xlabel("x")
plt.ylabel("y")
plt.show()