import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# --- 1. Загрузка и нормировка данных --------------------------------
def load_and_prepare_data(path, chunksize=1_000_000):
    """Загружает CSV, очищает NaN и возвращает центрированные и нормированные X,Y,Z"""
    dfs = []
    total_bytes = os.path.getsize(path)
    with tqdm(total=total_bytes, unit='B', unit_scale=True, desc='Загрузка CSV') as pbar:
        for df_chunk in pd.read_csv(path, chunksize=chunksize):
            dfs.append(df_chunk)
            pbar.update(df_chunk.memory_usage(deep=True).sum())
    df = pd.concat(dfs, ignore_index=True)

    with tqdm(total=3, desc='Очистка данных', leave=False) as p:
        df = df.dropna(subset=['X','Y','Z']);        p.update(1)
        df[['X','Y','Z']] = df[['X','Y','Z']].apply(pd.to_numeric, errors='coerce'); p.update(1)
        df = df.dropna(subset=['X','Y','Z']);        p.update(1)
    print(f"Всего точек: {len(df):,}")

    x = df['X'].to_numpy(dtype=np.float32)
    y = df['Y'].to_numpy(dtype=np.float32)
    z = df['Z'].to_numpy(dtype=np.float32)

    # Центрирование и перевод в километры
    x0, y0 = x.mean(), y.mean()
    x = (x - x0) / 1000.0
    y = (y - y0) / 1000.0

    return x, y, z, x0, y0

# --- 2. Обучение GPR --------------------------------------------------
def train_gpr(x, y, z, sample_size=None, n_restarts=2):
    if sample_size and len(x) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), sample_size, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        print(f"Подвыборка: {sample_size} из {len(x)} точек")

    X = np.column_stack([x, y])
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=[0.5,0.5], length_scale_bounds=(0.01,5.0), nu=1.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5,1))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        alpha=0.0
    )
    print("Обучение GPR...")
    gpr.fit(X, z)
    print("Обучено. Ядро:", gpr.kernel_)
    return gpr

# --- 3. Предсказание на сетке и визуализация ---------------------------
def predict_and_plot(gpr, x0, y0, x_min, x_max, y_min, y_max,
                     grid_size=800, chunk_size=50000, out_png="relief_map.png"):
    gx_raw = np.linspace(x_min, x_max, grid_size)
    gy_raw = np.linspace(y_min, y_max, grid_size)
    gx = (gx_raw - x0) / 1000.0
    gy = (gy_raw - y0) / 1000.0
    xv, yv = np.meshgrid(gx, gy)
    coords = np.column_stack([xv.ravel(), yv.ravel()])

    n_pts = coords.shape[0]
    n_ch = math.ceil(n_pts / chunk_size)
    zv_flat = np.empty(n_pts, dtype=np.float32)
    with tqdm(total=n_ch, desc="Predict grid") as p:
        for i in range(n_ch):
            s, e = i*chunk_size, min((i+1)*chunk_size, n_pts)
            zv_flat[s:e] = gpr.predict(coords[s:e])
            p.update(1)
    zv = zv_flat.reshape(xv.shape)

    # Подготовка уровней: гарантируем монотонность
    zmin, zmax = np.nanmin(zv), np.nanmax(zv)
    if zmin >= zmax:
        zmin -= 0.5
        zmax += 0.5
    levels = np.linspace(zmin, zmax, 60)

    xv_plot = xv * 1000.0 + x0
    yv_plot = yv * 1000.0 + y0

    fig, ax = plt.subplots(figsize=(10,8))
    cf = ax.contourf(xv_plot, yv_plot, zv, levels=levels, cmap='terrain', antialiased=True)
    cs = ax.contour(xv_plot, yv_plot, zv, levels=levels, colors='k', linewidths=0.6)
    ax.clabel(cs, fmt='%.2f', fontsize=6)
    fig.colorbar(cf, ax=ax, label='Глубина, м')
    ax.set_title('Карта рельефа (GPR)')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Сохранено: {out_png}")

# ======================= MAIN =========================================
if __name__ == '__main__':
    FILE = 'all_points_cleaned.csv'
    x, y, z, x0, y0 = load_and_prepare_data(FILE)
    gpr = train_gpr(x, y, z, sample_size=5000, n_restarts=2)
    predict_and_plot(
        gpr, x0, y0,
        x.min(), x.max(), y.min(), y.max(),
        grid_size=800, chunk_size=50000,
        out_png='relief_map.png'
    )
