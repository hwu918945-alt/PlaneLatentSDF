"""Generate a tiny nonclinical integration fixture, not an M&Ms-2 replacement."""
import argparse
from pathlib import Path
import numpy as np
from core.data import polygon_sdf, resample_contour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='synthetic')
    args = parser.parse_args()
    rng = np.random.default_rng(19)
    root = Path(args.output)
    for split, count in [('train', 4), ('val', 2)]:
        (root / split).mkdir(parents=True, exist_ok=True)
        for c in range(count):
            s = np.linspace(-0.5, 0.5, 7).astype(np.float32)
            qs, ds, bs = [], [], []
            angle = np.linspace(0, 2 * np.pi, 128, endpoint=False)
            for pos in s:
                r = np.sqrt(1 - (pos / 0.7) ** 2)
                ax = (0.34 + 0.018 * c) * r
                ay = (0.25 + 0.008 * c) * r
                polygon = np.stack([ax * np.cos(angle) + 0.05 * pos, ay * np.sin(angle)], -1)
                bc = resample_contour(polygon, 256)
                point = bc[rng.integers(len(bc), size=384)] + rng.normal(0, 0.035, (384, 2))
                point = np.r_[point, rng.uniform(-0.5, 0.5, size=(128, 2))]
                qs.append(np.c_[point, np.full(len(point), pos)])
                ds.append(polygon_sdf(point, polygon))
                bs.append(np.c_[bc, np.full(len(bc), pos)])
            np.savez_compressed(root / split / f'{split}_{c}_ED.npz',
                                q=np.asarray(qs, np.float32), sdf2d=np.asarray(ds),
                                boundary=np.asarray(bs, np.float32), s=s,
                                axis=np.array([0, 0, 1], np.float32), center_mm=np.zeros(3),
                                scale_mm=np.array(100.), phase=np.array(0), case_id=np.array(f'{split}_{c}'))
    print(f'Synthetic fixtures created: {root.resolve()}')


if __name__ == '__main__':
    main()
