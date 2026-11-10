
## Cấu trúc thư mục

```
Experiments/
├── config.py              # Table 1: toàn bộ tham số, f(x)=0.5x, tol, max_iter, seed
├── run_all.py             # CLI chạy các ví dụ, xuất bảng + hình
├── requirements.txt
├── src/
│   ├── operators.py       # soft-threshold, sigmoid, λ_max(TᵀT)
│   ├── problems.py        # Ex 4.1 (SFP trong L²), 4.2 LASSO, 4.3 Elastic Net, 4.4 SLR
│   ├── algorithms.py      # Our Alg. (Algorithm 1), Alg1–Alg4 (dạng hàm, không OOP)
│   ├── runner.py          # vòng lặp chung, TOL(k), đo CPU time, bảng kết quả
│   └── plotting.py        # đồ thị TOL(k) theo vòng lặp / CPU time
├── notebooks/
│   └── experiments.ipynb  # chạy từng ví dụ, hiển thị bảng và hình
├── tests/test_algorithms.py
└── results/{figures,tables}/
```

## Chạy

```bash
pip install -r requirements.txt
python run_all.py                          # Ex 4.1–4.4, err = 1e-5
python run_all.py --examples 4.1 --tol 1e-8 --x0-case 2t_16
pytest -q
```

## Thuật toán ↔ nguồn

| Tên | Nguồn | Hàm |
|---|---|---|
| Our Alg. | Algorithm 1 (ARAIFBS) | `make_ours` |
| Alg1 | Boţ–Sedlmayer–Vuong (JMLR 2023), RIFBF | `make_alg1` |
| Alg2 | Cholamjiak–Hieu–Cho (JSC 2021), Algorithm 2 | `make_alg2` |
| Alg3 | Thuy–Ly (Acta Appl. Math. 2026), Algorithm 3.1, N=1 | `make_alg3` |
| Alg4 | Thuy–Tung–Ly (COAM 2024), Algorithm 3.1, N=1 | `make_alg4` |



