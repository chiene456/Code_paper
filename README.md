# ARAIFBS — mã nguồn thí nghiệm (Section 4)

Cài đặt Example 4.1–4.4 của bài báo *An Alternated Inertial Viscosity Method with Adaptive
Stepsizes and Relaxation for Monotone Inclusions* (ThuySonChien_Final3) và 4 thuật toán đối sánh.

## Cấu trúc thư mục

```
araifbs_experiments/
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

Tiêu chuẩn dừng chung: `TOL(k) = ‖x_k − J_{λ_k}^A(x_k − λ_k B x_k)‖ ≤ err` (λ_k là bước nhảy hiện hành của từng thuật toán).
Điểm khởi đầu chung: `x_{-1} = x_0 = x_1 = x0`.

## Các giả định (paper chưa nêu rõ — chỉnh trong `config.py` / `problems.py`)

1. **Điểm khởi tạo Ex 4.1** không được nêu → mặc định `x0(t) = t²/10`.
2. **Alg1** (Table 1 ghi `λ_k = a/ℓ`, `ρ_k ≤ 2/(1+aθ_k)`, `θ_k = λ_k/λ_{k+1}`): mặc định bước nhảy hằng `a/ℓ`
   (`θ_k = 1`) và `ρ_k = 2/(1+aθ_k)`. Có thể đổi `stepsize_mode="adaptive"` hoặc `rho_mode="theory"`.
3. **Alg3** cần toán tử `F`: dùng `F(x) = 0.5x` (bản gốc ACTAS dùng `7x`).
4. Ex 4.2–4.4: ma trận/nhãn sinh theo mô tả, `seed = 42`; điểm khởi tạo `randn`.
