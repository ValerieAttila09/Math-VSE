# MathVSE

MathVSE adalah aplikasi lokal-first untuk memecahkan persamaan linear satu variabel dari teks maupun citra, dengan jejak transformasi dan penjelasan dalam bahasa Indonesia.

## Fitur yang sudah diimplementasikan

- Lexer dan parser kustom untuk ekspresi matematika
- AST eksplisit untuk angka, variabel, unary, binary, dan persamaan
- Pendeteksi implicit multiplication seperti `2x` dan `3(x + 2)`
- Solver linear satu variabel dengan cakupan:
  - solusi unik
  - tidak ada solusi
  - tak hingga banyak solusi
  - persamaan pecahan seperti `x/2 + 3 = 7`
- Trace transformasi yang dihasilkan oleh solver itu sendiri
- Validator solusi terpisah
- Generator penjelasan deterministik dalam bahasa Indonesia
- Workflow text-input via pipeline orchestrator

## Struktur utama

- `src/mathvse/math_engine/`: lexer, parser, AST, formatter, solver, validator
- `src/mathvse/explanation/`: template dan generator penjelasan
- `src/mathvse/contracts/`: kontrak data dan model hasil
- `src/mathvse/pipeline/`: orchestrator workflow text input
- `tests/unit/`: pengujian parser, solver, trace, validator, dan penjelasan

## Prasyarat

- Python 3.10+
- `pip`
- koneksi internet saat instalasi untuk mengunduh dependensi `sympy`

## Instalasi

```bash
cd /workspaces/Math-VSE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Menjalankan pengujian

```bash
cd /workspaces/Math-VSE
pytest -q
```

## Contoh penggunaan

```python
from mathvse.pipeline.orchestrator import solve_text_expression

result = solve_text_expression("3*x - 6 = 12")
print(result.status)
print(result.solution)
print(result.trace[-1].after)
```

## Cakupan yang didukung saat ini

- Persamaan linear satu variabel
- Bilangan bulat dan rasional sederhana
- Penjumlahan, pengurangan, perkalian, dan pembagian
- Kurung
- Tanda unary seperti `-x`
- Implicit multiplication seperti `2x`

## Batasan yang diketahui

- Belum ada OCR atau pemrosesan citra
- Belum ada parser spasial untuk ekspresi 2D kompleks
- Belum ada model CNN atau image preprocessing
- Fokus saat ini adalah fondasi matematika dan alur text-input

## Catatan pengembangan

Sistem ini sengaja dibangun secara modular dan deterministik tanpa penggunaan `eval`/`exec` pada input pengguna. Semua hasil solusi divalidasi terhadap persamaan awal dan dijelaskan berdasarkan trace transformasi yang dihasilkan oleh solver.
