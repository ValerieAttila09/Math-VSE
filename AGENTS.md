# MathVSE — Project Instructions

Build MathVSE, a local-first mathematical image solver in Python. No LLMs, generative AI, cloud inference, or remote APIs at runtime.

## Architecture
- Vision: OpenCV, NumPy, Pillow.
- OCR: optional small PyTorch CNN; no fake predictions or automatic model downloads.
- Spatial parser: reconstruct mathematical 2D structure.
- Math engine: custom lexer, parser, typed AST, rule-based linear solver, transformation trace.
- Verification: SymPy adapter plus independent solution/trace validation.
- Explanations: deterministic Indonesian templates generated only from validated trace events.
- UI: Streamlit.

## MVP
Support linear equations in one variable, integer/rational arithmetic, parentheses, unary signs, explicit/implicit multiplication, unique solutions, no solutions, and infinitely many solutions. Reject unsupported expressions safely.

## Engineering Rules
- Inspect the repository before editing; preserve unrelated work.
- Implement one phase at a time; prefer minimal, targeted diffs.
- Use type hints, clear module boundaries, meaningful errors, and pytest.
- Never use eval/exec on user input.
- Never fabricate OCR output, confidence, test results, or mathematical steps.
- Never solve ambiguous OCR output without review/confirmation.
- Keep UI, OCR, parser, solver, validator, and explanations separate.
- Record every solver transformation with before/after states, rule ID, operation parameters, and preconditions.
- Verify candidate solutions against the original equation using exact arithmetic where possible.
- Keep CPU execution supported; GPU is optional.
- Runtime must work offline once dependencies and model assets are installed.
- Do not add dependencies, abstractions, or files without a clear need.

## Workflow
1. Read the current code and relevant tests.
2. Identify the smallest change that fulfills the request.
3. Implement only the requested phase/task.
4. Run targeted tests, then relevant regression tests.
5. Fix failures caused by the changes.
6. Report changed files, tests and actual results, limitations, and next step concisely.

Do not reimplement completed features. Do not rewrite unrelated files. Do not claim completion without verification. Update README/docs only when behavior or setup changes.