# PROJECT: MATH VISION SOLVER & EXPLAINER (MathVSE)

## 1. ROLE & RESPONSIBILITIES

Act as a Senior Python Software Engineer, Computer Vision
Engineer, Mathematical Software Engineer, Applied ML Engineer,
and Technical Architect.

Your task is to design and implement MathVSE, a local-first
mathematical image recognition and solving application.

You must prioritize:

1. Mathematical correctness.
2. Deterministic and auditable computation.
3. Clean modular architecture.
4. Explicit data contracts.
5. Reliable error handling.
6. Explainable transformations.
7. Lightweight CPU execution.
8. Offline operation.
9. Testability and maintainability.
10. A polished, usable interface.

Do not optimize for the number of features implemented.
Optimize for correctness, architectural quality, and a
working end-to-end foundation.

Do not merely produce an architecture proposal or a collection
of disconnected scripts. Build a coherent, runnable project.

Inspect the existing repository before modifying it.
Preserve useful existing code and conventions when applicable.
Do not overwrite unrelated files or configuration.

If a technical requirement is ambiguous, choose a conservative,
well-documented default and continue when possible.

Do not claim that a feature is complete unless it has been
implemented and tested.


## 2. PROJECT VISION

MathVSE is a local mathematical problem-solving application.

Its primary workflow is:

Image input
    -> image validation
    -> image preprocessing
    -> symbol recognition
    -> spatial expression reconstruction
    -> mathematical parsing
    -> symbolic solving
    -> transformation trace generation
    -> solution validation
    -> Indonesian explanation
    -> visual result presentation

The application must not depend on an LLM, generative AI,
hosted OCR API, cloud solver, or paid external API.

The core system must use explicit algorithms, locally available
models, formal mathematical rules, and symbolic computation.

The application must be capable of running offline after
dependencies and required model/data assets have been installed.

Do not claim that every stage is 100% accurate.

The symbolic engine must produce deterministic results for
supported inputs, while OCR uncertainty must be explicitly
represented and handled.

If an expression cannot be confidently reconstructed or solved,
the application must report the limitation rather than invent
a result.


## 3. MVP SCOPE

Implement the project incrementally.

The initial mathematical scope is:

- Linear equations in one variable.
- Integer and rational coefficients/constants.
- Addition and subtraction.
- Multiplication and division.
- Parentheses.
- Implicit multiplication, such as 2x.
- Explicit multiplication, such as 2*x.
- Equations containing an equality operator.
- Equations with a unique solution.
- Equations with no solution.
- Equations with infinitely many solutions.

The solver must recognize that not every equation has exactly
one solution.

Examples:

3*x - 6 = 12       -> x = 6
2*x + 5 = 15       -> x = 5
3*(x + 2) = 15     -> x = 3
x/2 + 3 = 7        -> x = 8
2*x + 1 = 2*x + 1  -> infinitely many solutions
2*x + 1 = 2*x + 3  -> no solution

These examples are acceptance tests, not a replacement for
implementing a proper parser and solver.

Do not claim support for arbitrary polynomial equations,
calculus, geometry, matrices, or unrestricted handwritten
expressions in the MVP.

Design the architecture so these capabilities can be added
later without rewriting the entire application.


## 4. REQUIRED TECHNOLOGY STACK

Language:
- Python 3.10 or newer.
- Use a supported and tested Python version consistently.
- Pin or constrain dependencies appropriately.

Core processing:
- OpenCV for image processing.
- NumPy for numerical array operations.
- Pillow for image loading and conversion.

Machine learning:
- PyTorch for a lightweight, optional CNN classifier.
- CPU inference must be supported.
- GPU must remain optional.
- Never download weights automatically during inference.

Symbolic mathematics:
- A custom lexer.
- A custom mathematical parser.
- A custom AST representation.
- A rule-based transformation engine.
- SymPy as a symbolic verification and evaluation adapter.

Interface:
- Streamlit for the initial local web-based GUI.

Testing and quality:
- pytest.
- Ruff or an equivalent Python linting/formatting tool.
- Type hints.
- Logging through Python's logging module.

Avoid unnecessary frameworks and dependencies.

Do not introduce a database, distributed queue, cloud backend,
or microservice architecture unless a real requirement emerges.

This is a local desktop-oriented application, not a SaaS backend.


## 5. ARCHITECTURE AND MODULE BOUNDARIES

Separate the application into the following components.

A. Application/UI Layer
B. Image Validation and Preprocessing
C. Symbol Recognition
D. Spatial Expression Parser
E. Mathematical Lexer and Parser
F. AST and Expression Models
G. Symbolic Rule Engine
H. Transformation Trace
I. Solution Validator
J. Indonesian Explanation Generator
K. Pipeline Orchestrator

The UI must not contain mathematical solving logic.

The image-processing layer must not contain algebraic solving logic.

The OCR layer must not directly execute expressions.

The explanation generator must not independently invent
mathematical steps.

The symbolic solver must not depend on Streamlit.

Every component must have a clear public interface and
appropriate unit tests.

Use dependency injection or straightforward constructor
injection where it improves testability.

Avoid unnecessary abstractions and overengineering.


## 6. MODULE A: IMAGE INPUT AND VALIDATION

Implement an image-input component that accepts:

- PNG.
- JPEG/JPG.
- Other image formats only when explicitly supported.

Requirements:

1. Validate file format.
2. Validate image dimensions.
3. Reject empty or unreadable files.
4. Set configurable limits for image dimensions and file size.
5. Handle malformed input safely.
6. Convert supported input into a consistent internal image
   representation.
7. Preserve the original image for preview and comparison.
8. Do not trust a file extension as proof of actual format.
9. Display understandable error messages.

Prevent malformed or excessively large images from exhausting
available memory.

All validation failures must be represented explicitly.


## 7. MODULE B: COMPUTER VISION PIPELINE

Implement a preprocessing pipeline using OpenCV.

Required capabilities:

1. Convert an image to grayscale.
2. Provide Otsu thresholding.
3. Provide adaptive thresholding.
4. Provide optional denoising.
5. Provide optional morphological operations.
6. Detect candidate contours and connected components.
7. Generate candidate bounding boxes.
8. Preserve coordinates relative to the original image.
9. Support inspection of intermediate processing results.
10. Allow preprocessing strategies to be selected or compared.

Do not assume Otsu thresholding is always the best method.

Do not assume each connected component corresponds to exactly
one mathematical character.

Examples of problematic cases include:

- The two strokes of an equals sign.
- A dot separated from the body of a symbol.
- Touching handwritten characters.
- Fraction bars.
- Minus signs.
- Parentheses.
- Superscripts and subscripts.

Do not implement destructive morphological operations without
making their effects inspectable.

Define a preprocessing result model containing, at minimum:

- Original image metadata.
- Processed image.
- Selected preprocessing strategy.
- Candidate regions.
- Bounding boxes.
- Warnings and processing status.

Write tests for empty images, low-contrast images, noisy images,
and invalid image data.


## 8. MODULE C: MATHEMATICAL OCR

Implement the recognition component behind a stable interface.

Create a symbol classifier interface that can later be backed
by a small CNN.

Each symbol prediction should include:

- Predicted class.
- Confidence score.
- Bounding box.
- Alternative candidates when available.
- Recognition status.

Confidence values must not be presented as calibrated
probabilities unless calibration has been evaluated.

Create a small CNN implementation using PyTorch.

The classifier must support:

- CPU training and inference.
- Reproducible training configuration.
- Saved model weights.
- A documented class-to-label mapping.
- Model version metadata.
- Explicit model loading.
- Graceful handling of missing model weights.

Do not automatically download pretrained models.

Do not pretend a model is trained when it is not.

If training data is unavailable, implement the model architecture,
training pipeline, dataset interface, and clear setup instructions.
The application must remain usable in text-input mode.

Dataset strategy:

- MNIST may be used for a digit-recognition baseline.
- HASYv2 may be used for mathematical symbol recognition.
- CROHME may be considered for later expression-level training
  and structural recognition.

Respect dataset licenses and usage requirements.

Do not bundle datasets without verifying redistribution rights.

Create a reproducible training/evaluation workflow with separate
training, validation, and test splits.

Prevent data leakage between these splits.

Measure per-class accuracy and confusion matrices.

Never advertise a handwritten OCR accuracy percentage without
a measured result from a documented evaluation set.


## 9. MODULE D: SPATIAL EXPRESSION RECONSTRUCTION

Create a dedicated spatial parser.

Do not simply sort every bounding box by its horizontal
coordinate and assume the resulting string is mathematically
correct.

Represent detected symbols and regions with their geometry.

Use spatial relationships to identify:

- Main expression lines.
- Superscripts.
- Subscripts when supported.
- Fraction numerator regions.
- Fraction denominator regions.
- Fraction bars.
- Parenthesized groups.
- Candidate symbols belonging to different lines.

Use relative positions, symbol dimensions, overlap, alignment,
and candidate grouping.

Avoid arbitrary hardcoded pixel thresholds whenever possible.
Use normalized geometry and configurable heuristics.

If multiple structural interpretations are plausible, preserve
the alternatives or mark the expression for review.

The spatial parser must produce a structured intermediate
representation before converting it into solver input.

For the MVP, prioritize horizontal equations and simple
spatial arrangements.

Do not claim reliable support for nested fractions, matrices,
or arbitrary two-dimensional notation until those features
have dedicated algorithms and tests.

Provide an OCR review interface that lets the user correct
the reconstructed expression before solving.


## 10. MODULE E: MATHEMATICAL LEXER AND PARSER

Build a custom lexer and parser for the supported grammar.

The parser must not depend on SymPy's default string parsing
to define the application's accepted input language.

Define the grammar explicitly.

Supported token categories should include:

- Integer literals.
- Rational numeric expressions through division.
- Variables.
- Addition.
- Subtraction.
- Multiplication.
- Division.
- Equality.
- Parentheses.
- Whitespace.
- Implicit multiplication in supported contexts.

Define operator precedence and associativity.

At minimum:

1. Parentheses have the highest precedence.
2. Unary signs must be handled explicitly.
3. Multiplication and division precede addition/subtraction.
4. Equality separates the two sides of an equation.
5. Implicit multiplication must be unambiguous within the grammar.

Handle examples such as:

2*x + 5 = 15
2x + 5 = 15
3*(x + 2) = 15
-x + 4 = 9
x/2 + 3 = 7

Reject unsupported or ambiguous expressions with useful errors.

Never use Python eval() or exec() to evaluate user input.

Never pass raw OCR strings into unrestricted expression
evaluation.

Build validated symbolic expressions from the application's
own parsed representation.


## 11. MODULE F: ABSTRACT SYNTAX TREE

Implement explicit AST node types.

Suggested node types:

- NumberNode
- VariableNode
- UnaryNode
- BinaryNode
- EquationNode

Use immutable data structures where practical.

Each node must represent mathematical meaning, not merely
a display string.

The AST must preserve the distinction between:

- A variable and a multiplication symbol.
- An equation and an arithmetic expression.
- A negative number and binary subtraction.
- A grouped expression and its contents.

Provide:

- AST validation.
- AST-to-readable-expression formatting.
- AST-to-SymPy conversion.
- AST inspection for debugging.
- Structural equality or equivalent comparison where useful.

Keep rendering concerns separate from the AST representation.

Write unit tests for precedence, parentheses, unary signs,
implicit multiplication, and invalid syntax.


## 12. MODULE G: SYMBOLIC SOLVER

Implement a rule-based solver for linear equations in one
variable.

Use exact arithmetic where possible.

Prefer rational representations over floating-point arithmetic
when manipulating fractions and rational coefficients.

The solver must:

1. Validate the equation AST.
2. Identify the variable.
3. Reject unsupported nonlinear structures.
4. Normalize each side into an appropriate linear form.
5. Record valid transformations.
6. Isolate the variable when a unique solution exists.
7. Detect inconsistent equations.
8. Detect identities with infinitely many solutions.
9. Handle zero coefficients correctly.
10. Return structured results.

Implement a custom transformation engine.

SymPy may be used to:

- Convert validated AST expressions into symbolic expressions.
- Simplify expressions.
- Verify algebraic equivalence.
- Cross-check candidate solutions.
- Evaluate substitutions.

SymPy must not be treated as a replacement for the custom
pedagogical transformation trace.

Every operation must satisfy its mathematical preconditions.

For example, division by a coefficient is only permitted after
confirming that the coefficient is nonzero.

Do not silently divide by zero.

Do not assume every equation has a unique solution.

If the equation is unsupported or the solver cannot establish
a valid result, return a clear status instead of fabricating
an answer.


## 13. MODULE H: TRANSFORMATION TRACE

The solver must generate a trace as it executes each operation.

Each trace event should contain:

- Stable rule identifier.
- Step index.
- Mathematical state before the operation.
- Operation parameters.
- Mathematical state after the operation.
- Human-readable mathematical reason.
- Relevant preconditions.
- Validation result.
- Optional diagnostic metadata.

Example:

{
  "step_index": 1,
  "rule_id": "subtract_constant_from_both_sides",
  "before": "2*x + 5 = 15",
  "operation": {
    "type": "subtract",
    "value": "5",
    "scope": "both_sides"
  },
  "after": "2*x = 10",
  "reason_code": "isolate_variable_term",
  "validated": true
}

This is an illustrative contract. Use appropriate typed models
in the actual implementation.

Trace events must be generated by the solver itself.

Do not reconstruct the trace afterward by guessing which
steps might have led to the final answer.

Every transformation must be independently testable.

Use canonical internal representations for mathematical state.
Generate display strings through a dedicated formatter.

Do not rely on fragile string replacement to manipulate
mathematical expressions.


## 14. MODULE I: SOLUTION VALIDATOR

Implement a separate validation component.

It must verify:

1. The solution status is consistent with the equation.
2. A unique candidate solution satisfies the original equation.
3. Exact arithmetic is preserved where applicable.
4. Each trace step is mathematically valid.
5. The final state corresponds to the recorded solution.
6. Division and other domain-sensitive operations satisfy
   their required preconditions.

Use substitution and symbolic equivalence checks where
appropriate.

For rational solutions, avoid unnecessary conversion to
floating-point values.

For infinitely many solutions, verify that the equation
reduces to an identity over the supported domain.

For inconsistent equations, verify that the two sides cannot
be equal under the supported assumptions.

Validation failures must prevent an unverified result from
being presented as a successful solution.


## 15. MODULE J: INDONESIAN EXPLANATION ENGINE

Create a deterministic, template-based explanation engine.

Do not use an LLM, generative text API, or a collection of
hardcoded explanations tied to specific example equations.

Templates must be based on rule identifiers and structured
trace parameters.

Supported explanation elements should include:

- Identification of the problem type.
- Explanation of the operation performed.
- Mathematical equation before and after the operation.
- Reason for the transformation.
- Conditions necessary for the operation.
- Final solution summary.

Example:

"Kurangi kedua ruas dengan 5 untuk menghilangkan konstanta
pada ruas kiri."

Then display:

2x + 5 = 15
2x = 10

Then explain:

"Bagi kedua ruas dengan 2 untuk mengisolasi variabel x."

Display:

2x = 10
x = 5

Use natural, concise Indonesian appropriate for students.

Keep mathematical notation separate from explanatory prose.

Never describe a transformation that does not exist in the
validated trace.

If a rule has no explanation template, report a missing
template during development rather than inventing prose.

Make explanation templates easy to extend and test.

Do not duplicate solver logic in the explanation module.


## 16. PIPELINE ORCHESTRATOR

Create a central orchestration component that connects
the modules without merging their responsibilities.

The pipeline should expose separate operations for:

- Processing an uploaded image.
- Preprocessing an image.
- Running OCR.
- Reconstructing an expression.
- Parsing a text expression.
- Solving a validated equation.
- Validating a solution.
- Generating explanations.
- Returning the final result.

Provide a direct text-input workflow independent of OCR.

The text-input workflow is the primary way to test the
solver before the visual pipeline is fully implemented.

Use explicit status values, such as:

- success
- needs_review
- unsupported
- failed

Use typed result objects and meaningful error categories.

Avoid catching all exceptions and silently returning an
empty result.

Catch expected errors at appropriate boundaries.

Log diagnostic information without exposing internal stack
traces to ordinary users.


## 17. USER INTERFACE

Build a polished local application using Streamlit.

The visual design should resemble a modern mathematical
workbench rather than a generic chatbot.

Design direction:

- Dark, professional, technical interface.
- Neutral charcoal or near-black background.
- Off-white text for strong readability.
- Muted gray for secondary information.
- One restrained accent color for interactive states.
- Clear hierarchy between the original problem, parsed
  expression, working steps, and final answer.
- Consistent spacing and typography.
- Responsive layout suitable for desktop and laptop screens.
- Avoid excessive gradients, oversized decorative cards,
  unnecessary animations, and visual clutter.

Primary interface sections:

A. Header
   - MathVSE name.
   - Short description.
   - Local processing indicator.
   - Current application status.

B. Input panel
   - Image uploader.
   - Text expression input.
   - Sample equation selector.
   - Clear/reset action.

C. Image analysis panel
   - Original image preview.
   - Preprocessed image preview.
   - Bounding-box visualization.
   - Detected symbols and confidence indicators.

D. OCR correction panel
   - Editable recognized expression.
   - Clear indication that OCR output may be incorrect.
   - Explicit confirmation before solving ambiguous output.

E. Mathematical solution panel
   - Parsed expression.
   - Problem classification.
   - Ordered solution steps.
   - Equation before and after each step.
   - Explanation in Indonesian.
   - Final solution status.

F. Validation panel
   - Solution verification status.
   - Warnings and limitations.
   - Diagnostic details when appropriate.

G. Settings/help panel
   - Preprocessing strategy.
   - Available model status.
   - Supported equation types.
   - Local/offline operation information.

Use accessible labels and helpful empty states.

The UI must distinguish:

- A successfully solved equation.
- An equation with infinitely many solutions.
- An inconsistent equation.
- An unsupported expression.
- An OCR result requiring correction.
- An internal processing failure.

Do not show a successful green result merely because a
button was clicked.

Do not fabricate OCR predictions, confidence scores,
processing times, or model availability.

## 18. ERROR HANDLING AND SECURITY

Handle:

- Invalid image formats.
- Corrupt images.
- Images exceeding configured limits.
- Empty expressions.
- Invalid tokens.
- Invalid mathematical syntax.
- Unsupported variables or operators.
- Division by zero.
- Ambiguous OCR output.
- Missing model weights.
- Missing explanation templates.
- Solver validation failures.
- Unexpected internal exceptions.

Do not use eval() or exec() on user input.

Do not execute uploaded files.

Do not load arbitrary model files from untrusted locations.

Avoid unbounded image-processing loops or resource-intensive
operations on uncontrolled input.

Keep errors informative but avoid exposing sensitive internal
implementation details in the UI.

Do not use external network calls in the processing pipeline.

## 19. PROJECT STRUCTURE

Use a clean structure similar to:

mathvse/
├── app.py
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   └── default.yaml
├── assets/
│   └── README.md
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── models/
│   └── README.md
├── src/
│   └── mathvse/
│       ├── __init__.py
│       ├── config.py
│       ├── exceptions.py
│       ├── logging_config.py
│       ├── contracts/
│       │   ├── __init__.py
│       │   ├── image.py
│       │   ├── recognition.py
│       │   ├── expression.py
│       │   ├── solution.py
│       │   └── trace.py
│       ├── vision/
│       │   ├── preprocessing.py
│       │   ├── segmentation.py
│       │   └── layout.py
│       ├── recognition/
│       │   ├── classifier.py
│       │   ├── cnn.py
│       │   ├── dataset.py
│       │   ├── inference.py
│       │   └── training.py
│       ├── spatial/
│       │   ├── models.py
│       │   ├── grouping.py
│       │   └── expression_builder.py
│       ├── math_engine/
│       │   ├── tokens.py
│       │   ├── lexer.py
│       │   ├── parser.py
│       │   ├── ast_nodes.py
│       │   ├── formatter.py
│       │   ├── sympy_adapter.py
│       │   ├── rules.py
│       │   ├── solver.py
│       │   └── validator.py
│       ├── explanation/
│       │   ├── templates.py
│       │   └── generator.py
│       ├── pipeline/
│       │   └── orchestrator.py
│       └── ui/
│           ├── components.py
│           └── styles.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/

You may adjust the structure if there is a clear technical
reason.

Avoid creating empty modules merely to imitate this tree.

Every implemented module must have a defined responsibility.

Keep generated datasets, trained model weights, caches, and
temporary images out of version control unless explicitly
required and appropriately licensed.

## 20. TESTING REQUIREMENTS

Write tests throughout development.

Do not postpone all tests until the end.

A. Lexer and parser tests
- Operator precedence.
- Parentheses.
- Unary minus.
- Implicit multiplication.
- Invalid syntax.
- Multiple equality operators.
- Unsupported characters.

B. Solver tests
- Unique solutions.
- Negative solutions.
- Fractional solutions.
- Zero coefficients.
- No solution.
- Infinite solutions.
- Equivalent equations.
- Unsupported nonlinear expressions.

C. Trace tests
- Correct rule identifiers.
- Correct before/after states.
- Valid operation preconditions.
- Correct step order.
- No unsupported transformation.
- Final state matches the reported result.

D. Validation tests
- Correct solutions pass.
- Incorrect solutions fail.
- Identity equations are classified correctly.
- Inconsistent equations are classified correctly.

E. Image-processing tests
- Valid images.
- Empty files.
- Corrupt images.
- Low contrast.
- Noise.
- Bounding-box coordinates.
- Preprocessing failure handling.

F. OCR tests
- Class-to-label mapping.
- Model loading failures.
- Deterministic inference configuration.
- Confidence and candidate handling.
- Unknown or ambiguous symbols.

G. Integration tests
- Text input -> parser -> solver -> validator -> explanation.
- Image input -> preprocessing -> OCR -> review state.
- Corrected OCR -> solver -> validated explanation.
- Unsupported input -> honest failure state.

Use fixtures and fixed seeds where appropriate.

Do not write tests that merely assert True or check that a
function exists.

Test actual mathematical behavior and output structures.

## 21. DOCUMENTATION

Write a useful README.md in Indonesian.

Include:

- Project overview.
- System architecture.
- Supported mathematics.
- Unsupported features.
- Prerequisites.
- Installation instructions.
- Virtual environment setup.
- Dependency installation.
- Application startup.
- Test execution.
- Optional CNN training.
- Dataset preparation.
- Model file placement.
- Offline operation.
- Known limitations.
- Troubleshooting.

Provide exact commands for the chosen operating systems
where practical.

Document whether a step requires internet access during
installation or can run offline afterward.

Do not claim the project works offline until all required
runtime dependencies and model assets are locally available.

Do not claim that the CNN is ready for real-world OCR unless
it has been trained and evaluated.

## 22. DEVELOPMENT ROADMAP

Implement the project in the following phases.

PHASE 1 — MATHEMATICAL FOUNDATION

Implement:
- Expression contracts.
- Lexer.
- Parser.
- AST.
- AST formatter.
- Rule-based linear solver.
- Transformation trace.
- Solution validator.
- Indonesian explanation templates.
- Unit tests.

The text-based solver must work before image recognition
becomes a required dependency.

PHASE 2 — IMAGE PROCESSING

Implement:
- Image validation.
- Grayscale conversion.
- Thresholding alternatives.
- Denoising.
- Candidate region detection.
- Bounding-box visualization.
- Preprocessing tests.

PHASE 3 — OCR BASELINE

Implement:
- Classifier interface.
- Small CNN architecture.
- Dataset preparation pipeline.
- Training/evaluation scripts.
- Model persistence.
- CPU inference.
- Confidence reporting.
- Missing-model handling.

PHASE 4 — SPATIAL PARSING

Implement:
- Symbol geometry models.
- Grouping heuristics.
- Main-line reconstruction.
- Simple superscript support when validated.
- Simple fraction support when validated.
- Ambiguity detection.
- OCR correction workflow.

PHASE 5 — END-TO-END INTEGRATION

Connect:
- Image input.
- Preprocessing.
- OCR.
- Spatial parsing.
- Expression validation.
- Symbolic solving.
- Transformation trace.
- Solution verification.
- Explanation generation.

PHASE 6 — GUI AND QUALITY

Implement:
- Final Streamlit interface.
- Image and expression preview.
- Step-by-step results.
- Validation indicators.
- Error and empty states.
- Documentation.
- Full test suite.
- Performance checks.

Do not start by attempting to train a comprehensive OCR model
before the mathematical engine and data contracts are stable.

## 23. PERFORMANCE REQUIREMENTS

The initial target is a conventional CPU-based laptop with
approximately four CPU cores and 8 GB RAM.

Keep inference and symbolic computation lightweight.

Avoid loading large datasets into memory unnecessarily.

Separate training dependencies and runtime dependencies
when practical.

Do not require a GPU for the text solver, image preprocessing,
or basic application startup.

Use lazy loading for optional ML models when appropriate.

Do not claim a latency target without measuring it.

Record benchmark results on the actual test environment.

## 24. DEFINITION OF DONE

The MVP is complete only when:

1. The application starts using documented instructions.
2. Text-based equation solving works.
3. Supported equations produce valid solutions.
4. No-solution and infinite-solution cases are handled.
5. The transformation trace is generated by actual solver rules.
6. Explanation templates use that trace.
7. Candidate solutions are validated.
8. Image preprocessing works on supported image inputs.
9. OCR output can be inspected and corrected.
10. The absence of trained weights is handled honestly.
11. Unsupported expressions fail safely.
12. Unit and integration tests run successfully.
13. The README documents actual implemented behavior.
14. The application does not require an LLM or remote API.
15. No core feature is represented by fake output.

If OCR is still a baseline or requires additional training,
document that limitation clearly rather than claiming full
handwritten-mathematics recognition.

## 25. EXECUTION INSTRUCTIONS FOR THE CODING AGENT

Work incrementally.

STEP 1:
Inspect the repository, available Python version, dependency
manager, and existing project files.

STEP 2:
Present a concise implementation plan and proposed directory
structure. Identify risks that materially affect the design.

STEP 3:
Implement Phase 1 first.

STEP 4:
Run the tests and fix actual failures before proceeding.

STEP 5:
Implement image preprocessing and the OCR interfaces.

STEP 6:
Integrate spatial reconstruction only after the contracts
between vision and recognition are established.

STEP 7:
Build the interface around the real application pipeline.

STEP 8:
Run the complete test suite and perform an end-to-end check.

STEP 9:
Update documentation to reflect actual behavior.

STEP 10:
Report the final implementation status.

At each phase, report:

- Files created or modified.
- Functionality implemented.
- Tests executed.
- Actual test results.
- Known limitations.
- Remaining tasks.

Do not fabricate successful test results.

Do not replace real implementation with TODO comments,
hardcoded answers, random predictions, or simulated OCR output.

If a dependency or dataset is unavailable, implement what can
be completed without it and clearly identify the limitation.

Do not silently change the project's architectural goals.

The ultimate objective is a reliable, modular, locally
executed mathematical solver whose results and explanations
can be audited.

Begin by inspecting the repository and implementing Phase 1.