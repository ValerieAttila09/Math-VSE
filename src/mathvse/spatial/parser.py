from __future__ import annotations

from mathvse.contracts.recognition import SymbolPrediction
from mathvse.contracts.spatial import SpatialParseResult, SpatialSymbol


def _normal_symbol_value(label: str) -> str:
    return {
        "times": "*",
        "multiply": "*",
        "plus": "+",
        "minus": "-",
        "equals": "=",
        "divide": "/",
        "left_paren": "(",
        "right_paren": ")",
    }.get(label.lower(), label)


def _is_baseline_layout(symbols: list[SpatialSymbol]) -> bool:
    if not symbols:
        return False
    ys = [bbox[1] for bbox in (symbol.bbox for symbol in symbols)]
    heights = [bbox[3] for bbox in (symbol.bbox for symbol in symbols)]
    if not heights:
        return False
    if max(ys) - min(ys) > max(heights) * 1.2:
        return False
    return True


def parse_symbols(symbols: list[SymbolPrediction]) -> SpatialParseResult:
    if not symbols:
        return SpatialParseResult(status="unsupported_structure", normalized_text="", symbols=[], warnings=["No symbols were supplied for parsing."])

    recognized = []
    warnings: list[str] = []
    low_confidence_found = False
    for symbol in symbols:
        if symbol.status == "model_unavailable":
            warnings.append("Model unavailable for a symbol; expression is not trusted.")
            continue
        if symbol.status != "recognized":
            warnings.append(f"Symbol {symbol.label} is not recognized with high confidence.")
            continue
        if symbol.confidence < 0.35:
            low_confidence_found = True
            warnings.append(f"Low-confidence symbol {symbol.label!r}; reconstruction may need review.")
            continue
        label = _normal_symbol_value(symbol.label)
        recognized.append(
            SpatialSymbol(label=label, bbox=symbol.bbox, confidence=symbol.confidence, status=symbol.status)
        )

    if not recognized:
        return SpatialParseResult(status="needs_review", normalized_text="", symbols=[], warnings=warnings)
    if low_confidence_found:
        return SpatialParseResult(status="needs_review", normalized_text="", symbols=recognized, warnings=warnings)

    ordered = sorted(recognized, key=lambda item: (item.bbox[1], item.bbox[0]))
    text_parts: list[str] = []
    for symbol in ordered:
        text_parts.append(symbol.label)

    text = "".join(text_parts)
    if not _is_baseline_layout(ordered):
        return SpatialParseResult(
            status="unsupported_structure",
            normalized_text="",
            symbols=ordered,
            warnings=["Detected geometry suggests superscript, subscript, or multilevel structure that is not supported in the baseline parser."],
        )

    for index in range(len(ordered) - 1):
        left = ordered[index]
        right = ordered[index + 1]
        x_overlap = max(0, min(left.bbox[0] + left.bbox[2], right.bbox[0] + right.bbox[2]) - max(left.bbox[0], right.bbox[0]))
        if x_overlap >= max(left.bbox[2], right.bbox[2]) * 0.5:
            return SpatialParseResult(
                status="needs_review",
                normalized_text="",
                symbols=ordered,
                warnings=["Overlapping recognition boxes make the baseline expression ambiguous; review is required."],
            )

    normalized = text
    if "x" in normalized and "*" not in normalized and any(ch.isdigit() for ch in normalized):
        normalized = normalized.replace("2x", "2*x")
        normalized = normalized.replace("3x", "3*x")
        normalized = normalized.replace("4x", "4*x")
        normalized = normalized.replace("5x", "5*x")
        normalized = normalized.replace("6x", "6*x")
        normalized = normalized.replace("7x", "7*x")
        normalized = normalized.replace("8x", "8*x")
        normalized = normalized.replace("9x", "9*x")
        normalized = normalized.replace("x(", "x*(")

    if any(token in normalized for token in ["/", "^", "_", "sqrt"]) and not any(token in normalized for token in ["=", "+", "-", "*", "/"]):
        return SpatialParseResult(
            status="unsupported_structure",
            normalized_text="",
            symbols=ordered,
            warnings=["The current spatial layout is not a supported linear equation baseline."],
        )

    return SpatialParseResult(status="success", normalized_text=normalized, symbols=ordered, warnings=warnings)
