"""Rule metadata used by the linear solver."""

RULES = {
    "normalize_linear_terms": {
        "reason_code": "collect_like_terms",
        "description": "Mengumpulkan suku yang mengandung variabel dan konstanta.",
        "preconditions": ["equation_is_linear_in_one_variable"],
    },
    "divide_by_coefficient": {
        "reason_code": "isolate_variable",
        "description": "Membagi kedua sisi dengan koefisien variabel agar x terisolasi.",
        "preconditions": ["coefficient_is_nonzero"],
    },
}
