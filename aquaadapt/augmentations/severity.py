"""Severity-level validation."""


def validate_severity(severity: int) -> int:
    value = int(severity)
    if value not in {0, 1, 2, 3}:
        raise ValueError(f"Severity must be one of 0, 1, 2, 3; got {severity}")
    return value

