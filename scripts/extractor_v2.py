"""답 추출기 v1(현행 baseline.py) / v2(2026-08-27 감사 수정안) 를 나란히 제공한다.

v1의 결함 (CONTEXT 2026-08-27 오전 20절):
  1. `\\boxed{}` 가 `Final answer:` 보다 먼저 검사된다. 모델이 중간 단계에서 boxed 를
     쓰고 마지막에 Final answer 로 정정하면 중간값을 집는다.
  2. `Final answer: \\(-7\\)` 처럼 숫자가 LaTeX 래퍼에 싸이면 2번 패턴이 실패해
     "마지막 숫자" fallback 으로 떨어진다. hybrid_3145_verify 는 831 중 44개가 이 경로다.

v2는 `Final answer` 최우선 + 숫자 앞 비숫자 15자 허용. tir_inference.py 가 이미 쓰는
검증된 구현과 문자 단위로 동일하다.

홀드아웃 검증(464문제, 5개 독립 예측 파일): 개선 4 / 악화 0.
"""
import re

ANSWER_PATTERNS_V1 = (
    re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}"),
    re.compile(r"(?:final answer|answer|정답)\s*(?:is|:|=)?\s*(-?\d[\d,]*)", re.I),
    re.compile(r"(-?\d[\d,]*)"),
)

ANSWER_PATTERNS_V2 = (
    re.compile(r"(?:final answer|정답)\s*(?:is|:|=)?\s*[^\d\-]{0,15}?(-?\d[\d,]*)", re.I),
    re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}"),
    re.compile(r"(-?\d[\d,]*)"),
)


def _extract(text, patterns):
    if not text:
        return None
    for pattern in patterns:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].replace(",", "")
    return None


def extract_v1(text):
    return _extract(text, ANSWER_PATTERNS_V1)


def extract_v2(text):
    return _extract(text, ANSWER_PATTERNS_V2)


def norm(value):
    """정수 정규화. inf/nan/비정수는 None."""
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s or s.lower() in {"none", "inf", "-inf", "nan"}:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    try:
        return int(f)
    except (OverflowError, ValueError):
        return None
