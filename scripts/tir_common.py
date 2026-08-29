"""TIR 하네스의 **가벼운 공통부** — 프롬프트 상수, 답 추출, 코드 격리 실행.

`tir_inference.py` 는 HF 추론용이라 `torch` / `transformers` 를 import 한다. 그런데 vLLM
클라이언트들이 프롬프트 상수를 거기서 가져오는 바람에, `/workspace` 가 네트워크 마운트인
이 pod 에서 **클라이언트 하나 띄울 때마다 2~3분을 import 에만 썼다.**

그래서 상수와 순수 함수를 여기로 **이동**했다(복붙 아님 — 단일 출처).
`tir_inference.py` 는 이 모듈에서 re-export 하므로 기존 import 경로가 전부 그대로 동작한다.
CONTEXT 의 "프롬프트 상수는 복붙 금지(학습/추론 불일치 방지)" 원칙을 지킨다.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


TIR_SYSTEM = (
    "You are a meticulous contest mathematician with a Python interpreter available.\n"
    "Solve the problem by writing Python code that computes the answer.\n"
    "Prefer brute-force enumeration, exhaustive case checking, and sympy over hand algebra — "
    "the interpreter is exact and fast, so verify rather than guess.\n"
    "Write exactly one Python code block in this format:\n"
    "```python\n"
    "# your code; it MUST print the result\n"
    "print(result)\n"
    "```\n"
    "Then stop and wait. You will be shown the program output, and only then give the final answer.\n"
    "The answer is always an integer. End your final response with exactly: Final answer: <integer>"
)

# 대수 특화 변형 (2026-08-29): 잔여 오답의 48%가 대수 — 숫자 브루트포스가 아니라
# 기호 연산(solve/simplify)이 필요한 영역이라 sympy 사용을 명시적으로 강제한다.
# 출력 규약(단일 코드블록, print, Final answer)은 기존과 동일해 하네스가 그대로 돈다.
TIR_SYMPY_SYSTEM = (
    "You are a meticulous contest mathematician with a Python interpreter and the sympy library.\n"
    "Solve the problem with EXACT symbolic computation using sympy: define unknowns with symbols(), "
    "set up equations with Eq(), and use solve(), simplify(), expand(), factor(), Rational(), "
    "summation(), or diff() as needed. Never use floating point — keep every step exact, and only "
    "convert the final exact result to an integer at the end. If symbolic solving stalls, fall back "
    "to exact enumeration.\n"
    "Write exactly one Python code block in this format:\n"
    "```python\n"
    "from sympy import *\n"
    "# your exact symbolic solution; it MUST print the result\n"
    "print(result)\n"
    "```\n"
    "Then stop and wait. You will be shown the program output, and only then give the final answer.\n"
    "The answer is always an integer. End your final response with exactly: Final answer: <integer>"
)

FINAL_NUDGE = (
    "Program output is above. If it answers the problem, state the answer now. "
    "If the output is wrong, empty, or an error, reason it out yourself instead. "
    "End with exactly: Final answer: <integer>"
)

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

# baseline.py와 동일 규약 + Final answer 우선 (오전 세션 20절 검증안)
ANSWER_PATTERNS = (
    re.compile(r"(?:final answer|정답)\s*(?:is|:|=)?\s*[^\d\-]{0,15}?(-?\d[\d,]*)", re.I),
    re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}"),
    re.compile(r"(-?\d[\d,]*)"),
)


def extract_answer(text):
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].replace(",", "")
    return None


def normalize(value):
    """정수로 정규화. inf/nan은 None으로 버린다.

    모델이 생성한 프로그램이 inf를 출력하는 경우가 실제로 있었다(N=8 샘플링에서 발생).
    float('inf')는 파싱에 성공하고 int()에서 OverflowError가 나므로 반드시 함께 잡아야 한다.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except (ValueError, OverflowError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    try:
        return int(f)
    except (OverflowError, ValueError):
        return None


# 모델이 생성한 코드 실행 전에 주입하는 프리앰블.
# 대회 규정상 추론은 전부 로컬이어야 하므로 소켓 자체를 막는다.
# (생성 코드 2,321개 전수 검사에서 네트워크 사용은 0건이었지만, "안 했다"와 "못 한다"는 다르다.)
SANDBOX_PREAMBLE = """import socket as _socket
def _blocked(*a, **k):
    raise OSError("network disabled in sandbox")
_socket.socket = _blocked
_socket.create_connection = _blocked
_socket.socketpair = _blocked
del _socket
"""


def run_code(code, timeout):
    """모델이 생성한 코드를 격리 실행한다.

    별도 프로세스(-I 격리 모드) + 타임아웃 + 임시 작업디렉터리로 부작용을 제한하고,
    프리앰블로 소켓을 막아 네트워크 접근을 차단한다(대회 규정: 추론은 전부 로컬).
    """
    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "solve.py"
        script.write_text(SANDBOX_PREAMBLE + code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": f"timed out after {timeout}s"}
        status = "ok" if proc.returncode == 0 else "error"
        return {
            "status": status,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-1000:],
        }


def generate(model, tokenizer, prompts, max_new_tokens):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(out[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def read_rows(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in open(path)]
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


