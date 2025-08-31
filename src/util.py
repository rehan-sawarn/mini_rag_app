import time, uuid


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str = "doc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"