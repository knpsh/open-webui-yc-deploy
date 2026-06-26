import logging
import os
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger("limits")

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
RATELIMIT_ENABLED = os.environ.get("RATELIMIT_ENABLED", "true").lower() == "true"

_redis: Optional[aioredis.Redis] = None

# Atomically INCR a key and set its TTL only on first creation.
# KEYS[1]=key, ARGV[1]=ttl_seconds. Returns the new counter value.
_INCR_TTL_LUA = """
local v = redis.call('INCR', KEYS[1])
if v == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return v
"""

# Atomically INCRBY a key and set its TTL only on first creation.
# KEYS[1]=key, ARGV[1]=ttl_seconds, ARGV[2]=amount. Returns the new value.
_INCRBY_TTL_LUA = """
local v = redis.call('INCRBY', KEYS[1], ARGV[2])
if v == tonumber(ARGV[2]) then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return v
"""

# TTLs: comfortably longer than each window so keys self-expire.
_DAY_TTL = 48 * 3600        # 48h
_MONTH_TTL = 32 * 24 * 3600  # 32d


class LimitError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def get_user(request) -> tuple[Optional[str], Optional[str]]:
    """Return (user_id, role) from forwarded OpenWebUI headers."""
    user_id = request.headers.get("x-openwebui-user-id")
    role = request.headers.get("x-openwebui-user-role")
    return user_id, role


async def _incr(key: str, ttl: int) -> int:
    redis = _get_redis()
    return int(await redis.eval(_INCR_TTL_LUA, 1, key, ttl))


async def enforce(
    user_id: Optional[str],
    role: Optional[str],
    path_key: str,
    daily_limit: int,
    monthly_limit: int,
) -> None:
    """Pre-increment per-user daily/monthly counters for path_key.

    Raises LimitError(403) on missing user id, LimitError(429) when over limit.
    No-op when disabled or for admins.
    """
    if not RATELIMIT_ENABLED:
        return
    if role == "admin":
        return
    if not user_id:
        raise LimitError(403, "User identity required for this request")

    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")
    month = now.strftime("%Y%m")

    day_key = f"ratelimit:{user_id}:{path_key}:day:{day}"
    month_key = f"ratelimit:{user_id}:{path_key}:month:{month}"

    try:
        day_count = await _incr(day_key, _DAY_TTL)
        month_count = await _incr(month_key, _MONTH_TTL)
    except Exception as e:
        # Fail-open on Redis errors so an outage doesn't block all usage.
        logger.error(f"Rate limit backend error ({path_key}): {e}")
        return

    logger.info(
        f"[limit:{path_key}] user={user_id} "
        f"day={day_count}/{daily_limit} month={month_count}/{monthly_limit}"
    )

    if daily_limit > 0 and day_count > daily_limit:
        raise LimitError(
            429,
            f"Daily limit reached for {path_key} "
            f"({daily_limit}/day). Try again tomorrow.",
        )
    if monthly_limit > 0 and month_count > monthly_limit:
        raise LimitError(
            429,
            f"Monthly limit reached for {path_key} "
            f"({monthly_limit}/month). Try again next month.",
        )


async def check_tokens(
    user_id: Optional[str],
    role: Optional[str],
    path_key: str,
    daily_limit: int,
    monthly_limit: int,
) -> None:
    """Read-only check of token budgets before forwarding.

    Raises LimitError(403) on missing user id, LimitError(429) when already
    over budget. No-op when disabled, for admins, or when no limits set.
    """
    if not RATELIMIT_ENABLED:
        return
    if role == "admin":
        return
    if not user_id:
        raise LimitError(403, "User identity required for this request")
    if daily_limit <= 0 and monthly_limit <= 0:
        return

    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")
    month = now.strftime("%Y%m")
    day_key = f"ratelimit:{user_id}:{path_key}:day:{day}"
    month_key = f"ratelimit:{user_id}:{path_key}:month:{month}"

    try:
        redis = _get_redis()
        day_count = int(await redis.get(day_key) or 0)
        month_count = int(await redis.get(month_key) or 0)
    except Exception as e:
        logger.error(f"Token limit check backend error ({path_key}): {e}")
        return

    if daily_limit > 0 and day_count >= daily_limit:
        raise LimitError(
            429,
            f"Daily token limit reached for {path_key} "
            f"({daily_limit} tokens/day). Try again tomorrow.",
        )
    if monthly_limit > 0 and month_count >= monthly_limit:
        raise LimitError(
            429,
            f"Monthly token limit reached for {path_key} "
            f"({monthly_limit} tokens/month). Try again next month.",
        )


async def add_tokens(user_id: Optional[str], path_key: str, tokens: int) -> None:
    """Add `tokens` to the user's daily/monthly token counters. Fail-open."""
    if not RATELIMIT_ENABLED or not user_id or tokens <= 0:
        return
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")
    month = now.strftime("%Y%m")
    day_key = f"ratelimit:{user_id}:{path_key}:day:{day}"
    month_key = f"ratelimit:{user_id}:{path_key}:month:{month}"
    try:
        redis = _get_redis()
        day_count = int(
            await redis.eval(_INCRBY_TTL_LUA, 1, day_key, _DAY_TTL, tokens)
        )
        month_count = int(
            await redis.eval(_INCRBY_TTL_LUA, 1, month_key, _MONTH_TTL, tokens)
        )
        logger.info(
            f"[tokens:{path_key}] user={user_id} +{tokens} "
            f"day={day_count} month={month_count}"
        )
    except Exception as e:
        logger.error(f"Token limit add backend error ({path_key}): {e}")


def limits_for(path_key: str) -> tuple[int, int]:
    """Read (daily, monthly) limits for a path_key from env. 0 = unlimited."""
    key = path_key.upper()
    daily = int(os.environ.get(f"RATELIMIT_{key}_PER_DAY", "0") or "0")
    monthly = int(os.environ.get(f"RATELIMIT_{key}_PER_MONTH", "0") or "0")
    return daily, monthly
