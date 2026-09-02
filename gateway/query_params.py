"""Shared strict parsing for compatibility HTTP query parameters."""

_TRUE_QUERY_BOOL_TOKENS = frozenset({"1", "true", "yes", "on", "t", "y"})
_FALSE_QUERY_BOOL_TOKENS = frozenset({"0", "false", "no", "off", "f", "n"})


def parse_query_bool(value: str | None, *, default: bool = False) -> bool:
    """Parse the exact legacy-compatible boolean token set.

    Tokens are case-insensitive and surrounding ASCII/Unicode whitespace is
    ignored. Accepted true tokens are ``1,true,yes,on,t,y``; accepted false
    tokens are ``0,false,no,off,f,n``. Omission uses ``default``. Any explicit
    token outside that set, including an empty token, is rejected.
    """
    if value is None:
        return default
    token = value.strip().lower()
    if token in _TRUE_QUERY_BOOL_TOKENS:
        return True
    if token in _FALSE_QUERY_BOOL_TOKENS:
        return False
    raise ValueError("invalid boolean query token")
