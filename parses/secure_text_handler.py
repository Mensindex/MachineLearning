from typing import Dict

from parses.parses_body import ParsesBody
from parses.parses_cookies import ParsesCookies


class SecureTextHandler(ParsesBody, ParsesCookies):
    def __init__(self, request: Dict[str, object]):
        self.request = request

    def process(self):
        if self.is_authed():
            return len(self.body())
        else:
            return None
