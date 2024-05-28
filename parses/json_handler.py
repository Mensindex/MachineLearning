import json as js
from typing import Dict

from parses.parses_body import ParsesBody
from parses.parses_headers import ParsesHeaders


class JsonHandler(ParsesBody, ParsesHeaders):
    def __init__(self, request: Dict[str, object]):
        self.request = request

    def process(self):
        if self.need_json():
            if js.loads(self.body()) is None:
                return None
            else:
                data: Dict[str, object] = js.loads(self.body())
                return len(data.keys())
        else:
            return None
