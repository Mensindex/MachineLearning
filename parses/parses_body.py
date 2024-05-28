from typing import Dict


class ParsesBody:
    request = {
        "cookies": {"key_1": "value_1", "auth_key": "value_2", },
        "body": "a long time ago, in a Galaxy far, far away",
        "headers": {"content-type": "application/json", "Accept": "application/json"}
    }

    def body(self):
        return self.request.get("body")
