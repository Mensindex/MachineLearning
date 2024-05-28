from typing import Dict


class ParsesCookies:
    request = {
        "cookies": {"key_1": "value_1", "auth_key": "value_2", },
        "body": "a long time ago, in a Galaxy far, far away",
        "headers": {"content-type": "application/json", "Accept": "application/json"}
    }

    def cookies(self):
        return self.request.get("cookies")

    def is_authed(self):
        return self.cookies().__contains__("auth_key")
