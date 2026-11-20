"""Interactive relay login; store refreshed cookies without printing credentials."""
import argparse
import getpass
import http.cookiejar
import json
import os
import urllib.error
import urllib.request

from relay_api import CACHE, relay_base


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        base = relay_base()
        name = input("队内用户名: ").strip()
        password = getpass.getpass("队内密码（隐藏输入）: ")
    except EOFError:
        raise SystemExit("请在交互式终端运行此脚本以输入队内登录信息。") from None
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    jar = http.cookiejar.MozillaCookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    payload = json.dumps({"name": name, "password": password}).encode()
    req = urllib.request.Request(
        base + "/api/login", data=payload,
        headers={"X-Relay-Request": "1", "Content-Type": "application/json"},
    )
    try:
        with opener.open(req, timeout=45) as response:
            result = json.load(response)
        if result.get("ok") is not True:
            raise RuntimeError("登录未成功；没有更新本地会话。")
        req = urllib.request.Request(base + "/api/me", headers={"X-Relay-Request": "1"})
        with opener.open(req, timeout=45) as response:
            profile = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"登录或验证失败（HTTP {error.code}）；没有更新本地会话。") from None
    finally:
        password = None
        payload = None
    if not any(cookie.name == "relay_session" for cookie in jar):
        raise SystemExit("服务器没有返回队内会话 Cookie；没有更新本地文件。")
    CACHE.mkdir(parents=True, exist_ok=True)
    temp = CACHE / f".cookies-login-{os.getpid()}.tmp"
    try:
        # Create with private permissions before writing any cookie content.
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        jar.save(str(temp), ignore_discard=True, ignore_expires=True)
        os.replace(temp, CACHE / "cookies.txt")
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({
        "登录成功": True,
        "账号": (profile.get("user") or {}).get("name"),
        "会话文件": str(CACHE / "cookies.txt"),
        "队列状态": profile.get("counts"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
