from enum import Enum
from pathlib import Path
from urllib.parse import urlparse


class BrowserType(Enum):
    chromium = "chrome"
    firefox = "firefox"


def normalize_url(url: str) -> str:
    """Add http:// protocol if missing"""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def get_filename_from_url(url: str) -> str:
    """Generate a safe filename from URL"""
    parsed = urlparse(url)
    filename = parsed.netloc.replace("www.", "").replace(".", "_")
    if parsed.path and parsed.path != "/":
        filename += "_" + parsed.path.strip("/").replace("/", "_")[:50]
    return filename + ".png"


def parse_textfile(file: Path) -> list[str]:
    with open(file, "r") as f:
        return f.read().splitlines()


def parse_csv(file: Path) -> list[str]:
    pass
