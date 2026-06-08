from enum import Enum
from pathlib import Path
from urllib.parse import urlparse


class BrowserType(Enum):
    """
    Enumeration representing different browser types used in web automation or testing.

    This class defines a set of browser types that are supported for automation or testing purposes.
    Each browser type is represented as a member of the enumeration, with its corresponding string
    value used for identification.

    Attributes:
        chromium (str): Represents the Chromium-based browsers.
        firefox (str): Represents the Mozilla Firefox browser.
    """

    chromium = "chrome"
    firefox = "firefox"


def normalize_url(url: str) -> str:
    """
    Normalizes a given URL by ensuring it includes a proper scheme.

    This function takes a URL as input and ensures it starts with either
    "http://" or "https://". If the scheme is missing, "https://" is
    prefixed to the URL before returning it.

    Args:
        url (str): The input URL to normalize.

    Returns:
        str: The normalized URL with a valid scheme.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def get_filename_from_url(url: str) -> str:
    """
    Generate a filename based on a given URL.

    The function takes a URL as input, parses it, and constructs a
    filename by replacing certain characters in the domain and path.
    It ensures the generated filename is valid and appends a `.png`
    extension. The path component, if present, is truncated to a
    maximum of 50 characters to maintain concise filenames.
    The generated filename can be used to uniquely reference content
    associated with the provided URL.

    Parameters:
        url (str): The URL from which the filename is to be derived.

    Returns:
        str: A string representing the generated filename.
    """
    parsed = urlparse(url)
    filename = parsed.netloc.replace("www.", "").replace(".", "_")
    if parsed.path and parsed.path != "/":
        filename += "_" + parsed.path.strip("/").replace("/", "_")[:50]
    return filename + ".png"


def parse_textfile(file: Path) -> list[str]:
    """
    Parses a text file and returns a list of lines.

    This function opens a text file, reads its contents, and splits the text into
    lines which are returned as elements of a list. It is intended to handle text
    data and assumes the file exists and is accessible at the provided path.

    Parameters:
    file : Path
        A Path object representing the file to be read.

    Returns:
    list[str]
        A list of strings, where each string corresponds to a line in the text
        file.

    Raises:
    FileNotFoundError
        If the file does not exist at the given path.
    PermissionError
        If the file cannot be accessed due to insufficient permissions.
    """
    with open(file, "r") as f:
        return f.read().splitlines()
