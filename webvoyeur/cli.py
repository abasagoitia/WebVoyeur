import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from typer import BadParameter, Context, Option, Typer, echo

from webvoyeur.nmap_parser import NmapParser
from webvoyeur.peeker import Peeker
from webvoyeur.utilities import BrowserType, parse_textfile


class LogLevel(str, Enum):
    """
    Enumeration for different log levels.

    This class defines various levels of logging as enumeration values.
    It extends the `Enum` class and uses the string data type to represent
    each log level. These levels can be utilized in applications to categorize
    and manage logs based on their severity or purpose.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Config:
    """
    Configuration class for managing application settings.

    This class encapsulates various configuration options used throughout the
    application. It provides default settings and allows customization of
    parameters related to output directories, browser preferences, execution timeouts,
    worker limits, screen dimensions, and logging levels.

    Attributes:
        output_dir (Path): The directory where output files will be saved.
        browser (BrowserType): The type of browser to be used (e.g., Firefox, Chrome).
        timeout (int): The maximum time, in seconds, to wait for operations.
        normalize_urls (bool): Whether URLs should be normalized.
        max_workers (int): The maximum number of worker threads or processes.
        width (int): The width of the browser window in pixels.
        height (int): The height of the browser window in pixels.
        log_level (LogLevel | int): The logging level for the application.
    """

    output_dir: Path = Path("./output")
    browser: BrowserType = BrowserType.firefox
    timeout: int = 10
    normalize_urls: bool = True
    max_workers: int = 4
    width: int = 1280
    height: int = 720
    ignore_https_errors: bool = True
    log_level: LogLevel | int = LogLevel.INFO


CONFIG = Config()

app: Typer = Typer(
    help="Capture webpages as PNG screenshots with rendered CSS and JavaScript",
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.callback()
def setup_cb(
        ctx: Context,
        log_level: Annotated[LogLevel, Option("--log-level", "-l")] = CONFIG.log_level,
        output_dir: Annotated[
            Path, Option("--output", "-o", dir_okay=True, file_okay=False, exists=False)
        ] = CONFIG.output_dir,
        normalize_urls: Annotated[
            bool, Option("--no-normalize", help="Don't auto-add https:// to URL without protocol")
        ] = CONFIG.normalize_urls,
        max_workers: Annotated[
            int, Option("--max-workers", "-j", help="Maximum number of concurrent capture operations")
        ] = CONFIG.max_workers,
        width: Annotated[int, Option("--width", "-w", help="Viewport width in pixels")] = CONFIG.width,
        height: Annotated[
            int, Option("--height", "-h", help="Viewport height in pixels")
        ] = CONFIG.height,
        ignore_https_errors: Annotated[
            bool,
            Option(
                "--ignore-https-errors/--strict-tls",
                help="Ignore HTTPS certificate errors, or use --strict-tls to validate certificates",
            ),
        ] = CONFIG.ignore_https_errors,
        browser: Annotated[
            BrowserType, Option("--browser", "-b", help="Browser: chrome or firefox")
        ] = CONFIG.browser,
):
    """
    Configures the application based on the provided context and command-line options.

    This function sets up the global configuration by overriding default values with
    specified command-line arguments. It also maps the selected log level to the
    appropriate logging configuration.

    Parameters:
    ctx (Context): Command-line context object used to process options and arguments.

    log_level (LogLevel, optional): Specifies the logging level for the output. Default
    value is derived from CONFIG.log_level.

    output_dir (Path, optional): Directory for the output files. Only directories are
    allowed, and it cannot already exist. Default value is derived from CONFIG.output_dir.

    normalize_urls (bool, optional): Determines whether to automatically add "https://" to
    URLs that do not specify a protocol. Defaults to the value in CONFIG.normalize_urls.

    max_workers (int, optional): Number of concurrent capture operations allowed. Defaults
    to the value in CONFIG.max_workers.

    width (int, optional): Width of the browser’s viewport in pixels. Defaults to the value
    in CONFIG.width.

    height (int, optional): Height of the browser’s viewport in pixels. Defaults to the
    value in CONFIG.height.

    browser (BrowserType, optional): Specifies the browser type to be used, such as
    "chrome" or "firefox". Defaults to the value in CONFIG.browser.

    Returns:
    None
    """
    global CONFIG
    level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
    }
    CONFIG = Config(
        output_dir=output_dir,
        browser=browser,
        timeout=10,
        normalize_urls=normalize_urls,
        max_workers=max_workers,
        width=width,
        height=height,
        ignore_https_errors=ignore_https_errors,
        log_level=level_map[log_level],
    )


@app.command()
def single(
        url: Annotated[str, Option("--url", "-u", help="URL to capture")],
        filename: Annotated[
            Path,
            Option(
                "--filename",
                "-f",
                dir_okay=False,
                file_okay=True,
                exists=False,
                help="Filepath to save output file",
            ),
        ] = None,
        wait_time: Annotated[
            int, Option("--wait-time", "-t", help="Seconds to wait before capturing screenshot")
        ] = 2,
        scroll: Annotated[bool, Option("--scroll", "-s", help="Capture full scrollable page")] = False,
):
    """
    Capture a screenshot of a single webpage and save it to a specified file.

    This function uses a Peeker instance to capture a screenshot of the provided
    URL and saves it to the specified filepath. Additional options include waiting
    for a specified time before taking the screenshot and capturing the full
    scrollable page if needed.

    Attributes:
        url (str): URL of the webpage to capture.
        filename (Path, optional): Path to save the captured screenshot.
        wait_time (int, optional): Number of seconds to wait before taking the
            screenshot. Defaults to 2 seconds.
        scroll (bool, optional): Whether to capture the entire scrollable page.
            Defaults to False.

    Args:
        url: URL to capture.
        filename: Filepath to save output file. Optional.
        wait_time: Seconds to wait before capturing screenshot. Optional.
        scroll: Whether to capture the full scrollable page. Optional.

    Returns:
        None
    """
    with Peeker(
            output_dir=CONFIG.output_dir,
            browser=CONFIG.browser,
            timeout=CONFIG.timeout,
            normalize_urls=CONFIG.normalize_urls,
            max_workers=CONFIG.max_workers,
            width=CONFIG.width,
            height=CONFIG.height,
            ignore_https_errors=CONFIG.ignore_https_errors,
            log_level=CONFIG.log_level,
    ) as peeker:
        output = peeker.capture_single(url, filename=filename, wait_time=wait_time, scroll=scroll)

    echo(f"Screenshot saved to {output}")


@app.command()
def batch(
        urls_txt: Annotated[
            Path | None,
            Option(
                "--urls_txt",
                help="File containing URLs to capture",
                file_okay=True,
                dir_okay=False,
                exists=True,
            ),
        ] = None,
        urls_csv: Annotated[
            Path | None,
            Option(
                "--urls_csv",
                help="File containing URLs to capture (CSV format)",
                file_okay=True,
                dir_okay=False,
                exists=True,
            ),
        ] = None,
        wait_time: Annotated[
            int, Option("--wait-time", "-t", help="Seconds to wait before capturing screenshot")
        ] = 2,
        scroll: Annotated[bool, Option("--scroll", "-s", help="Capture full scrollable page")] = False,
):
    """
    Execute a batch process to capture screenshots for a list of URLs provided via text
    file or CSV file.

    Parameters:
        urls_txt (Path | None): File containing URLs to capture. Must be a valid file path
            to a text file.
        urls_csv (Path | None): File containing URLs to capture in CSV format. Must be a
            valid file path to a CSV file.
        wait_time (int): Seconds to wait before capturing the screenshot. Defaults to 2.
        scroll (bool): If set to True, captures the full scrollable area of the page.

    Raises:
        BadParameter: Raised if neither --urls_txt nor --urls_csv is specified, or both
            are provided simultaneously.

    This function reads a list of URLs from either a text file or a CSV file and utilizes
    the Peeker utility to capture screenshots for each URL. Options are provided to
    customize the delay before capturing and the ability to capture a full scrollable page.
    A receipt file containing the list of captured URLs is generated in the configured
    output directory.
    """
    if urls_txt is None and urls_csv is None:
        raise BadParameter("Either --urls_txt or --urls_csv must be specified")

    if urls_txt and urls_csv:
        raise BadParameter("Only one of --urls_txt or --urls_csv can be specified")

    if urls_txt:
        urls = parse_textfile(urls_txt)

    if urls_csv:
        urls = urls_csv.read_text().splitlines()

    with Peeker(
            output_dir=CONFIG.output_dir,
            browser=CONFIG.browser,
            timeout=CONFIG.timeout,
            normalize_urls=CONFIG.normalize_urls,
            max_workers=CONFIG.max_workers,
            width=CONFIG.width,
            height=CONFIG.height,
            ignore_https_errors=CONFIG.ignore_https_errors,
            log_level=CONFIG.log_level,
    ) as peeker:
        output = peeker.capture_batch(urls, wait_time=wait_time, scroll=scroll)
        receipt = Path(CONFIG.output_dir, "receipt.txt")
        receipt.write_text("\n".join(output.keys()))


@app.command()
def nmap(
        nmap_xml: Annotated[
            Path,
            Option(
                "--nmap_xml",
                help="Nmap XML output file",
                file_okay=True,
                dir_okay=False,
                exists=True,
            ),
        ],
        wait_time: Annotated[
            int, Option("--wait-time", "-t", help="Seconds to wait before capturing screenshot")
        ] = 2,
        scroll: Annotated[bool, Option("--scroll", "-s", help="Capture full scrollable page")] = False,
):
    """
    Parses an Nmap XML output file to extract HTTP/HTTPS services and captures screenshots of the discovered
    web services. Outputs the results to a specified directory.

    Args:
        nmap_xml (Path): Nmap XML output file. The file must exist and only files are allowed.
        wait_time (int): Seconds to wait before capturing the screenshot. Defaults to 2.
        scroll (bool): Indicates whether to capture the full scrollable webpage. Defaults to False.

    Raises:
        Any errors encountered during the parsing of the Nmap XML file or the screenshot
        capture process will be propagated.

    Returns:
        None
    """
    parser = NmapParser(nmap_xml)
    hosts: list[str] = []
    for host in parser.hosts:
        for port in host.ports:
            if port.service == "http":
                hosts.append(f"http://{host.ip}:{port.port}")
            elif port.service == "https":
                hosts.append(f"https://{host.ip}:{port.port}")

    with Peeker(
            output_dir=CONFIG.output_dir,
            browser=CONFIG.browser,
            timeout=CONFIG.timeout,
            normalize_urls=CONFIG.normalize_urls,
            max_workers=CONFIG.max_workers,
            width=CONFIG.width,
            height=CONFIG.height,
            ignore_https_errors=CONFIG.ignore_https_errors,
            log_level=CONFIG.log_level,
    ) as peeker:
        output = peeker.capture_batch(hosts, wait_time=wait_time, scroll=scroll)
        receipt = Path(CONFIG.output_dir, "receipt.txt")
        receipt.write_text("\n".join(output.keys()))


def cli_main():
    """
    Entry point for the command-line interface (CLI).

    This function initializes and runs the CLI application by invoking the app function.

    Functions:
        app (function): Represents the main CLI application logic.

    Returns:
        None
    """
    app()


if __name__ == "__main__":
    cli_main()
