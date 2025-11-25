import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from typer import BadParameter, Context, Option, Typer, echo

from webvoyeur.peeker import Peeker
from webvoyeur.utilities import BrowserType, parse_textfile


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Config:
    output_dir: Path = Path("./output")
    browser: BrowserType = BrowserType.firefox
    timeout: int = 10
    normalize_urls: bool = True
    max_workers: int = 4
    width: int = 1280
    height: int = 720
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
    ] = Config.height,
    browser: Annotated[
        BrowserType, Option("--browser", "-b", help="Browser: chrome or firefox")
    ] = CONFIG.browser.value,
):
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
    with Peeker(
        output_dir=CONFIG.output_dir,
        browser=Config.browser,
        timeout=CONFIG.timeout,
        normalize_urls=CONFIG.normalize_urls,
        max_workers=CONFIG.max_workers,
        width=CONFIG.width,
        height=CONFIG.height,
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
        browser=Config.browser,
        timeout=CONFIG.timeout,
        normalize_urls=CONFIG.normalize_urls,
        max_workers=CONFIG.max_workers,
        width=CONFIG.width,
        height=CONFIG.height,
        log_level=CONFIG.log_level,
    ) as peeker:
        output = peeker.capture_batch(urls, wait_time=wait_time, scroll=scroll)
        receipt = Path(CONFIG.output_dir, "receipt.txt")
        receipt.write_text("\n".join(output.keys()))


def cli_main():
    app()


if __name__ == "__main__":
    cli_main()
