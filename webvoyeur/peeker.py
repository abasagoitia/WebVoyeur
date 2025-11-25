"""Webpage screenshot capture module using Playwright.

This module provides the Peeker class for capturing webpage screenshots using
Playwright's browser automation framework. It supports both single and batch
capture operations with configurable browser types, timeouts, and concurrent
workers.

Example:
    Basic usage with context manager::

        from webvoyeur.peeker import Peeker
        from webvoyeur.utilities import BrowserType

        with Peeker(browser=BrowserType.firefox) as peeker:
            path = peeker.capture_single("https://example.com")
            print(f"Screenshot saved to: {path}")

    Batch capture::

        urls = ["https://example.com", "https://google.com"]
        with Peeker(max_workers=4) as peeker:
            results = peeker.capture_batch(urls)

Constants:
    DEFAULT_WAIT_TIME (int): Default wait time after page load in seconds.
    DEFAULT_SHORT_TIMEOUT_MS (int): Short timeout for DOM content loading.
    DEFAULT_MAX_WORKERS (int): Default number of concurrent workers.
    DEFAULT_WIDTH (int): Default viewport width in pixels.
    DEFAULT_HEIGHT (int): Default viewport height in pixels.
    DEFAULT_TIMEOUT (int): Default page load timeout in seconds.
"""

import asyncio
import logging
import subprocess
import sys
import time
from logging import Logger
from pathlib import Path

from playwright.async_api import Browser, BrowserContext
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page, Playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import ViewportSize, async_playwright

from webvoyeur.utilities import BrowserType, get_filename_from_url, normalize_url

DEFAULT_WAIT_TIME = 2
DEFAULT_SHORT_TIMEOUT_MS = 2000
DEFAULT_MAX_WORKERS = 4
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_TIMEOUT = 10


class Peeker:
    """Webpage screenshot capture tool using Playwright browser automation.

    This class provides an interface for capturing webpage screenshots with
    configurable browser types, timeouts, viewport sizes, and concurrent worker
    limits. It manages the Playwright browser lifecycle and provides both
    synchronous and asynchronous capture methods.

    Attributes:
        _output_dir (Path): Directory where screenshots are saved.
        _timeout (int): Page load timeout in seconds.
        _normalize_urls (bool): Whether to auto-add https:// to URLs.
        _max_workers (int): Maximum number of concurrent capture workers.
        _width (int): Viewport width in pixels.
        _height (int): Viewport height in pixels.
        _browser_type (BrowserType): Type of browser to use.
        _browser (Browser | None): Playwright browser instance.
        _playwright (Playwright | None): Playwright instance.
        _loop (asyncio.AbstractEventLoop | None): Event loop for async operations.
        _initialized (bool): Whether the browser is initialized.
        _logger (Logger): Logger instance for this class.

    Example:
        >>> from webvoyeur.peeker import Peeker
        >>> from webvoyeur.utilities import BrowserType
        >>> peeker = Peeker(
        ...     output_dir="./screenshots",
        ...     browser=BrowserType.chromium,
        ...     timeout=15,
        ...     max_workers=8
        ... )
        >>> try:
        ...     path = peeker.capture_single("https://example.com")
        ... finally:
        ...     peeker.close()
    """

    def __init__(
        self,
        output_dir: Path | str = "./output",
        browser: BrowserType = BrowserType.firefox,
        timeout: int = DEFAULT_TIMEOUT,
        normalize_urls: bool = True,
        max_workers: int = DEFAULT_MAX_WORKERS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        log_level: int = logging.INFO,
    ):
        """Initialize the Peeker with configuration parameters.

        Args:
            output_dir: Directory path where screenshots will be saved.
                Will be created if it doesn't exist.
            browser: Browser type to use (firefox or chromium).
            timeout: Maximum time in seconds to wait for page load.
                Must be positive.
            normalize_urls: If True, automatically adds https:// to URLs
                that don't have a protocol.
            max_workers: Maximum number of concurrent capture operations
                for batch processing. Must be positive.
            width: Viewport width in pixels. Must be positive.
            height: Viewport height in pixels. Must be positive.
            log_level: Logging level (e.g., logging.INFO, logging.DEBUG).

        Raises:
            TypeError: If a browser is not a BrowserType instance.
            ValueError: If timeout, max_workers, width, or height are invalid.
            RuntimeError: If browser initialization fails.
        """
        _ensure_playwright_browsers()
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = _validate_timeout(timeout)
        self._normalize_urls = normalize_urls
        self._max_workers = _validate_max_workers(max_workers)
        self._width = _validate_dimension(width, "Width")
        self._height = _validate_dimension(height, "Height")
        self._browser: Browser | None = None
        self._playwright: Playwright | None = None
        if not isinstance(browser, BrowserType):
            raise TypeError(f"browser must be a BrowserType, got {type(browser).__name__}")
        self._browser_type = browser
        self._loop: asyncio.AbstractEventLoop | None = None
        self._initialized = False
        self._logger: Logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self._initialize_eventloop()

    def _initialize_eventloop(self) -> None:
        """Initialize the event loop and browser.

        Creates a new event loop, sets it as the current loop, and
        initializes the Playwright browser. Sets the _initialized flag
        to True on success.

        Raises:
            RuntimeError: If browser initialization fails.
        """
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._async_init())
            self._initialized = True
            self._logger.info("Browser initialized")
        except Exception as e:
            self._logger.error(f"Error while launching browser: {e}")
            if self._loop:
                self._loop.close()
            raise

    async def _async_init(self) -> None:
        """Asynchronously initialize Playwright and launch the browser.

        Starts Playwright and launches the browser based on the configured
        browser type (firefox or chromium) in headless mode.

        Raises:
            PlaywrightError: If browser launch fails.
            ValueError: If an invalid browser type is configured.
        """
        try:
            self._playwright = await async_playwright().start()
            match self._browser_type:
                case BrowserType.firefox:
                    self._browser = await self._playwright.firefox.launch(headless=True)
                case BrowserType.chromium:
                    self._browser = await self._playwright.chromium.launch(headless=True)
                case _:
                    raise ValueError(f"Invalid browser type: {self._browser_type}")

            self._logger.info(f"Initialized browser: {self._browser_type.value}")

        except PlaywrightError as e:
            self._logger.error(f"Error while launching browser: {e}")
            raise

    async def _async_capture(
        self,
        page: Page,
        url: str,
        filename: str | None = None,
        wait_time: int = DEFAULT_WAIT_TIME,
        scroll: bool = False,
    ) -> Path:
        """Asynchronously capture a screenshot of a webpage.

        Navigates to the URL, waits for page load, and captures a screenshot.
        Handles timeouts gracefully by attempting partial captures.

        Args:
            page: Playwright Page instance to use for navigation.
            url: URL of the webpage to capture.
            filename: Optional custom filename for the screenshot.
                If None, generates filename from URL.
            wait_time: Additional time in seconds to wait after a page load before capturing a screenshot.
            scroll: If True, captures a full scrollable page. If False,
                captures only viewport.

        Returns:
            Path object pointing to the saved screenshot file.

        Note:
            If page load times out, attempts to capture whatever is loaded.
            The URL will be normalized if normalize_urls is enabled.
        """
        if self._normalize_urls:
            url = normalize_url(url)
            self._logger.debug(f"Normalized URL: {url}")
        else:
            url = url.strip()

        full_timeout_ms = self._timeout * 1_000
        self._logger.debug(f"Using timeout of {self._timeout}s for {url}")

        try:
            await page.goto(url, wait_until="load", timeout=full_timeout_ms)
            self._logger.debug(f"Loaded {url}")
        except PlaywrightTimeoutError:
            self._logger.debug(f"Page load timeout for {url}, attempting partial capture...")

        short_timeout_ms = min(DEFAULT_SHORT_TIMEOUT_MS, full_timeout_ms)

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=short_timeout_ms)
            self._logger.debug(f"DOM content loaded for {url}")
        except PlaywrightTimeoutError:
            self._logger.debug(f"DOM content loading timeout for {url}, continuing...")

        if wait_time > 0:
            self._logger.debug(f"Waiting {wait_time}s before screenshot...")
            await asyncio.sleep(wait_time)

        if filename is None:
            filename = get_filename_from_url(url)
            self._logger.debug(f"No filename specified, using default: {filename}")

        screenshot_path = self._output_dir / filename
        await page.screenshot(path=str(screenshot_path), full_page=scroll)
        self._logger.info(f"Screenshot saved to: {screenshot_path}")

        return screenshot_path

    async def _async_single(
        self,
        url: str,
        filename: str | None = None,
        wait_time: int = DEFAULT_WAIT_TIME,
        scroll: bool = False,
    ) -> Path | None:
        """Asynchronously capture a single webpage screenshot.

        Creates a new browser context and page, captures the screenshot,
        and cleans up resources.

        Args:
            url: URL of the webpage to capture.
            filename: Optional custom filename for the screenshot.
            wait_time: Additional wait time in seconds after the page loads.
            scroll: Whether to capture a full scrollable page.

        Returns:
            Path to the saved screenshot, or None if capture failed.

        Raises:
            RuntimeError: If browser is not initialized.
        """
        context: BrowserContext | None = None

        try:
            if self._browser is None:
                raise RuntimeError("Browser is not initialized")

            viewport = ViewportSize(width=self._width, height=self._height)
            context = await self._browser.new_context(viewport=viewport)
            self._logger.debug(f"Created new context for {url}")
            page = await context.new_page()
            self._logger.debug(f"Created new page for {url}")

            return await self._async_capture(
                page=page, url=url, filename=filename, wait_time=wait_time, scroll=scroll
            )

        except PlaywrightError as e:
            self._logger.error(f"Error capturing page: {e}")
            return None

        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception as e:
                    self._logger.debug(f"Error closing context: {e}")

    async def _async_batch(
        self, urls: list[str], wait_time: int = DEFAULT_WAIT_TIME, scroll: bool = False
    ) -> dict[str, Path | None]:
        """Asynchronously capture multiple webpage screenshots concurrently.

        Uses semaphore to limit concurrent workers based on max_workers
        configuration. Each URL is processed in its own browser context.

        Args:
            urls: List of URLs to capture.
            wait_time: Additional wait time in seconds after the page loads
                for each URL.
            scroll: Whether to capture full scrollable pages.

        Returns:
            Dictionary mapping each URL to its screenshot Path (or None if
            capture failed).

        Note:
            Processing time is logged after all captures are complete.
        """
        semaphore = asyncio.Semaphore(self._max_workers)
        results: dict[str, Path | None] = {}
        start_time = time.time()

        async def worker(url: str) -> None:
            """Worker coroutine to capture a single URL.

            Args:
                url: URL to capture.
            """
            async with semaphore:
                context = None
                try:
                    viewport = ViewportSize(width=self._width, height=self._height)
                    context = await self._browser.new_context(viewport=viewport)
                    page = await context.new_page()
                    path = await self._async_capture(
                        page=page, url=url, wait_time=wait_time, scroll=scroll
                    )
                    results[url] = path
                except Exception as e:
                    self._logger.error(f"Worker error for {url}: {e}")
                    results[url] = None
                finally:
                    if context is not None:
                        try:
                            await context.close()
                        except Exception:
                            pass

        tasks = [asyncio.create_task(worker(url)) for url in urls]
        await asyncio.gather(*tasks)

        self._logger.info(f"Batch capture complete in {time.time() - start_time:.2f} seconds")
        return results

    def capture_single(
        self,
        url: str,
        filename: str | None = None,
        wait_time: int = DEFAULT_WAIT_TIME,
        scroll: bool = False,
    ) -> Path | None:
        """Capture a single webpage screenshot (synchronous).

        This is the main public method for capturing individual webpages.
        It wraps the asynchronous implementation with event loop management.

        Args:
            url: URL of the webpage to capture.
            filename: Optional custom filename for the screenshot. If None,
                generates a filename from the URL.
            wait_time: Additional time in seconds to wait after page load
                before capturing. Must be non-negative.
            scroll: If True, captures the full scrollable page. If False,
                captures only the visible viewport.

        Returns:
            Path object pointing to the saved screenshot file, or None if
            capture failed.

        Raises:
            RuntimeError: If the Peeker is not properly initialized.
            ValueError: If wait_time is negative.

        Example:
            >>> peeker = Peeker()
            >>> try:
            ...     path = peeker.capture_single(
            ...         "https://example.com",
            ...         filename="example.png",
            ...         wait_time=3,
            ...         scroll=True
            ...     )
            ...     print(f"Saved to: {path}")
            ... finally:
            ...     peeker.close()
        """
        if not self._initialized or not self._loop:
            raise RuntimeError("WebpageScreenshotter is not properly initialized")

        wait_time = _validate_wait_time(wait_time)

        self._logger.info(f"Capturing single page: {url}")
        return self._loop.run_until_complete(
            self._async_single(url=url, filename=filename, wait_time=wait_time, scroll=scroll)
        )

    def capture_batch(
        self, urls: list[str], wait_time: int = DEFAULT_WAIT_TIME, scroll: bool = False
    ) -> dict:
        """Capture multiple webpage screenshots concurrently (synchronous).

        This is the main public method for batch capturing webpages. It
        processes multiple URLs concurrently with a limit based on max_workers.

        Args:
            urls: List of URLs to capture. Each URL will be processed
                concurrently up to the max_workers limit.
            wait_time: Additional time in seconds to wait after page load
                for each URL. Must be non-negative.
            scroll: If True, captures full scrollable pages. If False,
                captures only viewports.

        Returns:
            Dictionary mapping each URL to its screenshot Path (or None if
            capture failed for that URL).

        Raises:
            RuntimeError: If the Peeker is not properly initialized.
            ValueError: If wait_time is negative.

        Example:
            >>> urls = [
            ...     "https://example.com",
            ...     "https://google.com",
            ...     "https://github.com"
            ... ]
            >>> peeker = Peeker(max_workers=4)
            >>> try:
            ...     results = peeker.capture_batch(urls, scroll=False)
            ...     for url, path in results.items():
            ...         if path:
            ...             print(f"✓ {url} -> {path}")
            ...         else:
            ...             print(f"✗ {url} failed")
            ... finally:
            ...     peeker.close()
        """
        if not self._initialized or not self._loop:
            raise RuntimeError("WebpageScreenshotter is not properly initialized")

        wait_time = _validate_wait_time(wait_time)

        self._logger.info(f"Capturing batch job")
        return self._loop.run_until_complete(
            self._async_batch(urls=urls, wait_time=wait_time, scroll=scroll)
        )

    def close(self) -> None:
        """Clean up browser resources and close the event loop.

        This method should be called when done using the Peeker to properly
        release browser resources. It safely handles cases where the browser
        was never initialized or already closed.

        The method attempts to close the browser, stop Playwright, and close
        the event loop. Errors during cleanup are logged but not raised.

        Example:
            >>> peeker = Peeker()
            >>> try:
            ...     peeker.capture_single("https://example.com")
            ... finally:
            ...     peeker.close()

        Note:
            This method is automatically called when using Peeker as a
            context manager (with statement).
        """
        if not self._initialized:
            return

        if self._loop is None or self._loop.is_closed():
            return

        try:
            self._loop.run_until_complete(self._async_cleanup())
        except Exception as e:
            self._logger.warning(f"Error during cleanup: {e}")
        finally:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.close()
                    self._initialized = False
            except Exception as e:
                self._logger.debug(f"Error closing event loop: {e}")

    async def _async_cleanup(self) -> None:
        """Asynchronously clean up Playwright and browser resources.

        Closes the browser and stops the Playwright instance. Errors are
        logged but not raised to ensure cleanup completes as much as possible.
        """
        if self._browser is not None:
            try:
                await self._browser.close()
                self._logger.debug("Browser closed")
            except Exception as e:
                self._logger.warning(f"Error closing browser: {e}")

        if self._playwright is not None:
            try:
                await self._playwright.stop()
                self._logger.debug("Playwright stopped")
            except Exception as e:
                self._logger.warning(f"Error stopping playwright: {e}")

    def __enter__(self):
        """Enter the context manager.

        Returns:
            Self for use in with statements.

        Example:
            >>> with Peeker() as peeker:
            ...     peeker.capture_single("https://example.com")
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up resources.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Returns:
            False to propagate any exception that occurred.
        """
        self.close()
        return False


def _validate_timeout(timeout: int) -> int:
    """Validate that timeout is a positive integer.

    Args:
        timeout: Timeout value in seconds to validate.

    Returns:
        The validated timeout value.

    Raises:
        ValueError: If timeout is not positive.
    """
    if timeout <= 0:
        raise ValueError("Timeout must be a positive integer")
    return timeout


def _validate_max_workers(max_workers: int) -> int:
    """Validate that max_workers is a positive integer.

    Args:
        max_workers: Number of concurrent workers to validate.

    Returns:
        The validated max_workers value.

    Raises:
        ValueError: If max_workers is not positive.
    """
    if max_workers <= 0:
        raise ValueError("Max workers must be a positive integer")
    return max_workers


def _validate_dimension(dimension: int, name: str) -> int:
    """Validate that a dimension (width or height) is a positive integer.

    Args:
        dimension: Dimension value in pixels to validate.
        name: Name of the dimension (for error messages).

    Returns:
        The validated dimension value.

    Raises:
        ValueError: If the dimension is not positive.
    """
    if dimension <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return dimension


def _validate_wait_time(wait_time: int) -> int:
    """Validate that wait_time is non-negative.

    Args:
        wait_time: Wait time in seconds to validate.

    Returns:
        The validated wait_time value.

    Raises:
        ValueError: If wait_time is negative.
    """
    if wait_time < 0:
        raise ValueError("Wait time cannot be negative")
    return wait_time


def _ensure_playwright_browsers():
    """
    Checks if Playwright browsers are installed and installs them if not.
    This function makes the application "immediately available" by handling
    Playwright's browser dependencies on first run.
    """
    print("Checking Playwright browser binaries...")
    try:
        # Use sys.executable to ensure the correct python environment's playwright is used.
        # The 'playwright install' command is idempotent; it only downloads missing browsers.
        subprocess.run(
            [sys.executable, "-m", "playwright", "install"],
            check=True,
            capture_output=True,  # Capture output to avoid polluting stdout unless there's an error
            text=True,
        )
        print("Playwright browser binaries are installed or already up-to-date.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Playwright browser binaries: {e.stderr}")
        print("Please ensure you have network connectivity and permissions to install browsers.")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Playwright command not found. Please ensure 'playwright' is installed in your environment."
        )
        sys.exit(1)
