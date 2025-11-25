"""Integration tests for the peeker module without mocking.

These tests verify that the Peeker class functions correctly in real scenarios
by using actual Playwright browser instances and capturing real webpages.
"""

import http.server
import logging
import shutil
import socketserver
import threading
import time
from pathlib import Path

import pytest

from webvoyeur.peeker import (
    DEFAULT_HEIGHT,
    DEFAULT_MAX_WORKERS,
    DEFAULT_TIMEOUT,
    DEFAULT_WAIT_TIME,
    DEFAULT_WIDTH,
    Peeker,
    _validate_dimension,
    _validate_max_workers,
    _validate_timeout,
    _validate_wait_time,
)
from webvoyeur.utilities import BrowserType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "test_screenshots"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup after test
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def simple_html_file(tmp_path):
    """Create a simple HTML file for testing."""
    html_file = tmp_path / "test.html"
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <style>
            body { background-color: #f0f0f0; padding: 20px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Test Page</h1>
        <p>This is a test page for screenshot capture.</p>
    </body>
    </html>
    """
    html_file.write_text(html_content)
    return html_file


@pytest.fixture
def large_html_file(tmp_path):
    """Create a large HTML file for scroll testing."""
    html_file = tmp_path / "large_test.html"
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Large Test Page</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .section { height: 500px; padding: 20px; margin: 10px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h1>Large Test Page</h1>
    """
    for i in range(10):
        html_content += f"""
        <div class="section">
            <h2>Section {i + 1}</h2>
            <p>This is section {i + 1} content.</p>
        </div>
        """
    html_content += """
    </body>
    </html>
    """
    html_file.write_text(html_content)
    return html_file


# ============================================================================
# Validation Functions Tests
# ============================================================================


class TestValidationFunctions:
    """Test all validation helper functions."""

    def test_validate_timeout_positive(self):
        """Test that positive timeout values are accepted."""
        assert _validate_timeout(1) == 1
        assert _validate_timeout(10) == 10
        assert _validate_timeout(100) == 100
        assert _validate_timeout(1000) == 1000

    def test_validate_timeout_zero(self):
        """Test that zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="Timeout must be a positive integer"):
            _validate_timeout(0)

    def test_validate_timeout_negative(self):
        """Test that negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="Timeout must be a positive integer"):
            _validate_timeout(-1)
        with pytest.raises(ValueError, match="Timeout must be a positive integer"):
            _validate_timeout(-100)

    def test_validate_max_workers_positive(self):
        """Test that positive max_workers values are accepted."""
        assert _validate_max_workers(1) == 1
        assert _validate_max_workers(4) == 4
        assert _validate_max_workers(16) == 16
        assert _validate_max_workers(100) == 100

    def test_validate_max_workers_zero(self):
        """Test that zero max_workers raises ValueError."""
        with pytest.raises(ValueError, match="Max workers must be a positive integer"):
            _validate_max_workers(0)

    def test_validate_max_workers_negative(self):
        """Test that negative max_workers raises ValueError."""
        with pytest.raises(ValueError, match="Max workers must be a positive integer"):
            _validate_max_workers(-1)
        with pytest.raises(ValueError, match="Max workers must be a positive integer"):
            _validate_max_workers(-10)

    def test_validate_dimension_positive(self):
        """Test that positive dimension values are accepted."""
        assert _validate_dimension(100, "Width") == 100
        assert _validate_dimension(1920, "Height") == 1920
        assert _validate_dimension(10000, "Width") == 10000

    def test_validate_dimension_zero(self):
        """Test that zero dimension raises ValueError."""
        with pytest.raises(ValueError, match="Width must be a positive integer"):
            _validate_dimension(0, "Width")

    def test_validate_dimension_negative(self):
        """Test that negative dimension raises ValueError."""
        with pytest.raises(ValueError, match="Height must be a positive integer"):
            _validate_dimension(-100, "Height")
        with pytest.raises(ValueError, match="Width must be a positive integer"):
            _validate_dimension(-1, "Width")

    def test_validate_wait_time_zero(self):
        """Test that zero wait_time is accepted."""
        assert _validate_wait_time(0) == 0

    def test_validate_wait_time_positive(self):
        """Test that positive wait_time values are accepted."""
        assert _validate_wait_time(1) == 1
        assert _validate_wait_time(5) == 5
        assert _validate_wait_time(100) == 100

    def test_validate_wait_time_negative(self):
        """Test that negative wait_time raises ValueError."""
        with pytest.raises(ValueError, match="Wait time cannot be negative"):
            _validate_wait_time(-1)
        with pytest.raises(ValueError, match="Wait time cannot be negative"):
            _validate_wait_time(-10)


# ============================================================================
# Peeker Initialization Tests
# ============================================================================


class TestPeekerInitialization:
    """Test Peeker initialization with various configurations."""

    def test_init_default_values(self, test_output_dir):
        """Test initialization with default values."""
        peeker = Peeker(output_dir=test_output_dir)
        try:
            assert peeker._output_dir == test_output_dir
            assert peeker._timeout == DEFAULT_TIMEOUT
            assert peeker._normalize_urls is True
            assert peeker._max_workers == DEFAULT_MAX_WORKERS
            assert peeker._width == DEFAULT_WIDTH
            assert peeker._height == DEFAULT_HEIGHT
            assert peeker._browser_type == BrowserType.firefox
            assert peeker._initialized is True
            assert peeker._browser is not None
            assert peeker._playwright is not None
            assert peeker._loop is not None
        finally:
            peeker.close()

    def test_init_custom_values(self, test_output_dir):
        """Test initialization with custom values."""
        peeker = Peeker(
            output_dir=test_output_dir,
            browser=BrowserType.chromium,
            timeout=20,
            normalize_urls=False,
            max_workers=8,
            width=1920,
            height=1080,
            log_level=logging.DEBUG,
        )
        try:
            assert peeker._output_dir == test_output_dir
            assert peeker._timeout == 20
            assert peeker._normalize_urls is False
            assert peeker._max_workers == 8
            assert peeker._width == 1920
            assert peeker._height == 1080
            assert peeker._browser_type == BrowserType.chromium
            assert peeker._initialized is True
        finally:
            peeker.close()

    def test_init_creates_output_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_screenshots_dir"
        assert not output_dir.exists()

        peeker = Peeker(output_dir=output_dir)
        try:
            assert output_dir.exists()
            assert output_dir.is_dir()
        finally:
            peeker.close()

    def test_init_invalid_browser_type(self, test_output_dir):
        """Test that invalid browser type raises TypeError."""
        with pytest.raises(TypeError, match="browser must be a BrowserType"):
            Peeker(output_dir=test_output_dir, browser="firefox")

    def test_init_invalid_timeout(self, test_output_dir):
        """Test that invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="Timeout must be a positive integer"):
            Peeker(output_dir=test_output_dir, timeout=0)

    def test_init_invalid_max_workers(self, test_output_dir):
        """Test that invalid max_workers raises ValueError."""
        with pytest.raises(ValueError, match="Max workers must be a positive integer"):
            Peeker(output_dir=test_output_dir, max_workers=-1)

    def test_init_invalid_width(self, test_output_dir):
        """Test that invalid width raises ValueError."""
        with pytest.raises(ValueError, match="Width must be a positive integer"):
            Peeker(output_dir=test_output_dir, width=0)

    def test_init_invalid_height(self, test_output_dir):
        """Test that invalid height raises ValueError."""
        with pytest.raises(ValueError, match="Height must be a positive integer"):
            Peeker(output_dir=test_output_dir, height=-100)

    def test_init_string_output_dir(self, tmp_path):
        """Test that string output_dir is converted to Path."""
        output_dir_str = str(tmp_path / "string_output")
        peeker = Peeker(output_dir=output_dir_str)
        try:
            assert isinstance(peeker._output_dir, Path)
            assert peeker._output_dir.exists()
        finally:
            peeker.close()

    def test_init_firefox_browser(self, test_output_dir):
        """Test initialization with Firefox browser."""
        peeker = Peeker(output_dir=test_output_dir, browser=BrowserType.firefox)
        try:
            assert peeker._browser_type == BrowserType.firefox
            assert peeker._browser is not None
        finally:
            peeker.close()

    def test_init_chromium_browser(self, test_output_dir):
        """Test initialization with Chromium browser."""
        peeker = Peeker(output_dir=test_output_dir, browser=BrowserType.chromium)
        try:
            assert peeker._browser_type == BrowserType.chromium
            assert peeker._browser is not None
        finally:
            peeker.close()


# ============================================================================
# Single Capture Tests
# ============================================================================


class TestCaptureSingle:
    """Test single webpage capture functionality."""

    def test_capture_single_local_file(self, test_output_dir, simple_html_file):
        """Test capturing a local HTML file."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            result = peeker.capture_single(file_url)

            assert result is not None
            assert result.exists()
            assert result.suffix == ".png"
            assert result.stat().st_size > 0
        finally:
            peeker.close()

    def test_capture_single_with_custom_filename(self, test_output_dir, simple_html_file):
        """Test capturing with a custom filename."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            custom_filename = "custom_screenshot.png"
            result = peeker.capture_single(file_url, filename=custom_filename)

            assert result is not None
            assert result.name == custom_filename
            assert result.exists()
        finally:
            peeker.close()

    def test_capture_single_with_scroll(self, test_output_dir, large_html_file):
        """Test capturing with scroll=True for full page."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{large_html_file.absolute()}"
            result_no_scroll = peeker.capture_single(
                file_url, filename="no_scroll.png", scroll=False
            )
            result_scroll = peeker.capture_single(file_url, filename="with_scroll.png", scroll=True)

            assert result_no_scroll is not None
            assert result_scroll is not None
            assert result_no_scroll.exists()
            assert result_scroll.exists()
            # Full page screenshot should be larger
            assert result_scroll.stat().st_size > result_no_scroll.stat().st_size
        finally:
            peeker.close()

    def test_capture_single_with_wait_time(self, test_output_dir, simple_html_file):
        """Test capturing with custom wait time."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            start_time = time.time()
            result = peeker.capture_single(file_url, wait_time=1)
            elapsed_time = time.time() - start_time

            assert result is not None
            assert result.exists()
            # Should take at least 1 second due to wait_time
            assert elapsed_time >= 1.0
        finally:
            peeker.close()

    def test_capture_single_zero_wait_time(self, test_output_dir, simple_html_file):
        """Test capturing with zero wait time."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            result = peeker.capture_single(file_url, wait_time=0)

            assert result is not None
            assert result.exists()
        finally:
            peeker.close()

    def test_capture_single_invalid_wait_time(self, test_output_dir):
        """Test that negative wait_time raises ValueError."""
        peeker = Peeker(output_dir=test_output_dir)
        try:
            with pytest.raises(ValueError, match="Wait time cannot be negative"):
                peeker.capture_single("file:///tmp/test.html", wait_time=-1)
        finally:
            peeker.close()

    def test_capture_single_without_normalization(self, test_output_dir, simple_html_file):
        """Test capturing without URL normalization."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            result = peeker.capture_single(file_url)

            assert result is not None
            assert result.exists()
        finally:
            peeker.close()

    def test_capture_single_nonexistent_file(self, test_output_dir):
        """Test capturing a nonexistent file URL."""
        peeker = Peeker(output_dir=test_output_dir, timeout=5, normalize_urls=False)
        try:
            # This file doesn't exist
            result = peeker.capture_single("file:///nonexistent/path/to/file.html")

            # Should handle gracefully - might return None or a path
            assert result is None or isinstance(result, Path)
        finally:
            peeker.close()


# ============================================================================
# Batch Capture Tests
# ============================================================================


class TestCaptureBatch:
    """Test batch webpage capture functionality."""

    def test_capture_batch_multiple_files(self, test_output_dir, tmp_path):
        """Test capturing multiple local HTML files."""
        # Create multiple HTML files
        html_files = []
        for i in range(3):
            html_file = tmp_path / f"test_{i}.html"
            html_file.write_text(f"<html><body><h1>Test Page {i}</h1></body></html>")
            html_files.append(html_file)

        peeker = Peeker(output_dir=test_output_dir, max_workers=2, normalize_urls=False)
        try:
            urls = [f"file://{f.absolute()}" for f in html_files]
            results = peeker.capture_batch(urls)

            assert isinstance(results, dict)
            assert len(results) == 3
            for url, path in results.items():
                assert url in urls
                assert path is not None
                assert path.exists()
                assert path.suffix == ".png"
        finally:
            peeker.close()

    def test_capture_batch_with_wait_time(self, test_output_dir, simple_html_file):
        """Test batch capture with custom wait time."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            urls = [file_url]

            start_time = time.time()
            results = peeker.capture_batch(urls, wait_time=1)
            elapsed_time = time.time() - start_time

            assert len(results) == 1
            # Should take at least 1 second due to wait_time
            assert elapsed_time >= 1.0
        finally:
            peeker.close()

    def test_capture_batch_with_scroll(self, test_output_dir, large_html_file):
        """Test batch capture with scroll=True."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{large_html_file.absolute()}"
            urls = [file_url]
            results = peeker.capture_batch(urls, scroll=True)

            assert len(results) == 1
            path = results[file_url]
            assert path is not None
            assert path.exists()
        finally:
            peeker.close()

    def test_capture_batch_empty_list(self, test_output_dir):
        """Test batch capture with empty URL list."""
        peeker = Peeker(output_dir=test_output_dir)
        try:
            results = peeker.capture_batch([])

            assert isinstance(results, dict)
            assert len(results) == 0
        finally:
            peeker.close()

    def test_capture_batch_single_url(self, test_output_dir, simple_html_file):
        """Test batch capture with single URL."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            results = peeker.capture_batch([file_url])

            assert len(results) == 1
            assert file_url in results
            assert results[file_url] is not None
        finally:
            peeker.close()

    def test_capture_batch_invalid_wait_time(self, test_output_dir):
        """Test batch capture with invalid wait_time."""
        peeker = Peeker(output_dir=test_output_dir)
        try:
            with pytest.raises(ValueError, match="Wait time cannot be negative"):
                peeker.capture_batch(["file:///tmp/test.html"], wait_time=-5)
        finally:
            peeker.close()

    def test_capture_batch_concurrent_workers(self, test_output_dir, tmp_path):
        """Test batch capture respects max_workers setting."""
        # Create multiple HTML files
        html_files = []
        for i in range(5):
            html_file = tmp_path / f"concurrent_{i}.html"
            html_file.write_text(f"<html><body><h1>Concurrent Test {i}</h1></body></html>")
            html_files.append(html_file)

        peeker = Peeker(output_dir=test_output_dir, max_workers=3, normalize_urls=False)
        try:
            urls = [f"file://{f.absolute()}" for f in html_files]
            start_time = time.time()
            results = peeker.capture_batch(urls)
            elapsed_time = time.time() - start_time

            assert len(results) == 5
            # With concurrency, should complete within reasonable time
            assert elapsed_time < 30
            # Verify all succeeded
            for path in results.values():
                assert path is not None
                assert path.exists()
        finally:
            peeker.close()

    def test_capture_batch_with_mix_of_files(self, test_output_dir, tmp_path, simple_html_file):
        """Test batch capture with mix of valid and nonexistent files."""
        # Create one valid file
        valid_url = f"file://{simple_html_file.absolute()}"
        # Use a nonexistent file
        invalid_url = "file:///definitely/does/not/exist/nowhere.html"
        urls = [valid_url, invalid_url]

        peeker = Peeker(output_dir=test_output_dir, timeout=5, normalize_urls=False)
        try:
            results = peeker.capture_batch(urls)

            assert len(results) == 2
            assert valid_url in results
            assert invalid_url in results
            # Valid URL should succeed
            assert results[valid_url] is not None
            assert results[valid_url].exists()
            # Invalid URL should fail
            assert results[invalid_url] is None
        finally:
            peeker.close()


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager_basic_usage(self, test_output_dir, simple_html_file):
        """Test basic context manager usage."""
        with Peeker(output_dir=test_output_dir, normalize_urls=False) as peeker:
            assert isinstance(peeker, Peeker)
            assert peeker._initialized is True

            file_url = f"file://{simple_html_file.absolute()}"
            result = peeker.capture_single(file_url)
            assert result is not None

        # After context, should be closed
        assert peeker._initialized is False

    def test_context_manager_with_exception(self, test_output_dir):
        """Test context manager properly closes even with exception."""
        try:
            with Peeker(output_dir=test_output_dir) as peeker:
                assert peeker._initialized is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be closed after exception
        assert peeker._initialized is False

    def test_context_manager_multiple_operations(self, test_output_dir, simple_html_file):
        """Test multiple operations within context manager."""
        with Peeker(output_dir=test_output_dir, normalize_urls=False) as peeker:
            file_url = f"file://{simple_html_file.absolute()}"

            result1 = peeker.capture_single(file_url, filename="shot1.png")
            result2 = peeker.capture_single(file_url, filename="shot2.png")
            results_batch = peeker.capture_batch([file_url])

            assert result1 is not None
            assert result2 is not None
            assert len(results_batch) == 1


# ============================================================================
# Resource Management Tests
# ============================================================================


class TestResourceManagement:
    """Test proper resource management and cleanup."""

    def test_close_method(self, test_output_dir):
        """Test explicit close method."""
        peeker = Peeker(output_dir=test_output_dir)
        assert peeker._initialized is True

        peeker.close()
        assert peeker._initialized is False

    def test_close_idempotent(self, test_output_dir):
        """Test that close can be called multiple times safely."""
        peeker = Peeker(output_dir=test_output_dir)
        peeker.close()
        peeker.close()  # Should not raise
        peeker.close()  # Should not raise

    def test_operations_after_close(self, test_output_dir, simple_html_file):
        """Test that operations after close raise RuntimeError."""
        peeker = Peeker(output_dir=test_output_dir)
        peeker.close()

        with pytest.raises(RuntimeError, match="not properly initialized"):
            file_url = f"file://{simple_html_file.absolute()}"
            peeker.capture_single(file_url)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test various configuration options."""

    def test_different_viewport_sizes(self, test_output_dir, simple_html_file):
        """Test capturing with different viewport sizes."""
        sizes = [(800, 600), (1920, 1080), (1024, 768)]

        for width, height in sizes:
            peeker = Peeker(
                output_dir=test_output_dir,
                width=width,
                height=height,
                normalize_urls=False,
            )
            try:
                file_url = f"file://{simple_html_file.absolute()}"
                result = peeker.capture_single(
                    file_url, filename=f"screenshot_{width}x{height}.png"
                )

                assert result is not None
                assert result.exists()
            finally:
                peeker.close()

    def test_different_timeouts(self, test_output_dir, simple_html_file):
        """Test with different timeout values."""
        timeouts = [5, 15, 30]

        for timeout in timeouts:
            peeker = Peeker(output_dir=test_output_dir, timeout=timeout, normalize_urls=False)
            try:
                assert peeker._timeout == timeout
                file_url = f"file://{simple_html_file.absolute()}"
                result = peeker.capture_single(file_url)
                assert result is not None
            finally:
                peeker.close()

    def test_different_browsers(self, test_output_dir, simple_html_file):
        """Test with different browser types."""
        browsers = [BrowserType.firefox, BrowserType.chromium]

        for browser in browsers:
            peeker = Peeker(output_dir=test_output_dir, browser=browser, normalize_urls=False)
            try:
                assert peeker._browser_type == browser
                file_url = f"file://{simple_html_file.absolute()}"
                result = peeker.capture_single(file_url)
                assert result is not None
            finally:
                peeker.close()


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_viewport(self, test_output_dir, simple_html_file):
        """Test with very small viewport size."""
        peeker = Peeker(output_dir=test_output_dir, width=100, height=100, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            result = peeker.capture_single(file_url)

            assert result is not None
            assert result.exists()
        finally:
            peeker.close()

    def test_very_large_viewport(self, test_output_dir, simple_html_file):
        """Test with very large viewport size."""
        peeker = Peeker(output_dir=test_output_dir, width=3840, height=2160, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            result = peeker.capture_single(file_url)

            assert result is not None
            assert result.exists()
        finally:
            peeker.close()

    def test_many_max_workers(self, test_output_dir, tmp_path):
        """Test with large number of max workers."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        peeker = Peeker(output_dir=test_output_dir, max_workers=20, normalize_urls=False)
        try:
            urls = [f"file://{html_file.absolute()}"] * 5
            results = peeker.capture_batch(urls)

            assert len(results) == 1
        finally:
            peeker.close()

    def test_single_max_worker(self, test_output_dir, tmp_path):
        """Test with single max worker (sequential processing)."""
        html_files = []
        for i in range(3):
            html_file = tmp_path / f"test_{i}.html"
            html_file.write_text(f"<html><body>Test {i}</body></html>")
            html_files.append(html_file)

        peeker = Peeker(output_dir=test_output_dir, max_workers=1, normalize_urls=False)
        try:
            urls = [f"file://{f.absolute()}" for f in html_files]
            results = peeker.capture_batch(urls)

            assert len(results) == 1
        finally:
            peeker.close()

    def test_screenshot_filename_special_characters(self, test_output_dir, simple_html_file):
        """Test capturing with filename containing special characters."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"
            # Use safe special characters
            filename = "test_screenshot-2024.png"
            result = peeker.capture_single(file_url, filename=filename)

            assert result is not None
            assert result.name == filename
            assert result.exists()
        finally:
            peeker.close()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test complete integration scenarios."""

    def test_complete_workflow_single(self, test_output_dir, simple_html_file):
        """Test complete workflow for single capture."""
        with Peeker(
            output_dir=test_output_dir,
            browser=BrowserType.firefox,
            timeout=15,
            width=1920,
            height=1080,
            normalize_urls=False,
        ) as peeker:
            file_url = f"file://{simple_html_file.absolute()}"
            result = peeker.capture_single(
                url=file_url,
                filename="integration_test.png",
                wait_time=1,
                scroll=True,
            )

            assert result is not None
            assert result.exists()
            assert result.name == "integration_test.png"
            assert result.stat().st_size > 0

    def test_complete_workflow_batch(self, test_output_dir, tmp_path):
        """Test complete workflow for batch capture."""
        # Create test HTML files
        html_files = []
        for i in range(5):
            html_file = tmp_path / f"page_{i}.html"
            content = f"""
            <html>
            <head><title>Page {i}</title></head>
            <body><h1>Page {i}</h1><p>Content for page {i}</p></body>
            </html>
            """
            html_file.write_text(content)
            html_files.append(html_file)

        with Peeker(
            output_dir=test_output_dir,
            browser=BrowserType.chromium,
            max_workers=3,
            timeout=10,
            normalize_urls=False,
        ) as peeker:
            urls = [f"file://{f.absolute()}" for f in html_files]
            results = peeker.capture_batch(urls, wait_time=0, scroll=False)

            assert len(results) == 5
            successful = sum(1 for path in results.values() if path is not None)
            assert successful == 5

    def test_reuse_peeker_instance(self, test_output_dir, simple_html_file):
        """Test reusing the same Peeker instance for multiple operations."""
        peeker = Peeker(output_dir=test_output_dir, normalize_urls=False)
        try:
            file_url = f"file://{simple_html_file.absolute()}"

            # Multiple single captures
            result1 = peeker.capture_single(file_url, filename="reuse1.png")
            result2 = peeker.capture_single(file_url, filename="reuse2.png")

            # Batch capture
            # When duplicate URLs are passed to capture_batch and the return type is a dictionary
            # with URLs as keys, only one entry for the unique URL will exist.
            # The last processed URL's result will overwrite previous ones.
            results = peeker.capture_batch([file_url, file_url])

            # Another single capture
            result3 = peeker.capture_single(file_url, filename="reuse3.png")

            assert result1 is not None and result1.exists()
            assert result2 is not None and result2.exists()
            # Expecting len(results) == 1 because the two identical URLs result in one dictionary key
            assert len(results) == 1
            assert file_url in results
            assert results[file_url] is not None and results[file_url].exists()
            assert result3 is not None and result3.exists()
        finally:
            peeker.close()


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Test module constants."""

    def test_default_constants_values(self):
        """Test that default constants have expected values."""
        assert DEFAULT_WAIT_TIME == 2
        assert DEFAULT_MAX_WORKERS == 4
        assert DEFAULT_WIDTH == 1280
        assert DEFAULT_HEIGHT == 720
        assert DEFAULT_TIMEOUT == 10

    def test_constants_are_positive(self):
        """Test that all default constants are positive or zero."""
        assert DEFAULT_WAIT_TIME >= 0
        assert DEFAULT_MAX_WORKERS > 0
        assert DEFAULT_WIDTH > 0
        assert DEFAULT_HEIGHT > 0
        assert DEFAULT_TIMEOUT > 0

    def test_peeker_uses_constants(self, test_output_dir):
        """Test that Peeker uses constants as defaults."""
        peeker = Peeker(output_dir=test_output_dir)
        try:
            assert peeker._timeout == DEFAULT_TIMEOUT
            assert peeker._max_workers == DEFAULT_MAX_WORKERS
            assert peeker._width == DEFAULT_WIDTH
            assert peeker._height == DEFAULT_HEIGHT
        finally:
            peeker.close()
