# WebVoyeur

WebVoyeur is a Python-based tool for capturing webpage screenshots with rendered CSS and JavaScript. It leverages 
Playwright for robust browser automation, allowing for both single and batch captures. It can be used as a command-line 
utility or as a Python library.

## Features

-   Capture single or multiple webpages.
-   Supports Chromium and Firefox browsers.
-   Configurable viewport size, timeouts, and wait times.
-   Ability to capture full-scrollable pages.
-   Automatic download and setup of browser binaries.
-   Usable as a standalone CLI or as a library in your own projects.

## Installation

This project uses Poetry for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abasagoitia/webvoyeur.git
    cd webvoyeur
    ```

2.  **Create a virtual environment and install dependencies:**
    Poetry will automatically create a virtual environment within the project folder (`.venv`) and install the required 
    dependencies. The first time you run it, it will also download the necessary Playwright browser binaries.
    ```bash
    poetry install
    ```

3.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

## How to Build

You can build a distributable wheel file for the project using Poetry.

1.  **Build the package:**
    ```bash
    poetry build
    ```
    This command will create the package in the `dist/` directory.

2.  **Install the wheel:**
    You can then install the generated `.whl` file using pip.
    ```bash
    pip install dist/*.whl
    ```

## Command-Line Usage

WebVoyeur provides a command-line interface for easy captures.

### Capture a Single URL

Use the `single` command to capture one webpage.

**Options:**

-   `--url, -u`: The URL to capture.
-   `--filename, -f`: (Optional) The output filename.
-   `--wait-time, -t`: (Optional) Seconds to wait before capturing. Default is 2.
-   `--scroll, -s`: (Optional) Capture the full scrollable page.
-   `--browser, -b`: (Optional) Browser to use: `chrome` or `firefox`. Default is `firefox`.
-   `--output, -o`: (Optional) Output directory. Default is `./output`.

### Capture a Batch of URLs

Use the `batch` command to capture multiple URLs from a file.

1.  Create a file (e.g., `urls.txt`) with one URL per line:
    ```
    https://www.google.com
    https://www.bing.com
    https://www.yahoo.com
    ```

2.  Run the `batch` command:
    ```bash
    webvoyeur batch --urls-txt urls.txt
    ```

**Options:**

-   `--urls-txt`: Path to a text file containing URLs.
-   `--urls-csv`: Path to a CSV file containing URLs (Note yet implemented).
-   `--max-workers, -j`: (Optional) Number of concurrent workers. Default is 4.

## API Usage

You can integrate WebVoyeur into your Python projects using the `Peeker` class. It is recommended to use it as a 
context manager to ensure browser resources are handled correctly.

### Initialize Peeker

Import `Peeker` and `BrowserType` to get started.

```python
from pathlib import Path 
from webvoyeur.peeker import Peeker 
from webvoyeur.utilities import BrowserType

output_directory = Path("./screenshots")
```

### Capture a Single URL

Use the `capture_single` method.

```python
# Use as a context manager

with Peeker(output_dir=output_directory, browser=BrowserType.chromium) as peeker: 
    screenshot_path = peeker.capture_single("https://www.google.com") # Capture a single URL

    if screenshot_path: 
        print(f"Screenshot saved to: {screenshot_path}")
```

### Capture a Batch of URLs

Use the `capture_batch` method for concurrent captures.

```python
urls_to_capture = ["https://www.google.com", "https://www.bing.com", "https://www.yahoo.com"]

# Use as a context manager with 4 workers

with Peeker(output_dir=output_directory, max_workers=4) as peeker:
    results = peeker.capture_batch(urls_to_capture, scroll=False)
    
    for url, path in results.items():
        if path:
            print(f"Successfully captured {url} -> {path}")
        else:
            print(f"Failed to capture {url}")
```

## Development

To install development dependencies, run:

```bash
poetry install
```

### Running Tests

This project uses `pytest`. To run the test suite:

```bash
poetry run pytest
```