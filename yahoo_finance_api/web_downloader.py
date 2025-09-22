"""Utility helpers to scan a web page and download linked resources.

This module provides a small helper that fetches a HTML page, extracts all
links (``<a href=...>`` tags) and downloads the linked resources to a local
folder.  The functionality is intentionally lightweight and only relies on the
Python standard library plus the `requests` package that is already used by the
project.

Example
-------

>>> from yahoo_finance_api.web_downloader import download_links_from_page
>>> download_links_from_page("https://example.com", "./downloads")
['downloads/index.html', 'downloads/license.txt']

The function returns the list of downloaded file paths which makes it easy to
inspect the result or perform additional processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Set
from urllib.parse import urljoin, urlparse, unquote

import os

import requests


class _LinkParser(HTMLParser):
    """HTML parser that collects ``href`` attributes from anchor tags."""

    def __init__(self) -> None:
        super().__init__()
        self.links: Set[str] = set()

    def handle_starttag(self, tag: str, attrs: Iterable[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        for attr, value in attrs:
            if attr.lower() == "href" and value:
                self.links.add(value)


@dataclass(frozen=True)
class DownloadedFile:
    """Represents a single downloaded file."""

    source_url: str
    destination: Path

    def __str__(self) -> str:  # pragma: no cover - human readable helper
        return str(self.destination)


def _extract_links(base_url: str, html: str) -> Set[str]:
    """Extract and normalize hyperlinks from *html*.

    Parameters
    ----------
    base_url:
        URL of the HTML page which is used to resolve relative links.
    html:
        HTML markup as returned by :func:`requests.Response.text`.

    Returns
    -------
    set[str]
        A set of absolute URLs pointing to linked resources.
    """

    parser = _LinkParser()
    parser.feed(html)
    links = set()
    for href in parser.links:
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme in {"http", "https"}:
            links.add(absolute)
    return links


def _filename_from_url(url: str) -> str:
    """Derive a reasonable file name from *url*.

    The function inspects the URL path and query string.  If the URL does not
    contain a file name, a generic name is generated to avoid collisions.
    """

    parsed = urlparse(url)
    path_name = Path(unquote(parsed.path)).name
    if not path_name:
        path_name = parsed.netloc.replace(":", "_")

    if parsed.query:
        safe_query = parsed.query.replace("/", "_").replace("&", "_")
        if path_name:
            path_name = f"{path_name}_{safe_query}"
        else:
            path_name = safe_query

    if not path_name:
        path_name = "downloaded_file"

    return path_name


def _ensure_unique_path(directory: Path, filename: str) -> Path:
    """Return a path that does not clobber existing files."""

    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        candidate = directory / new_name
        if not candidate.exists():
            return candidate
        counter += 1


def _download_file(session: requests.Session, url: str, destination: Path, chunk_size: int = 8192) -> DownloadedFile:
    """Stream *url* to *destination* using *session*."""

    response = session.get(url, stream=True, timeout=30)
    response.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive chunks
                file_handle.write(chunk)

    return DownloadedFile(source_url=url, destination=destination)


def download_links_from_page(url: str, destination_dir: str | os.PathLike[str]) -> List[str]:
    """Download all links referenced by the HTML page at *url*.

    Parameters
    ----------
    url:
        Address of the web page to scan.
    destination_dir:
        Directory on the local filesystem where downloaded files should be
        stored.  The directory is created automatically if it does not already
        exist.

    Returns
    -------
    list[str]
        A list of absolute paths to the downloaded files.

    Notes
    -----
    Only ``http`` and ``https`` resources are downloaded.  Duplicate links are
    ignored and files are saved under unique names to prevent accidental
    overwrites.
    """

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    response = session.get(url, timeout=30)
    response.raise_for_status()

    links = _extract_links(url, response.text)

    downloaded_files: List[str] = []
    for link in sorted(links):
        filename = _filename_from_url(link)
        path = _ensure_unique_path(destination, filename)
        downloaded = _download_file(session, link, path)
        downloaded_files.append(str(downloaded.destination))

    return downloaded_files


def main() -> None:  # pragma: no cover - convenience CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Download all links from a web page.")
    parser.add_argument("url", help="URL of the web page to scan")
    parser.add_argument("destination", help="Directory where files should be stored")
    args = parser.parse_args()

    try:
        files = download_links_from_page(args.url, args.destination)
    except requests.RequestException as exc:  # pragma: no cover - user feedback path
        parser.exit(1, f"Error downloading data: {exc}\n")

    for file_path in files:
        print(file_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
