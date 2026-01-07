import io
import logging
import re
from typing import List

import httpx
from bs4 import BeautifulSoup
from refinedoc.refined_document import RefinedDocument
from urllib3 import Retry

logger = logging.getLogger(__name__)

HEADERS = {
    "Accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "TE": "Trailers",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/136.0",
}


def delete_non_printable_character(text: str) -> str:
    """
    Delete non-printable characters from a text

    :param text: the text to clean

    :return: the cleaned text
    """
    return "".join([c for c in text if c.isprintable()])


def replace_ligatures(text: str) -> str:
    """
    Replace ligatures in text by their equivalent

    :param text: the text to clean

    :return: the cleaned text
    """
    ligatures = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        # "Ꜳ": "AA",
        # "Æ": "AE",
        "ꜳ": "aa",
    }
    for search, replace in ligatures.items():
        text = text.replace(search, replace)
    return text


def remove_extra_whitespace(text: str) -> str:
    """removes extra whitespace from text

    Args:
        text (str): text to evaluate

    Returns:
        str: text without extra whitespace
    """
    if not isinstance(text, str):
        return text
    return " ".join(text.split())


def delete_accents(text: str) -> str:
    """
    Delete accents in text

    :param text: the text to clean

    :return: the cleaned text
    """
    accents = {
        "´": "",
        "`": "",
        "ˆ": "",
        "˜": "",
        "¸": "",
        "˚": "",
        "¨": "",
        "˝": "",
        "˛": "",
        "˙": "",
        "ˇ": "",
        "˘": "",
    }
    for search, replace in accents.items():
        local_search_items = [f" {search}", f"{search} ", f"{search}"]
        for search_item in local_search_items:
            text = text.replace(search_item, replace)
    return text


def remove_hyphens(text: str) -> str:
    """
    This fails for:
    * Natural dashes: well-known, self-replication, use-cases, non-semantic,
                      Post-processing, Window-wise, viewpoint-dependent
    * Trailing math operands: 2 - 4
    * Names: Lopez-Ferreras, VGG-19, CIFAR-100
    """
    lines = [line.rstrip() for line in text.split("\n")]

    # Find dashes
    line_numbers = []
    for line_no, line in enumerate(lines[:-1]):
        if line.endswith("-"):
            line_numbers.append(line_no)

    # Replace
    for line_no in line_numbers:
        lines = _dehyphenate(lines, line_no)

    return "\n".join(lines)


def _dehyphenate(lines: List[str], line_no: int) -> List[str]:
    """
    Dehyphenate a line in a list of lines

    :param lines: the list of lines
    :param line_no: the line number to dehyphenate

    :return: the list of lines with the dehyphenated line
    """
    next_line = lines[line_no + 1]
    word_suffix = next_line.split(" ")[0]

    lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = lines[line_no + 1][len(word_suffix) :]
    return lines


def get_new_https_async_client(retry_total: int = 10) -> httpx.AsyncClient:
    # Define the retry strategy
    retry_strategy = Retry(
        total=retry_total,  # Maximum number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        backoff_factor=2,  # Factor to apply to the backoff
    )

    transport = httpx.AsyncHTTPTransport(retries=retry_total)

    # Create a new session object
    client = httpx.AsyncClient(
        transport=transport,
        headers=HEADERS,  # type: ignore
    )
    client.headers = HEADERS  # type: ignore
    return client


async def _send_pdf_to_tika(pdf_content: io.BytesIO, tika_base_url: str) -> dict:
    """
    Send a PDF document to Tika micro service and return the content as a dictionary
    :param pdf_content: the PDF document content as a byte stream
    :param tika_base_url: the base URL of the Tika micro service
    :return: the content returned by Tika micro service as a dictionary (JSON)
    """
    tika_base_url = re.sub(r"\/$", "", tika_base_url)
    pdf_process_addr = f"{tika_base_url}/tika"
    local_headers = {
        "Accept": "application/json",
        "Content-type": "application/octet-stream",
        "X-Tika-PDFOcrStrategy": "no_ocr",
    }

    async with get_new_https_async_client() as http_session:
        logger.info(f"Sending PDF to Tika micro service at {pdf_process_addr}")
        resp = await http_session.put(
            url=pdf_process_addr,
            files={"file": pdf_content},
            headers=local_headers,
        )
        resp.raise_for_status()
        tika_content = resp.json()

    return tika_content


def _parse_tika_content(tika_content: dict) -> list[list[str]]:
    """
    Parse the content returned by Tika micro service
    :param tika_content: the content returned by Tika micro service
    :return: the parsed content as a list of list of strings (one list per page
    """
    htmlx = tika_content.get("X-TIKA:content")
    soup = BeautifulSoup(htmlx, features="html.parser")
    pages = soup.find_all("div", {"class": "page"})
    res = [p.split("\n") for p in [page.get_text() for page in pages]]

    return res


async def extract_txt_from_pdf_with_tika(
    pdf_content: io.BytesIO, tika_base_url: str
) -> str:
    """
    Extract the text from a PDF document and return it as a list of strings for each page of the document and a list of
    strings for each page for a filtered document and the reference document (extracted with tika micro service)

    :param pdf_content: the PDF document content as a byte stream
    :param tika_base_url: the base URL of the Tika micro service
    :param with_metadata: if True, return a tuple with the refined document and the metadata extracted by Tika
    :return: Matrix of strings (list of list of strings) for each page of the document or a tuple with the refined
             body and the metadata extracted by Tika
    """
    tika_content = await _send_pdf_to_tika(pdf_content, tika_base_url)
    parsed_pdf_cont = _parse_tika_content(tika_content)

    refined_pdf_content = RefinedDocument(content=parsed_pdf_cont)

    pdf_cont = [
        [delete_non_printable_character(word) for word in page]
        for page in refined_pdf_content.body
    ]

    pages = []
    for content in pdf_cont:
        page_text = " ".join(content)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)
        page_text = delete_accents(page_text)

        pages.append(page_text)
    ret = remove_extra_whitespace(" ".join(pages))

    return ret
