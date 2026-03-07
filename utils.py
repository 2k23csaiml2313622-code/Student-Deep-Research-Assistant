import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults(num_results=5)

def web_search(query):
    results = search_tool.run(query)

    urls = []
    for r in results.split("\n"):
        if "link:" in r:
            urls.append(r.split("link:")[-1].strip())

    return urls

def scrape_website(url):
    response = requests.get(
        url,
        timeout=10,
        headers={"User-Agent":"Mozilla/5.0"}
    )

    soup = BeautifulSoup(response.text,"html.parser")

    for tag in soup(["script","style","nav","footer","header","aside"]):
        tag.decompose()

    return soup.get_text(separator=" ",strip=True)

def read_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text