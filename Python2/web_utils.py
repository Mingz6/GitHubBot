import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import time
from typing import Dict, List, Tuple, Set

def is_valid_url(url: str) -> bool:
    """Check if a URL is valid and accessible."""
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except:
        return False

def get_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""

def normalize_url(url: str, base_url: str) -> str:
    """Convert relative URLs to absolute URLs."""
    try:
        return urljoin(base_url, url)
    except:
        return url

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and special characters."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def extract_webpage_content(url: str, max_pages: int = 5, max_links_per_page: int = 10) -> Dict:
    """
    Extract content from a website by crawling its pages.
    Returns a dictionary with page URLs as keys and their content as values.
    
    Args:
        url: The starting URL
        max_pages: Maximum number of pages to crawl
        max_links_per_page: Maximum number of links to follow from each page
    """
    if not is_valid_url(url):
        return {"error": "Invalid URL format"}
    
    try:
        # Initialize variables for crawling
        domain = get_domain(url)
        visited_urls: Set[str] = set()
        pages_to_visit: List[str] = [url]
        extracted_content: Dict[str, str] = {}
        page_count = 0
        
        # Crawl pages until we reach the maximum or run out of pages
        while pages_to_visit and page_count < max_pages:
            # Get the next URL to visit
            current_url = pages_to_visit.pop(0)
            
            # Skip if we've already visited this URL
            if current_url in visited_urls:
                continue
                
            # Add to visited URLs
            visited_urls.add(current_url)
            
            try:
                # Fetch and parse the page
                response = requests.get(current_url, timeout=10)
                if response.status_code != 200:
                    continue
                    
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                body = soup.find('body')
                if not body:
                    continue
                
                # Remove script and style elements
                for script_or_style in body.find_all(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
                    script_or_style.decompose()
                
                # Get the text content
                page_text = clean_text(body.get_text())
                
                # Store the extracted content
                if page_text and len(page_text) > 100:  # Only store if there's sufficient content
                    page_title = soup.title.string if soup.title else current_url
                    extracted_content[current_url] = {
                        "title": clean_text(page_title),
                        "content": page_text
                    }
                
                # Find links to other pages on the same domain
                links = soup.find_all('a', href=True)
                link_count = 0
                
                for link in links:
                    if link_count >= max_links_per_page:
                        break
                        
                    href = link['href']
                    full_url = normalize_url(href, current_url)
                    
                    # Only follow links to the same domain
                    if get_domain(full_url) == domain and full_url not in visited_urls and full_url not in pages_to_visit:
                        pages_to_visit.append(full_url)
                        link_count += 1
                
                page_count += 1
                
                # Be gentle to the website - small delay between requests
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing {current_url}: {str(e)}")
                continue
        
        if not extracted_content:
            return {"error": "No content could be extracted from the website"}
            
        return extracted_content
        
    except Exception as e:
        return {"error": f"Error extracting website content: {str(e)}"}

def extract_relevant_sections(content: Dict, question: str) -> Tuple[str, List[str]]:
    """
    Extract sections from the website content that are most relevant to the question.
    
    Args:
        content: Dictionary with page URLs as keys and their content as values
        question: The user's question
        
    Returns:
        Tuple containing combined content string and list of source URLs
    """
    # Extract keywords from the question
    question_words = set(re.findall(r'\w+', question.lower()))
    # Filter out common stop words
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been", 
                  "being", "in", "on", "at", "to", "for", "with", "by", "about", 
                  "of", "and", "or", "not"}
    question_keywords = question_words - stop_words
    
    # Find relevant sections from each page
    relevant_sections = []
    source_urls = []
    
    for url, page_data in content.items():
        if "error" in page_data:
            continue
            
        page_text = page_data["content"]
        
        # Split into paragraphs
        paragraphs = re.split(r'\n+', page_text)
        
        for paragraph in paragraphs:
            # Check if paragraph is long enough and contains keywords
            if len(paragraph) < 50:
                continue
                
            paragraph_words = set(re.findall(r'\w+', paragraph.lower()))
            matching_keywords = paragraph_words.intersection(question_keywords)
            
            # If the paragraph contains at least 2 keywords or 30% of keywords, consider it relevant
            if len(matching_keywords) >= 2 or (len(question_keywords) > 0 and len(matching_keywords) / len(question_keywords) >= 0.3):
                relevant_sections.append(paragraph)
                if url not in source_urls:
                    source_urls.append(url)
    
    # Combine relevant sections
    combined_content = "\n\n".join(relevant_sections)
    
    if not combined_content:
        return "", []
        
    return combined_content, source_urls
