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
    """
    Clean extracted text by removing extra whitespace and normalizing formatting.
    Prevents issues with character spacing and formatting that can cause RAG output problems.
    """
    if not text:
        return ""
    
    # First, normalize newlines
    text = text.replace('\r\n', '\n')
    
    # Fix common spacing issues where text is s p a c e d  o u t
    # This pattern looks for single character followed by space pattern
    spaced_out_pattern = r'(?<=[a-zA-Z])\s(?=[a-zA-Z](\s[a-zA-Z])+)'
    while re.search(spaced_out_pattern, text):
        text = re.sub(spaced_out_pattern, '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newlines to preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up any remaining non-printable or control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Convert any remaining unicode to ASCII equivalent where possible, or remove if not possible
    text = ''.join(c if ord(c) < 128 else ' ' for c in text)
    
    return text.strip()

def extract_webpage_content(url: str, max_pages: int = 10, max_links_per_page: int = 10) -> Dict:
    """
    Extract content from a website by crawling its pages.
    Returns a dictionary with page URLs as keys and their content as values.
    
    Args:
        url: The starting URL
        max_pages: Maximum number of pages to crawl (default: 10)
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
                
                # Print progress update every 10 pages
                if page_count % 10 == 0:
                    print(f"Crawled {page_count} pages. Found {len(extracted_content)} pages with content.")
                
            except Exception as e:
                print(f"Error processing {current_url}: {str(e)}")
                continue
        
        if not extracted_content:
            return {"error": "No content could be extracted from the website"}
        
        print(f"Crawl completed: Visited {len(visited_urls)} pages, extracted content from {len(extracted_content)} pages.")
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
        if isinstance(page_data, dict) and "error" in page_data:
            continue
        
        # Handle different page_data formats
        if isinstance(page_data, dict) and "content" in page_data:
            page_text = page_data["content"]
        elif isinstance(page_data, str):
            page_text = page_data
        else:
            continue
            
        # Split into paragraphs
        paragraphs = re.split(r'\n+', page_text)
        
        # Store paragraphs with their relevance scores
        scored_paragraphs = []
        
        for paragraph in paragraphs:
            # Check if paragraph is long enough
            if len(paragraph) < 50:
                continue
                
            paragraph_words = set(re.findall(r'\w+', paragraph.lower()))
            matching_keywords = paragraph_words.intersection(question_keywords)
            
            # Calculate relevance score based on keyword matches and density
            score = 0
            if len(question_keywords) > 0:
                # Base score from percentage of matched keywords
                keyword_match_ratio = len(matching_keywords) / len(question_keywords)
                # Bonus for high density of keywords
                keyword_density = len(matching_keywords) / max(1, len(paragraph_words))
                score = (keyword_match_ratio * 0.7) + (keyword_density * 0.3)
            
            # Only consider paragraphs with sufficient relevance
            if score > 0.2 or len(matching_keywords) >= 2:
                scored_paragraphs.append((paragraph, score))
        
        # Sort paragraphs by relevance score (descending)
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        # Take only the top relevant paragraphs (max 3 per page)
        for para, score in scored_paragraphs[:3]:
            relevant_sections.append(para)
            if url not in source_urls:
                source_urls.append(url)
    
    # Limit the total number of sections to prevent token limit issues
    if len(relevant_sections) > 10:
        relevant_sections = relevant_sections[:10]
    
    # Combine relevant sections
    combined_content = "\n\n".join(relevant_sections)
    
    if not combined_content:
        return "", []
        
    return combined_content, source_urls
