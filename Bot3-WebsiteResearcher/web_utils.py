import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import time
import json
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
        extracted_content: Dict[str, dict] = {}
        page_count = 0
        max_retries = 3  # Maximum number of retry attempts
        retry_delay = 2  # Initial delay between retries in seconds

        # Common patterns for contact information
        phone_pattern = r'\b(?:\+?1[-.]?)?\s*(?:\([0-9]{3}\)|[0-9]{3})[-.]?\s*[0-9]{3}[-.]?\s*[0-9]{4}\b'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Crawl pages until we reach the maximum or run out of pages
        while pages_to_visit and page_count < max_pages:
            # Get the next URL to visit
            current_url = pages_to_visit.pop(0)
            
            # Skip if we've already visited this URL
            if current_url in visited_urls:
                continue
                
            # Add to visited URLs
            visited_urls.add(current_url)
            
            # Initialize retry counter
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    # Fetch and parse the page with timeout
                    response = requests.get(current_url, timeout=10, 
                                         headers={'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'})
                    
                    # Handle specific status codes
                    if response.status_code == 503:
                        retry_count += 1
                        if retry_count < max_retries:
                            # Exponential backoff
                            sleep_time = retry_delay * (2 ** (retry_count - 1))
                            print(f"Received 503 error for {current_url}, retrying in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                            continue
                        else:
                            print(f"Max retries reached for {current_url}")
                            raise requests.exceptions.RequestException(f"Service Unavailable (503) after {max_retries} retries")
                    
                    elif response.status_code == 429:  # Too Many Requests
                        # Get retry-after header or use default
                        retry_after = int(response.headers.get('Retry-After', retry_delay))
                        print(f"Rate limited, waiting {retry_after} seconds...")
                        time.sleep(retry_after)
                        retry_count += 1
                        continue
                        
                    elif response.status_code != 200:
                        print(f"HTTP {response.status_code} error for {current_url}")
                        break
                    
                    # Parse the HTML content
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract text content
                    body = soup.find('body')
                    if not body:
                        print(f"No body tag found in {current_url}")
                        break
                    
                    # Remove script and style elements
                    for script_or_style in body.find_all(['script', 'style', 'noscript']):
                        script_or_style.decompose()

                    # Extract structured data (Schema.org)
                    structured_data = []
                    for script in soup.find_all('script', type='application/ld+json'):
                        try:
                            data = json.loads(script.string)
                            structured_data.append(data)
                        except:
                            continue

                    # Extract meta tags
                    meta_tags = {}
                    for meta in soup.find_all('meta'):
                        name = meta.get('name', meta.get('property', ''))
                        content = meta.get('content', '')
                        if name and content:
                            meta_tags[name] = content

                    # Extract contact information
                    contact_info = {
                        'phones': [],
                        'emails': [],
                        'addresses': []
                    }

                    # Look for contact information in specific elements first
                    contact_elements = body.find_all(['address', 'footer', 'header'], 
                                                   class_=['contact', 'footer', 'header'],
                                                   id=['contact', 'footer', 'header'])
                    
                    for element in contact_elements:
                        # Extract phones
                        phones = re.findall(phone_pattern, str(element))
                        contact_info['phones'].extend(phones)
                        
                        # Extract emails
                        emails = re.findall(email_pattern, str(element))
                        contact_info['emails'].extend(emails)
                        
                        # Look for address elements
                        address_elements = element.find_all('address')
                        for addr in address_elements:
                            contact_info['addresses'].append(clean_text(addr.get_text()))

                    # Then look in the entire body
                    all_text = body.get_text()
                    phones = re.findall(phone_pattern, all_text)
                    emails = re.findall(email_pattern, all_text)
                    contact_info['phones'].extend(phones)
                    contact_info['emails'].extend(emails)

                    # Remove duplicates while preserving order
                    contact_info['phones'] = list(dict.fromkeys(contact_info['phones']))
                    contact_info['emails'] = list(dict.fromkeys(contact_info['emails']))
                    
                    # Get the text content
                    page_text = clean_text(body.get_text())
                    
                    # Store the extracted content with enhanced metadata
                    if page_text:
                        page_title = soup.title.string if soup.title else current_url
                        extracted_content[current_url] = {
                            "title": clean_text(page_title),
                            "content": page_text,
                            "meta_tags": meta_tags,
                            "structured_data": structured_data,
                            "contact_info": contact_info,
                            "html_content": str(body)
                        }
                    
                    # Find links to other pages on the same domain
                    links = soup.find_all('a', href=True)
                    contact_links = []
                    link_count = 0
                    
                    for link in links:
                        href = link['href']
                        text = link.get_text().lower()
                        full_url = normalize_url(href, current_url)
                        
                        # Prioritize contact pages
                        if any(word in text for word in ['contact', 'about', 'location']):
                            contact_links.append(full_url)
                        
                        # Only follow links to the same domain
                        if get_domain(full_url) == domain and full_url not in visited_urls and full_url not in pages_to_visit:
                            if link_count < max_links_per_page:
                                pages_to_visit.append(full_url)
                                link_count += 1
                    
                    # Add contact pages to the beginning of pages_to_visit
                    for contact_url in reversed(contact_links):
                        if contact_url not in visited_urls and contact_url not in pages_to_visit:
                            pages_to_visit.insert(0, contact_url)
                    
                    page_count += 1
                    
                    # Successful request, break retry loop
                    break
                    
                except requests.exceptions.Timeout:
                    print(f"Timeout error for {current_url}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(retry_delay)
                    last_error = "Timeout"
                    
                except requests.exceptions.ConnectionError:
                    print(f"Connection error for {current_url}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(retry_delay)
                    last_error = "Connection Error"
                    
                except Exception as e:
                    print(f"Error processing {current_url}: {str(e)}")
                    last_error = str(e)
                    break
                    
            # Be gentle to the website - add delay between pages
            time.sleep(1)
        
        if not extracted_content:
            error_msg = f"No content could be extracted from the website. Last error: {last_error}" if last_error else "No content could be extracted from the website"
            return {"error": error_msg}
        
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
    question_lower = question.lower()
    question_words = set(re.findall(r'\w+', question_lower))
    
    # Filter out common stop words
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been", 
                  "being", "in", "on", "at", "to", "for", "with", "by", "about", 
                  "of", "and", "or", "not"}
    question_keywords = question_words - stop_words
    
    # Check if question is about contact information
    contact_keywords = {'phone', 'contact', 'email', 'address', 'location', 'number', 'call'}
    is_contact_query = bool(question_keywords.intersection(contact_keywords))
    
    # Find relevant sections from each page
    relevant_sections = []
    source_urls = []
    contact_info_found = {}
    
    # Set stricter limits
    max_sections = 5  # Maximum number of relevant sections to include
    max_section_chars = 4000  # Maximum characters per section
    total_chars = 0
    max_total_chars = 15000  # Maximum total characters across all sections
    
    for url, page_data in content.items():
        if not isinstance(page_data, dict):
            continue
        
        # Process contact information first if relevant
        if is_contact_query and 'contact_info' in page_data:
            contact_info = page_data['contact_info']
            
            for info_type, info_list in contact_info.items():
                if not isinstance(info_list, list):
                    continue
                    
                for info in info_list[:2]:  # Limit to 2 items per type
                    if info not in contact_info_found.get(info_type, []):
                        contact_info_found.setdefault(info_type, []).append(info)
                        relevant_sections.append(f"{info_type.title()}: {info}")
                        if url not in source_urls:
                            source_urls.append(url)
        
        # Process regular content if we haven't hit the total character limit
        if total_chars < max_total_chars and 'content' in page_data:
            page_text = page_data['content']
            paragraphs = re.split(r'\n+', page_text)
            
            for paragraph in paragraphs:
                if len(paragraph) < 50:  # Skip very short paragraphs
                    continue
                    
                # Calculate relevance score
                paragraph_words = set(re.findall(r'\w+', paragraph.lower()))
                matching_keywords = paragraph_words.intersection(question_keywords)
                
                score = 0
                if len(question_keywords) > 0:
                    keyword_match_ratio = len(matching_keywords) / len(question_keywords)
                    keyword_density = len(matching_keywords) / max(1, len(paragraph_words))
                    contact_bonus = 0.3 if is_contact_query and paragraph_words.intersection(contact_keywords) else 0
                    score = (keyword_match_ratio * 0.5) + (keyword_density * 0.3) + contact_bonus
                
                # Only include highly relevant sections
                if score > 0.3 or len(matching_keywords) >= 2:
                    # Truncate section if needed
                    if len(paragraph) > max_section_chars:
                        paragraph = paragraph[:max_section_chars] + "..."
                    
                    # Check if adding this section would exceed total limit
                    if total_chars + len(paragraph) > max_total_chars:
                        break
                    
                    if paragraph not in relevant_sections:
                        relevant_sections.append(paragraph)
                        total_chars += len(paragraph)
                        if url not in source_urls:
                            source_urls.append(url)
                    
                    # Stop if we have enough sections
                    if len(relevant_sections) >= max_sections:
                        break
        
        # Stop processing pages if we have enough content
        if len(relevant_sections) >= max_sections or total_chars >= max_total_chars:
            break
    
    # If no relevant sections found but we have contact info, use that
    if not relevant_sections and contact_info_found:
        combined_info = []
        for info_type, info_list in contact_info_found.items():
            combined_info.extend([f"{info_type.title()}: {info}" for info in info_list[:2]])
        return "\n".join(combined_info), source_urls
    
    return "\n\n".join(relevant_sections), source_urls
