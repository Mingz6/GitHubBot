import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import time
import uuid
from typing import Dict, List, Tuple, Set, Any

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
    
    return text.strip()

def parse_knowledge_file(file_content: str) -> Dict[str, Any]:
    """
    Parse a knowledge file to extract custom information and website URLs.
    
    The file format is expected to be:
    - Custom knowledge statements (plain text)
    - URLs optionally followed by a max page number (e.g., "https://example.com 20")
    
    Args:
        file_content: String content of the uploaded file
        
    Returns:
        Dictionary containing parsed information:
        {
            "id": unique identifier for this knowledge base,
            "content": list of plain text statements,
            "websites": list of {url, max_pages} dictionaries,
            "content_preview": text preview for display,
            "line_count": total number of lines,
            "url_count": number of URLs found
        }
    """
    # Generate a unique ID for this knowledge base
    kb_id = str(uuid.uuid4())
    
    # Clean and split the content into lines
    lines = clean_text(file_content).split('\n')
    
    # Initialize storage
    knowledge_content = []
    websites = []
    
    # URL pattern matches "https://example.com" or "https://example.com 20"
    url_pattern = r'^(https?://\S+)(?:\s+(\d+))?$'
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if the line is a URL possibly followed by a number
        url_match = re.match(url_pattern, line)
        
        if url_match:
            url = url_match.group(1)
            max_pages = int(url_match.group(2)) if url_match.group(2) else 1  # Default to 1 if not specified
            
            if is_valid_url(url):
                websites.append({
                    "url": url,
                    "max_pages": max_pages
                })
        else:
            # If not a URL, treat as knowledge content
            knowledge_content.append(line)
    
    # Create a content preview (limit to first 3 lines and 200 chars)
    preview_text = '\n'.join(knowledge_content[:3])
    if len(preview_text) > 200:
        preview_text = preview_text[:197] + '...'
        
    # Add ellipsis if there are more than 3 lines
    if len(knowledge_content) > 3:
        preview_text += '\n...'
    
    return {
        "id": kb_id,
        "content": knowledge_content,
        "websites": websites,
        "content_preview": preview_text,
        "line_count": len(lines),
        "url_count": len(websites)
    }

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
        print(f"Starting extraction from website: {url} (max pages: {max_pages})")
        # Initialize variables for crawling
        domain = get_domain(url)
        visited_urls: Set[str] = set()
        pages_to_visit: List[str] = [url]
        extracted_content: Dict[str, Dict] = {}
        page_count = 0
        
        # Add headers to avoid being blocked by some websites
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }
        
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
                print(f"Fetching page: {current_url}")
                # Fetch and parse the page
                response = requests.get(current_url, timeout=15, headers=headers)
                if response.status_code != 200:
                    print(f"Failed to fetch {current_url} - Status code: {response.status_code}")
                    continue
                    
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                body = soup.find('body')
                if not body:
                    print(f"No body tag found in {current_url}")
                    continue
                
                # First, remove script and style elements
                for script_or_style in body.find_all(['script', 'style', 'noscript', 'iframe']):
                    script_or_style.decompose()
                
                # Get the page title
                page_title = soup.title.string if soup.title else current_url
                
                # Extract text from the main content, preserving structure
                page_text = ""
                
                # Try to find important content sections
                main_content_areas = []
                
                # Look for main content areas by common ID/class names
                for content_selector in ['main', 'article', 'section', 'div#content', 'div.content', 
                                         'div#main', 'div.main', '.main-content', '#main-content',
                                         '.page-content', '#page-content', '.content-area']:
                    elements = soup.select(content_selector)
                    if elements:
                        main_content_areas.extend(elements)
                
                # Add all headings with their text for better context and search
                headings = body.find_all(['h1', 'h2', 'h3', 'h4'])
                for heading in headings:
                    page_text += f"{heading.get_text().strip()}\n\n"
                
                # Extract text from content areas if found
                if main_content_areas:
                    for content_area in main_content_areas:
                        content_text = content_area.get_text(separator='\n', strip=True)
                        if content_text:
                            page_text += content_text + "\n\n"
                else:
                    # If no specific content areas found, extract from the body
                    page_text += body.get_text(separator='\n', strip=True)

                # Extract content from definition lists, tables and other structured elements
                # which often contain important information like fees, numbers, etc.
                
                # Process definition lists (dl/dt/dd) which often contain key terms and definitions
                dl_elements = soup.find_all('dl')
                for dl in dl_elements:
                    dt_elements = dl.find_all('dt')
                    dd_elements = dl.find_all('dd')
                    
                    for i in range(min(len(dt_elements), len(dd_elements))):
                        term = dt_elements[i].get_text(strip=True)
                        definition = dd_elements[i].get_text(strip=True)
                        if term and definition:
                            page_text += f"{term}: {definition}\n\n"
                
                # Process tables which often contain structured information
                tables = soup.find_all('table')
                for table in tables:
                    table_text = ""
                    # Get table headers if available
                    headers = []
                    th_elements = table.find_all('th')
                    if th_elements:
                        headers = [th.get_text(strip=True) for th in th_elements]
                    
                    # Process table rows
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if cells:
                            row_text = " | ".join([cell.get_text(strip=True) for cell in cells])
                            table_text += row_text + "\n"
                    
                    if table_text:
                        page_text += f"Table content:\n{table_text}\n\n"
                
                # Process lists (ul/ol) which often contain important bullet points
                lists = soup.find_all(['ul', 'ol'])
                for list_element in lists:
                    items = list_element.find_all('li')
                    list_text = ""
                    for item in items:
                        item_text = item.get_text(strip=True)
                        if item_text:
                            list_text += f"• {item_text}\n"
                    
                    if list_text:
                        page_text += f"List content:\n{list_text}\n\n"
                
                # Look for fee-related information specifically (since you're looking for "troll free/fee")
                fee_elements = soup.find_all(string=re.compile(r'(?i)(fee|fees|free|cost|payment|toll|troll)'))
                fee_info = []
                for element in fee_elements:
                    # Get the parent element for better context
                    parent = element.parent
                    if parent:
                        context = parent.get_text(strip=True)
                        if context:
                            fee_info.append(context)
                
                if fee_info:
                    page_text += "Fee-related information:\n" + "\n".join(fee_info) + "\n\n"
                
                # Clean the extracted text
                clean_content = clean_text(page_text)
                
                # Store the extracted content if it's substantial enough
                if clean_content and len(clean_content) > 50:  # Only store if there's sufficient content
                    extracted_content[current_url] = {
                        "title": clean_text(page_title),
                        "content": clean_content
                    }
                    page_count += 1
                    print(f"Extracted {len(clean_content)} characters from {current_url}")
                
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
                        # Skip login pages and other irrelevant pages
                        if not any(excluded in full_url.lower() for excluded in ['login', 'signin', 'signup', 'register', 'account']):
                            # Prioritize pages with keywords that might have fee information
                            priority = 0
                            link_text = link.get_text().lower()
                            if any(term in link_text for term in ['fee', 'fees', 'cost', 'price', 'payment', 'toll', 'troll']):
                                # Add high-priority links to the front of the queue
                                pages_to_visit.insert(0, full_url)
                                print(f"Found priority link: {full_url}")
                            else:
                                # Add regular links to the end
                                pages_to_visit.append(full_url)
                            link_count += 1
                
                # Be gentle to the website - small delay between requests
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing {current_url}: {str(e)}")
                continue
        
        if not extracted_content:
            return {"error": "No content could be extracted from the website"}
        
        print(f"Crawl completed: Visited {len(visited_urls)} pages, extracted content from {len(extracted_content)} pages")
        return extracted_content
        
    except Exception as e:
        print(f"Error extracting website content: {str(e)}")
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
                  "of", "and", "or", "not", "what", "where", "when", "who", "how"}
    question_keywords = question_words - stop_words
    
    print(f"Extracted keywords from question: {question_keywords}")
    
    # Handle common misspellings and variants for key terms
    keyword_variations = {
        "fee": ["fee", "fees", "cost", "costs", "price", "prices", "payment"],
        "troll": ["troll", "toll", "trolls", "tolls"],
        "free": ["free", "freely", "no cost", "no fee", "without cost"],
        "number": ["number", "contact", "phone", "telephone", "hotline", "call"]
    }
    
    # Expand question keywords with variations
    expanded_keywords = set(question_keywords)
    for keyword in question_keywords:
        for base_word, variations in keyword_variations.items():
            if keyword in variations or keyword == base_word:
                expanded_keywords.update(variations)
                break
    
    print(f"Expanded keywords include: {expanded_keywords}")
    
    # Find relevant sections from each page
    relevant_sections = []
    source_urls = []
    
    for url, page_data in content.items():
        if isinstance(page_data, dict) and "error" in page_data:
            continue
        
        # Handle different page_data formats
        if isinstance(page_data, dict) and "content" in page_data:
            page_text = page_data["content"]
            page_title = page_data.get("title", url)
        elif isinstance(page_data, str):
            page_text = page_data
            page_title = url
        else:
            continue
            
        # Split into paragraphs
        paragraphs = re.split(r'\n+', page_text)
        
        # Store paragraphs with their relevance scores
        scored_paragraphs = []
        
        for paragraph in paragraphs:
            # Check if paragraph is long enough
            if len(paragraph) < 15:  # Reduced minimum length to catch short phrases like "toll free: 123-456-7890"
                continue
                
            paragraph_lower = paragraph.lower()
            
            # Count exact phrase matches for higher precision
            exact_phrase_score = 0
            if len(question) > 5:
                for i in range(2, 5):  # Try 2-4 word phrases from question
                    question_phrases = extract_ngrams(question.lower(), i)
                    for phrase in question_phrases:
                        if phrase in paragraph_lower:
                            exact_phrase_score += 2  # Increased weight for exact phrase matches
            
            # Count keyword matches with expanded variations
            paragraph_words = set(re.findall(r'\w+', paragraph_lower))
            matching_keywords = paragraph_words.intersection(expanded_keywords)
            
            # Check for special patterns like "toll free" or "toll-free number"
            special_patterns = [
                r'(?:toll|troll)[\s-]*free',  # Matches "toll free", "toll-free", "troll free", etc.
                r'(?:toll|troll)[\s-]*free[\s-]*(?:number|line|phone|contact)',  # Matches "toll free number", etc.
                r'(?:fee|cost|price)[\s-]*(?:structure|information|details)'  # Matches "fee structure", "price information" etc.
            ]
            
            pattern_matches = 0
            for pattern in special_patterns:
                if re.search(pattern, paragraph_lower):
                    pattern_matches += 3  # High score for special pattern matches
            
            # Calculate relevance score based on keyword matches, density and exact phrases
            score = 0
            if len(expanded_keywords) > 0:
                # Base score from percentage of matched keywords
                keyword_match_ratio = len(matching_keywords) / len(expanded_keywords)
                # Bonus for high density of keywords
                keyword_density = len(matching_keywords) / max(1, len(paragraph_words))
                # Bonus for exact phrase matches and special patterns
                phrase_bonus = min(1.0, (exact_phrase_score * 0.2) + (pattern_matches * 0.3))
                score = (keyword_match_ratio * 0.5) + (keyword_density * 0.3) + phrase_bonus
            
            # Only consider paragraphs with sufficient relevance
            if score > 0.1 or len(matching_keywords) >= 1 or exact_phrase_score > 0 or pattern_matches > 0:
                if url not in source_urls:
                    source_urls.append(url)
                scored_paragraphs.append((paragraph, score))
        
        # Sort paragraphs by relevance score (descending)
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        # Add page title as context for better retrieval
        if scored_paragraphs:
            relevant_sections.append(f"From page: {page_title}")
            
        # Take only the top relevant paragraphs (max 5 per page)
        for para, score in scored_paragraphs[:5]:
            relevant_sections.append(para)
    
    # Limit the total number of sections to prevent token limit issues
    if len(relevant_sections) > 20:  # Increased from 15 to 20 to allow for more context
        relevant_sections = relevant_sections[:20]
    
    # Combine relevant sections
    combined_content = "\n\n".join(relevant_sections)
    
    if not combined_content:
        print("No relevant content found matching the question")
        return "", []
    
    print(f"Found {len(relevant_sections)} relevant sections from {len(source_urls)} pages")
    return combined_content, source_urls

def extract_ngrams(text, n):
    """Extract n-grams from text for better phrase matching"""
    words = re.findall(r'\b\w+\b', text.lower())
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
