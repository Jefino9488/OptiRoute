import os
import json
import urllib.request
from html.parser import HTMLParser

class SidebarParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_nav = False
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'nav' and 'aria-label' in attrs_dict and attrs_dict['aria-label'] == 'Pages':
            self.in_nav = True
        
        if self.in_nav and tag == 'a' and 'href' in attrs_dict:
            href = attrs_dict['href']
            if href.startswith('/'):
                self.links.append('https://docs.fireworks.ai' + href)

    def handle_endtag(self, tag):
        if tag == 'nav' and self.in_nav:
            self.in_nav = False

def fetch_and_save():
    print("Fetching introduction page...")
    req = urllib.request.Request("https://docs.fireworks.ai/getting-started/introduction", headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    
    parser = SidebarParser()
    parser.feed(html)
    
    # Filter out duplicate links, but keep order
    links = []
    seen = set()
    for link in parser.links:
        if link not in seen:
            seen.add(link)
            links.append(link)
            
    print(f"Found {len(links)} links in the sidebar.")
    
    os.makedirs('reference', exist_ok=True)
    
    # We will fetch up to 15 key pages to avoid taking too long
    for link in links[:15]:
        filename = link.split('/')[-1]
        if not filename:
            filename = link.split('/')[-2]
        filename = filename + '.html'
        filepath = os.path.join('reference', filename)
        
        print(f"Fetching {link}...")
        try:
            req = urllib.request.Request(link, headers={'User-Agent': 'Mozilla/5.0'})
            content = urllib.request.urlopen(req).read()
            with open(filepath, 'wb') as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to fetch {link}: {e}")

if __name__ == '__main__':
    fetch_and_save()
