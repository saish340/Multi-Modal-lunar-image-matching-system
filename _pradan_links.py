import sys, re, urllib.request
from urllib.parse import urljoin

def links(url):
    html = urllib.request.urlopen(url, timeout=25).read().decode("utf-8", "ignore")
    out = [urljoin(url, l) for l in re.findall(r'href=["\']([^"\']+)["\']', html)]
    # dedupe preserving order
    seen = set(); res = []
    for l in out:
        if l not in seen and not l.startswith("mailto"):
            seen.add(l); res.append(l)
    return html, res

url = sys.argv[1]
html, ls = links(url)
print("URL:", url, "html_len:", len(html))
for l in ls:
    print("  ", l)
