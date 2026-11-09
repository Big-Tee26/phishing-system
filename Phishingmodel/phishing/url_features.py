import re
import numpy as np
from urllib.parse import urlparse

KEYWORDS = ['login', 'verify', 'update', 'secure', 'account', 'bank',
            'confirm', 'password', 'signin', 'paypal', 'bonus', 'free']

FEATURE_NAMES = ['url_length', 'host_length', 'path_length', 'dots_in_url',
                 'dots_in_host', 'hyphens_in_url', 'hyphens_in_host', 'has_at',
                 'question_marks', 'equals', 'percent_signs', 'digits',
                 'is_https', 'ip_host', 'suspicious_words']


def _one(url):
    url = str(url).strip()
    try:
        p = urlparse(url if '://' in url else 'http://' + url)
        host, path = (p.hostname or ''), p.path
    except ValueError:  # malformed URL such as "http://[bad"
        host, path = '', ''
    low = url.lower()
    return [
        len(url), len(host), len(path),
        url.count('.'), host.count('.'),
        url.count('-'), host.count('-'),
        int('@' in url), url.count('?'), url.count('='), url.count('%'),
        sum(c.isdigit() for c in url),
        int(low.startswith('https')),
        int(bool(re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', host))),
        sum(w in low for w in KEYWORDS),
    ]


def extract_matrix(urls):
    """Turn a list/Series of URLs into a (n_urls, n_features) array."""
    return np.array([_one(u) for u in urls], dtype=float)
