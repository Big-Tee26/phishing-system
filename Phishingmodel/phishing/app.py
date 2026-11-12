import re
from urllib.parse import urlparse

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import url_features  # noqa: F401  (needed so joblib can load the pipeline)

app = Flask(__name__)

# One file holds the features AND the model. Labels: phishing = 1, legitimate = 0.
model = joblib.load('phishing_pipeline.joblib')

# Well-known domains. A host matches if it equals one of these or is a
# subdomain of it ("mail.google.com" matches, "google.com.evil.top" does not).
TRUSTED = {
    'google.com', 'youtube.com', 'github.com', 'microsoft.com', 'apple.com',
    'amazon.com', 'wikipedia.org', 'linkedin.com', 'facebook.com', 'instagram.com',
    'x.com', 'twitter.com', 'stackoverflow.com', 'whatsapp.com', 'paypal.com',
    'netflix.com', 'zoom.us', 'dropbox.com', 'mozilla.org', 'python.org',
    'kaggle.com', 'anthropic.com', 'claude.ai',
}
# Trusted sites where anyone can publish content, so a link can still be bad.
SHARED_CONTENT = {'docs.google.com', 'drive.google.com', 'sites.google.com',
                  'forms.gle', 'github.com'}

KEYWORDS = ['login', 'verify', 'update', 'secure', 'account', 'bank',
            'confirm', 'password', 'signin', 'paypal', 'bonus', 'free']
BAD_SCHEMES = ('javascript:', 'data:', 'file:', 'vbscript:', 'mailto:')


def normalise(raw):
    """Clean the input so 'google.com' and 'https://www.google.com/' are treated alike."""
    url = str(raw or '').strip()
    if not url or ' ' in url:
        raise ValueError('Please enter a valid URL without spaces.')
    if len(url) > 2000:
        raise ValueError('That URL is too long.')
    if url.lower().startswith(BAD_SCHEMES):
        raise ValueError('Only http and https links can be scanned.')
    if '://' not in url:
        url = 'https://' + url
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or '').lower()
    except ValueError:
        raise ValueError('That does not look like a valid URL.')
    if parsed.scheme not in ('http', 'https') or '.' not in host:
        raise ValueError('Please enter a full web address, like example.com.')
    return parsed._replace(netloc=parsed.netloc.lower()).geturl(), host


def matches(host, domains):
    return any(host == d or host.endswith('.' + d) for d in domains)


def is_trusted(url, host):
    # A trusted domain does not make a link safe if it hides another URL or an @ trick
    return matches(host, TRUSTED) and '@' not in url and url.count('://') == 1


def red_flags(url):
    parsed = urlparse(url)
    host = parsed.hostname or ''
    flags = []
    if url.lower().startswith('http://'):
        flags.append('Uses HTTP instead of HTTPS')
    if re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', host):
        flags.append('Uses an IP address instead of a domain name')
    if '@' in url:
        flags.append('Contains an @ symbol, which can hide the real destination')
    if host.count('.') >= 3:
        flags.append('Has many subdomains')
    if host.count('-') >= 2:
        flags.append('Has many hyphens in the domain')
    if len(url) > 75:
        flags.append('Unusually long URL')
    hits = [w for w in KEYWORDS if w in url.lower()]
    if hits:
        flags.append('Contains sensitive words: ' + ', '.join(hits[:4]))
    return flags


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        url, host = normalise(data.get('url', ''))
    except ValueError as e:
        return jsonify(error=str(e)), 400

    p = float(model.predict_proba(pd.DataFrame({'url': [url]}))[0][1])
    trusted = is_trusted(url, host)
    note = ''
    if trusted:
        verdict = 'safe'
        note = 'This is a well-known domain.'
        if matches(host, SHARED_CONTENT):
            note += ' It also hosts content made by other people, so check who shared the link.'
    else:
        verdict = 'phishing' if p >= 0.7 else 'suspicious' if p >= 0.4 else 'safe'

    return jsonify(url=url, verdict=verdict, trusted=trusted, note=note,
                   phishing_probability=round(p * 100, 1),
                   flags=red_flags(url))


if __name__ == '__main__':
    app.run(debug=False)