import re
from urllib.parse import urlparse

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import url_features  # noqa: F401  (needed so joblib can load the pipeline)

app = Flask(__name__)

# One file holds the features AND the model, so they can never be mismatched.
# Labels: phishing = 1, legitimate = 0.
model = joblib.load('phishing_pipeline.joblib')

KEYWORDS = ['login', 'verify', 'update', 'secure', 'account', 'bank',
            'confirm', 'password', 'signin', 'paypal', 'bonus', 'free']


def red_flags(url):
    parsed = urlparse(url if '://' in url else 'http://' + url)
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
    url = str(data.get('url', '')).strip()[:2000]
    if not url or ' ' in url:
        return jsonify(error='Please enter a valid URL without spaces.'), 400
    p = float(model.predict_proba(pd.DataFrame({'url': [url]}))[0][1])
    verdict = 'phishing' if p >= 0.7 else 'suspicious' if p >= 0.4 else 'safe'
    return jsonify(url=url, verdict=verdict,
                   phishing_probability=round(p * 100, 1),
                   flags=red_flags(url))


if __name__ == '__main__':
    app.run(debug=False)
