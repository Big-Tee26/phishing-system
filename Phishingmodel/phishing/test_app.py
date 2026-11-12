"""Run with:  python -m unittest test_app -v"""
import unittest
from app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def scan(self, url):
        r = self.client.post('/api/predict', json={'url': url})
        return r.status_code, r.get_json()

    def test_trusted_domains_are_safe(self):
        for u in ['https://www.google.com/', 'google.com',
                  'https://mail.google.com/mail/u/0/#inbox',
                  'https://docs.google.com/document/d/1AbC/edit',
                  'https://github.com/Big-Tee26']:
            code, d = self.scan(u)
            self.assertEqual(code, 200)
            self.assertEqual(d['verdict'], 'safe', u)

    def test_same_result_with_or_without_scheme(self):
        _, a = self.scan('google.com')
        _, b = self.scan('https://www.google.com/')
        self.assertEqual(a['verdict'], b['verdict'])

    def test_lookalike_domains_are_not_trusted(self):
        for u in ['https://accounts.google.com.verify-session.top/signin',
                  'http://paypal.com@evil-login.xyz/confirm',
                  'https://evilgoogle.com/login',
                  'https://google.com/redirect?to=http://evil.top/login']:
            _, d = self.scan(u)
            self.assertFalse(d['trusted'], u)

    def test_obvious_phishing_is_flagged(self):
        _, d = self.scan('http://paypal-verify-login.secure-update.xyz/account/confirm')
        self.assertEqual(d['verdict'], 'phishing')

    def test_bad_input_is_rejected(self):
        for u in ['', '   ', 'javascript:alert(1)', 'has space.com', 'localhost', 'a' * 2500]:
            code, d = self.scan(u)
            self.assertEqual(code, 400, u)
            self.assertIn('error', d)


if __name__ == '__main__':
    unittest.main()
