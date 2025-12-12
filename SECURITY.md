# Security Policy

## Supported Versions

Currently supported versions of PlantDocBot:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do Not** Open a Public Issue

Please do not report security vulnerabilities through public GitHub issues.

### 2. Report Privately

Send an email to: **[your-email@example.com]** with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Response Time

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity

### 4. Disclosure Policy

- We will acknowledge your email within 48 hours
- We will provide a more detailed response within 7 days
- We will work on a fix and keep you updated
- Once fixed, we will publicly disclose the vulnerability (with your permission)

## Security Best Practices

### For Users

1. **Never commit `.env` files** with API keys
2. **Use environment variables** for sensitive data
3. **Keep dependencies updated** regularly
4. **Use HTTPS** in production
5. **Implement rate limiting** for production APIs

### For Contributors

1. **Review code** for security issues before submitting PRs
2. **Avoid hardcoding** secrets or credentials
3. **Validate all inputs** from users
4. **Sanitize file uploads** properly
5. **Follow OWASP** security guidelines

## Known Security Considerations

### API Keys

- Gemini API keys should be kept private
- Use environment variables, never hardcode
- Rotate keys regularly

### File Uploads

- Only accept image files (JPEG, PNG)
- Validate file types and sizes
- Scan for malicious content in production

### CORS

- Configure CORS properly for production
- Only allow trusted origins
- Don't use wildcard (*) in production

## Security Updates

Security updates will be released as patch versions and announced via:

- GitHub Security Advisories
- Release notes
- Email notifications (for critical issues)

## Bug Bounty

Currently, we do not have a bug bounty program. However, we greatly appreciate responsible disclosure and will acknowledge contributors in our security hall of fame.

---

**Thank you for helping keep PlantDocBot secure!** 🔒
