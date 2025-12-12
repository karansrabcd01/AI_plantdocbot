# 🤝 Contributing to PlantDocBot

Thank you for your interest in contributing to PlantDocBot! This document provides guidelines and instructions for contributing.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

---

## 🎯 How Can I Contribute?

### 1. Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When reporting a bug, include:**
- Clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, Node version)
- Error messages or logs

**Example:**
```markdown
**Title:** Image prediction fails for PNG files

**Description:**
When uploading a PNG image, the API returns a 500 error.

**Steps to Reproduce:**
1. Navigate to Image Analysis section
2. Upload a PNG file
3. Click "Analyze Image"

**Expected:** Image should be analyzed successfully
**Actual:** Error: "Failed to process image"

**Environment:**
- OS: Windows 11
- Python: 3.10
- Browser: Chrome 120

**Error Log:**
```
Traceback (most recent call last):
  File "main.py", line 123
  ...
```
```

### 2. Suggesting Features

We love feature suggestions! Please provide:
- Clear description of the feature
- Use case / problem it solves
- Proposed implementation (if you have ideas)
- Any relevant examples or mockups

**Example:**
```markdown
**Feature:** Add support for video analysis

**Description:**
Allow users to upload short videos of their plants to detect diseases over time.

**Use Case:**
Some diseases show progressive symptoms that are better captured in video format.

**Proposed Implementation:**
- Extract frames from video
- Run analysis on multiple frames
- Aggregate results
- Show timeline of disease progression
```

### 3. Code Contributions

We welcome code contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

---

## 🛠 Development Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Setup Steps

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/plantdocbot.git
cd plantdocbot

# 2. Create a branch
git checkout -b feature/your-feature-name

# 3. Set up backend
cd Backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Set up frontend
cd ../plantdoc-frontend
npm install

# 5. Run tests (if available)
pytest  # Backend
npm test  # Frontend
```

---

## 💻 Coding Standards

### Python (Backend)

**Style Guide:** PEP 8

```python
# Good
def predict_disease(image_path: str) -> dict:
    """
    Predict plant disease from image.
    
    Args:
        image_path: Path to the plant image
        
    Returns:
        Dictionary containing prediction results
    """
    # Implementation
    pass

# Bad
def predict(img):
    # No docstring, unclear parameter
    pass
```

**Best Practices:**
- Use type hints
- Write docstrings for functions
- Keep functions focused and small
- Handle errors gracefully
- Add comments for complex logic

### JavaScript/React (Frontend)

**Style Guide:** Airbnb JavaScript Style Guide

```javascript
// Good
const ImageUpload = ({ onUpload }) => {
  const [file, setFile] = useState(null);
  
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    onUpload(selectedFile);
  };
  
  return (
    <input 
      type="file" 
      onChange={handleFileChange}
      accept="image/*"
    />
  );
};

// Bad
function imgupload(props) {
  // Inconsistent naming, unclear structure
  return <input type="file" onChange={e=>props.upload(e.target.files[0])} />
}
```

**Best Practices:**
- Use functional components with hooks
- Destructure props
- Use meaningful variable names
- Keep components small and focused
- Add PropTypes or TypeScript

### CSS

```css
/* Good - BEM naming */
.chat-message {
  padding: 1rem;
}

.chat-message__bubble {
  background: white;
}

.chat-message__bubble--user {
  background: blue;
}

/* Bad - unclear naming */
.msg {
  padding: 1rem;
}

.blue {
  background: blue;
}
```

---

## 📝 Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
# Good commits
git commit -m "feat(chatbot): add support for plant care topics"
git commit -m "fix(image): handle PNG transparency correctly"
git commit -m "docs(api): update endpoint documentation"

# Bad commits
git commit -m "fixed stuff"
git commit -m "updates"
git commit -m "asdfasdf"
```

### Detailed Example

```
feat(chatbot): add conversation history support

- Store last 5 messages for context
- Pass history to API endpoint
- Update UI to show conversation flow

Closes #123
```

---

## 🔄 Pull Request Process

### Before Submitting

1. **Update your fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Test your changes**
   - Run all tests
   - Test manually in browser
   - Check for console errors

3. **Update documentation**
   - Update README if needed
   - Add API docs for new endpoints
   - Update CHANGELOG

4. **Clean up commits**
   ```bash
   git rebase -i HEAD~3  # Squash related commits
   ```

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tested locally
- [ ] Added/updated tests
- [ ] All tests passing

## Screenshots (if applicable)
Add screenshots here

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. **Submit PR** with clear description
2. **Automated checks** must pass
3. **Code review** by maintainers
4. **Address feedback** if requested
5. **Approval** and merge

---

## 🐛 Reporting Bugs

### Security Vulnerabilities

**DO NOT** open public issues for security vulnerabilities.

Instead, email: security@plantdocbot.com (or create private security advisory)

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What should happen

**Screenshots**
If applicable

**Environment:**
- OS: [e.g., Windows 11]
- Python Version: [e.g., 3.10]
- Node Version: [e.g., 18.0]
- Browser: [e.g., Chrome 120]

**Additional Context**
Any other relevant information
```

---

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Problem It Solves**
What problem does this address?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Mockups, examples, etc.
```

---

## 🏆 Recognition

Contributors will be:
- Listed in README
- Mentioned in release notes
- Given credit in documentation

---

## 📞 Questions?

- Open a discussion on GitHub
- Check existing issues and PRs
- Review documentation

---

## 🙏 Thank You!

Every contribution, no matter how small, is valuable. Thank you for helping make PlantDocBot better!

---

**Happy Contributing! 🌿✨**
