# Contributing to **Credit Risk Assessment**

Thank you for considering contributing to this project!
Improvements to code, documentation, tests, and project structure are welcome. This guide outlines how to get started and the standards to follow.

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [How to contribute](#how-to-contribute)
3. [Development workflow](#development-workflow)
4. [Coding standards](#coding-standards)
5. [Commit conventions](#commit-conventions)
6. [Pull request guidelines](#pull-request-guidelines)
7. [Issues and feature requests](#issues-and-feature-requests)
8. [License](#license)

---

## Code of conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
By participating, you agree to maintain a respectful, inclusive, and professional environment.

---

## How to contribute

There are many ways to contribute:

* Fix bugs or improve stability
* Add tests or improve coverage
* Enhance documentation (README, inline docstrings, tutorials)
* Improve data preprocessing or model evaluation
* Suggest or implement new visualizations
* Optimize Streamlit UI/UX
* Report issues or request features

---

## Development workflow

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/PrincessNosawema/credit_risk_assessment
   cd credit_risk_assessment
   ```
3. **Create a branch** for your change:
   ```bash
   git checkout -b feature/my-feature
   ```
4. **Set up environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   pip install -r requirements.txt
   ```
5. **Make changes** and test locally:
   * To run the Streamlit app:

     ```bash
     streamlit run app.py
     ```
   * To run the ML pipeline:

     ```bash
     python credit_risk_assessment.py
     ```
6. **Commit** your changes (see [Commit conventions](#commit-conventions)).
7. **Push** your branch and open a **Pull Request** on GitHub.

---

## Coding standards

* **Python version**: 3.8+
* **Style**: follow [PEP 8](https://peps.python.org/pep-0008/) as closely as possible.
* **Imports**:

  * Standard library → third-party → local modules (with blank lines in between).
* **Type hints**: use Python type hints where practical:

  ```python
  def predict_default(probability: float) -> str:
      ...
  ```
* **Docstrings**: use clear, concise [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings for functions and classes. Example:

  ```python
  def load_dataset(path: str) -> pd.DataFrame:
      """
      Load dataset from a CSV file.

      Args:
          path (str): Path to the dataset file.

      Returns:
          pd.DataFrame: Loaded dataset.
      """
  ```
* **Testing**: when adding new functionality, consider adding unit tests (`pytest` is recommended).

---

## Commit conventions

Please use conventional commits to keep history clean and meaningful:

```
<type>(<scope>): <short description>
```

**Types**:

* `feat`: new feature
* `fix`: bug fix
* `docs`: documentation only changes
* `style`: formatting, missing semi colons, etc. (no code change)
* `refactor`: code change that neither fixes a bug nor adds a feature
* `perf`: performance improvement
* `test`: adding or fixing tests
* `chore`: changes to build process, tooling, dependencies

**Examples**:

* `feat(model): add XGBoost training option`
* `fix(preprocessing): handle empty string values in categorical columns`
* `docs(readme): update installation instructions`

---

## Pull request guidelines

* Keep PRs focused on a single change.
* Update documentation if behavior or usage changes.
* Ensure code runs without errors:

  * `python credit_risk_assessment.py` should complete end-to-end.
  * `streamlit run app.py` should start without errors.
* Check formatting and linting before submission.

---

## Issues and feature requests

* Use GitHub [Issues](../../issues) to report bugs or request features.
* Before submitting, search existing issues to avoid duplicates.
* When reporting bugs, please include:

  * OS and Python version
  * Steps to reproduce
  * Expected vs actual behavior
  * Any relevant error messages

---

## License

By contributing, you agree that your contributions will be licensed under the same [MIT LICENSE](LICENSE) as the project.