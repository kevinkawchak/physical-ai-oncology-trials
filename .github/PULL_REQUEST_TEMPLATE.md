## Summary

Brief description of changes.

## Change Type

- [ ] New integration or pipeline
- [ ] Bug fix
- [ ] Documentation update
- [ ] Framework version update
- [ ] Unification / cross-platform tooling
- [ ] Regulatory or privacy tooling
- [ ] Benchmark or results update

## Checklist

### Code Quality
- [ ] Code passes `ruff check` with no errors
- [ ] Code passes `ruff format --check` with no errors
- [ ] YAML files are valid (if modified)
- [ ] New dependencies added to `requirements.txt` with pinned versions

### Documentation
- [ ] Updated relevant README or integration guide
- [ ] Results and benchmark claims include citations or are labeled as illustrative
- [ ] Framework versions and dates are accurate

### Safety and Compliance (if applicable)
- [ ] No PHI, PII, or patient data included in the PR
- [ ] No hardcoded credentials, API keys, or tokens
- [ ] Changes to regulatory/privacy tools have been reviewed for compliance
- [ ] Human oversight requirements documented for any automated clinical workflow

### Testing
- [ ] `python scripts/verify_installation.py` passes
- [ ] Cross-framework conversion tested (if modifying unification tools)
- [ ] Example scripts run without error (if modifying examples)

## Related Issues

Closes #
