# Deployment and Release Pipeline Guide

## 1. Release Versioning
ChakrView adheres to Semantic Versioning (SemVer 2.0.0).
Major version bumps occur upon breaking changes to the Python API or tokenizer vocabulary.
Minor versions introduce backwards-compatible features such as new retrievers or cache algorithms.

## 2. Supported Python Versions
ChakrView strictly requires Python 3.10 through Python 3.14. Python 3.9 and older versions are officially unsupported and deprecated.

## 3. Dependency Pinning Policy
- Core dependencies (torch, numpy) are pinned to minimum verified LTS versions.
- External web scraping packages and unverified cloud vector libraries are forbidden in the core runtime.
- Build artifacts are verified against pre-computed SHA-256 digests prior to distribution.
