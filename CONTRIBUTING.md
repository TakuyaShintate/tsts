# 🐛 CONTRIBUTING

We have a code of conduct. Please follow it in all your interactions with the project.

## Naming Conventions

In the following subsections, naming conventions adopted in this project are listed.

### Functions (or methods) starting with `infer_*`

Functions (or methods) which return value(s) implicitly determined by given inputs are named `infer_*`.

### Custom Type Hint

If List or Tuple is nested, the custom type hint must be defined for the readability of the code. Custom type hints are defined in `types.py`.

If List, Tuple or Union contains more than 2 elements, the custom type hint must be defined.
