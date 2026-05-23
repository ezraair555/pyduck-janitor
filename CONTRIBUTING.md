# Contributing to pyduck-janitor

Thank you for considering contributing to pyduck-janitor! We welcome contributions from the community to make this package as useful and efficient as possible.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## How to Contribute

### Reporting Bugs

If you find a bug, please [submit an issue](https://github.com/yourusername/pyduck-janitor/issues) with:

1. A clear description of the problem
2. A minimal, reproducible example
3. Your Python version and package version
4. Expected vs. actual behavior
5. DuckDB version

### Suggesting Features

We welcome feature suggestions! Please open an issue with:

1. A clear description of the feature
2. Why it would be useful
3. Example usage if possible
4. Whether it exists in pyjanitor (and how it works)

### Pull Requests

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format your code (`black .`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/pyduck-janitor.git
cd pyduck-janitor

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black pyduck_janitor examples tests
```

## Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints where possible
- Write docstrings in NumPy style
- Keep functions focused and single-purpose
- Write tests for new features
- Use DuckDB SQL for operations when possible (for performance)

## Testing

We use `pytest` for testing. Please:

1. Write tests for new features
2. Ensure existing tests pass
3. Test with both small and large datasets
4. Test edge cases (empty data, all nulls, etc.)

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyduck_janitor

# Run specific test file
pytest tests/test_duck_janitor.py
```

## Documentation

- Update docstrings for all public functions and classes
- Add examples to docstrings
- Update README.md if adding new features
- Consider adding examples to the `examples/` directory

## Architecture Notes

### How pyduck-janitor Works

1. **DuckJanitor wraps DuckDB relations** - Data lives in DuckDB, not pandas
2. **Methods translate to SQL** - Each cleaning operation becomes DuckDB SQL
3. **Lazy evaluation** - Operations build a query plan without executing
4. **Optimized execution** - DuckDB optimizes and executes the full pipeline
5. **Pandas output** - `.collect()` converts results back to pandas

### Adding New Cleaning Operations

To add a new cleaning operation:

1. **Add function to `cleaning_ops.py`**:
   ```python
   def my_operation(relation: duckdb.DuckDBPyRelation, ...) -> duckdb.DuckDBPyRelation:
       conn = relation.database
       # Implement using SQL
       query = "SELECT ... FROM relation"
       return conn.execute(query)
   ```

2. **Add method to `DuckJanitor` class**:
   ```python
   def my_operation(self, ...) -> 'DuckJanitor':
       from .cleaning_ops import my_operation
       new_relation = my_operation(self._relation, ...)
       return DuckJanitor(new_relation, self._connection)
   ```

3. **Export in `__init__.py`**:
   ```python
   from .cleaning_ops import my_operation
   __all__ = [..., 'my_operation']
   ```

4. **Write tests** in `tests/`

5. **Add example** to `examples/` (optional but encouraged)

## Performance Considerations

When contributing, keep these in mind:

- **Prefer SQL operations** - They're optimized by DuckDB
- **Avoid pandas conversions** - They defeat the purpose of out-of-core processing
- **Test with large data** - Ensure operations work on 1M+ row datasets
- **Lazy evaluation** - Don't execute until `.collect()` is called

## Questions and Discussions

For questions about the package, DuckDB, or how to use pyduck-janitor:

1. Check existing issues first
2. Open a new issue with the "question" label
3. Include a minimal example if applicable

## License

By contributing to pyduck-janitor, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

This package is inspired by:

- [pyjanitor](https://pyjanitor-devs.github.io/pyjanitor/) - Original data cleaning API
- [duckplyr](https://duckplyr.tidyverse.org/) - DuckDB-backed dplyr
- [DuckDB](https://duckdb.org/) - High-performance analytical database
