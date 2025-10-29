# Repository Guidelines

## Project Structure & Module Organization

This repository contains materials for Stanford's CS224N course Assignment 1, focusing on word vector exploration using Python and Jupyter notebooks.

```
.
├── README.txt                    # Setup instructions
├── env.yml                      # Conda environment dependencies
├── exploring_word_vectors.ipynb # Main assignment notebook
├── corpora/                     # Data directory
│   └── reuters.zip              # Reuters news corpus
├── imgs/                        # Images used in notebook
│   ├── test_plot.png
│   ├── svd.png
│   └── inner_product.png
└── .ipynb_checkpoints/          # Jupyter notebook checkpoints
    └── exploring_word_vectors-checkpoint.ipynb
```

## Build, Test, and Development Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f env.yml

# Activate environment
conda activate cs224n

# Install IPython kernel
python -m ipykernel install --user --name cs224n

# Run the notebook
jupyter notebook exploring_word_vectors.ipynb
```

### Running Tests
Tests are embedded within the Jupyter notebook as code cells with assertions. Execute these cells to verify implementation correctness.

## Coding Style & Naming Conventions

- Python 3.7+ required
- Use snake_case for function and variable names
- Use descriptive docstrings for all functions
- Constants should be in UPPER_CASE
- Import statements grouped at the beginning of cells
- Comments should be in both English and Chinese as demonstrated in the notebook

### Example Function Naming
```python
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words
            num_corpus_words (integer): number of distinct words
    """
    # implementation
```

## Testing Guidelines

- Tests are implemented as assertions within the notebook
- Each implementation should include a sanity check
- Test cases follow the format of `test_*` variables
- All tests in the notebook should pass before submission

## Commit & Pull Request Guidelines

Based on the Git history:

- Commit messages should be descriptive and include date
- Use both English and Chinese in commit messages when relevant
- Pull requests should reference the issue number when applicable

### Example Commit Messages
- `2025-10-26 update`
- `更新文档，添加torch.gather函数的详细解释及示例，修改部分文档更新时间`
- `Backup/pre clean (#4)`
