# Flow Segmenter

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Flow](https://img.shields.io/badge/Project-Flow-green.svg)](https://flow-project.net)

A Python package for segmenting XML files through automatic text structure recognition using YOLO.

## 🌟 Features

- 🎯 **Automatic Segmentation** - Automatically recognizes text structures
- 🔍 **Multiple Backends** - Supports YOLO and Kraken (for baseline and linemask detection)
- 📐 **Baseline Processing** - Intelligent baseline extraction
- 🛠️ **XML Utilities** - Comprehensive XML processing functions
- ✅ **Validation** - Robust input validation with Pydantic

## 🚀 Installation

### Standard Installation

```bash
# Clone repository
git clone https://github.com/The-Flow-Project/package-segmenter.git
cd package-segmenter

# Install
make install
```

### Development Installation

```bash
# With all dev dependencies
make install-dev

# Or with pip
pip install -e ".[dev]"
```

### With uv (recommended for faster installation)

```bash
uv pip install -e .
```

## 💻 Usage

### Simple Example

```python
from flow_segmenter import SegmenterYOLO
from flow_segmenter.config import SegmenterConfig

from lxml import etree

# Create configuration
config = SegmenterConfig(
    model_names=["Riksarkivet/yolov9-regions-1", "Riksarkivet/yolov9-lines-within-regions-1"],
    order_lines=True,
    baselines=True,
)

# Initialize segmenter
segmenter = SegmenterYOLO(config)

# Process XML file
result = segmenter.segment(
    image_path="path/to/image.jpg",
)

print(f"Segmentation completed: {result}")
print(etree.tostring(result, pretty_print=True, encoding="unicode"))
```

## 🧪 Testing and Linting

Have a look at the `Makefile` for various development commands, like:

```bash
make test
```

Tests created with Claude Sonnet 4.5.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add and run tests
4. Format code (`make format`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Flow Project Website](https://flow-project.net)
- [GitHub Repository](https://github.com/The-Flow-Project/package-segmenter)
- [Issue Tracker](https://github.com/The-Flow-Project/package-segmenter/issues)

## 📊 Changelog

### v0.1.5 (2025-12)

- ✨ Complete refactoring OOP structure
- 🔧 Improved configuration with Pydantic
- 🚀 Performance optimization with spatial indexing
- 🧪 Comprehensive test suite
- 📖 Extended documentation
- 🛠️ Makefile for development workflow

### v0.1.0

- 🎉 Initial release
- 🎯 Basic segmentation functionality
- 📐 Baseline extraction
- 🗺️ XML processing
