# lova2026
A memory-efficient optimizer that matches SGD with momemtum

# Usage

This optimizer can be used in your project just as easy as AdamW. We will include more examples after cleaning up the codebase.

```python
from lova import Lova

optimizer = Lova(net.parameters(), lr=3.0e-4, weight_decay=0.1, betas=(0.9, 0.99))
```

