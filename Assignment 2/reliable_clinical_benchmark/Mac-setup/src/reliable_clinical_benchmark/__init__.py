"""
Compatibility package providing the historical ``reliable_clinical_benchmark``
namespace on top of the flattened ``src`` layout.

The actual implementations live in top-level packages such as ``metrics``,
``data`` and ``utils``; tests and external callers that import
``reliable_clinical_benchmark.*`` continue to work via the lightweight
re-exports defined in the submodules of this package.
"""


