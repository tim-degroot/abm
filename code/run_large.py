"""Top-level wrapper that forwards to tools.run_large.run_large.

Keeps a simple entrypoint at the project root while moving implementation
into `tools/` so the code/ folder stays tidy.
"""

from tools.run_large import run_large

if __name__ == "__main__":
    run_large()
