"""Server MCP per previsioni e analisi di serie temporali con Timestral."""

__version__ = "0.1.0"


def main() -> None:
    from timestral_mcp.server import main as _main

    _main()
