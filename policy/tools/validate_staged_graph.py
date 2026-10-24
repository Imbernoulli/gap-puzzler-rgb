"""Validate a frozen project graph in the pinned native GaP CLI.

The pinned CLI does not interpret the project's schema/registry metadata. This
read-only adapter loads the graph's own schema extension, resolves its declared
registries and then delegates validation to that CLI. It never executes tools.
"""
import argparse
from pathlib import Path
import sys

from puzzler.vision_policy import load_graph_schema, required_skill_registries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    args = parser.parse_args()
    graph = args.graph.resolve()
    if graph.is_file():
        graph = graph.parent
    load_graph_schema(graph)
    registries = required_skill_registries(graph)
    native_args = [sys.argv[0], 'run', str(graph), '--validate-only']
    for registry in registries or []:
        native_args.extend(['--skills', str(registry)])
    sys.argv = native_args
    from gap.cli import main as native_main
    native_main()


if __name__ == '__main__':
    main()
