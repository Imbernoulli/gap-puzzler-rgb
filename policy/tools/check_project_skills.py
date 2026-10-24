"""Check graph-owned skill contracts in the pinned native GaP CLI.

As with validate_staged_graph, load this graph's own schema extension first.
No tool is invoked and no environment or renderer is created.
"""
import argparse
from pathlib import Path
import sys

from puzzler.vision_policy import load_graph_schema, required_skill_registries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--registry', help='Optional owning registry name to check')
    args = parser.parse_args()
    graph = args.graph.resolve()
    if graph.is_file():
        graph = graph.parent
    load_graph_schema(graph)
    registries = required_skill_registries(graph)
    native_args = [sys.argv[0], 'skills', 'check']
    for registry in registries or []:
        native_args.extend(['--skills', str(registry)])
    if args.registry:
        native_args.extend(['--registry', args.registry])
    sys.argv = native_args
    from gap.cli import main as native_main
    native_main()


if __name__ == '__main__':
    main()
