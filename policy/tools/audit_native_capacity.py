"""Read complete externally captured native logs for invalid contact-capacity diagnostics."""
import argparse
import json
from pathlib import Path
import re

# Preserve normal CUDA kernel-load lines and unrelated LM shared-memory
# fallback messages. Capacity overflow messages themselves invalidate the run.
FORBIDDEN = re.compile(
    r'(?:broadphase|narrowphase|nefc|njmax|nconmax|naconmax|contact|constraint|candidate)'
    r'.{0,120}(?:over\s*flow|exceed(?:ed|s)?|too many|insufficient capacity|capacity exhausted)'
    r'|(?:over\s*flow|exceed(?:ed|s)?|too many|insufficient capacity|capacity exhausted)'
    r'.{0,120}(?:broadphase|narrowphase|nefc|njmax|nconmax|naconmax|contact|constraint|candidate)',
    re.IGNORECASE)


def audit(paths):
    reports = []
    for path in paths:
        path = Path(path).resolve()
        matches = []
        with path.open(errors='replace') as stream:
            for number, line in enumerate(stream, 1):
                if FORBIDDEN.search(line):
                    matches.append(dict(line=number, diagnostic=line.rstrip()[:800]))
        reports.append(dict(path=str(path), bytes=path.stat().st_size,
                            forbidden_diagnostics=matches))
    return dict(capacity_log_check_passed=all(not item['forbidden_diagnostics'] for item in reports),
        complete_log_paths=reports, online_policy_input=False,
        scope='External complete-log read only. A clean result does not prove physical success or complete log capture; any matching capacity diagnostic invalidates qualification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.logs)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['capacity_log_check_passed'] else 1)
