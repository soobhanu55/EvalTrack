"""`evaltrack-gate run|check|report <scorer_module> ...`: one entry point over the three stages."""
from __future__ import annotations

import sys

from evaltrack_gate import regression, report, runner

COMMANDS = {"run": runner.main, "check": regression.main, "report": report.main}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"usage: evaltrack-gate {{{'|'.join(COMMANDS)}}} <scorer_module> [--history FILE] ...", file=sys.stderr)
        return 2
    command = sys.argv.pop(1)
    sys.argv[0] = f"evaltrack-gate {command}"
    return COMMANDS[command]()


if __name__ == "__main__":
    sys.exit(main())
