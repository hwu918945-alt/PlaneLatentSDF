"""CLI dispatch; each command also supports python -m plane_auto_occ.<module>."""
import argparse
import runpy
import sys

COMMANDS = {
    'prepare-run': 'prepare_run', 'train': 'train', 'evaluate': 'evaluate',
    'extract-mesh': 'extract_mesh', 'diagnostics': 'diagnostics',
    'area-pool': 'area_pool', 'convert-cache': 'convert_cache',
    'make-synthetic': 'make_synthetic', 'audit-splits': 'audit_splits',
    'deploy': 'deployment', 'test': 'tests', 'test-repair': 'repair_tests',
    'test-migration': 'migration_tests',
}


def main():
    parser = argparse.ArgumentParser(prog='plane-auto-occ', description='PlaneAutoOccupancy standalone commands')
    parser.add_argument('command', choices=COMMANDS)
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    module = 'plane_auto_occ.' + COMMANDS[args.command]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
