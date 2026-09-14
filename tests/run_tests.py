#!/usr/bin/env python3
"""
run_tests.py - end-to-end tests for verify.py and verify.mjs.

    python3 tests/run_tests.py                       run everything
    python3 tests/run_tests.py --only full-512       only samples whose name contains the text
    python3 tests/run_tests.py --keep-temp           keep the tampered receipts for inspection
    python3 tests/run_tests.py --ts-reference PATH   use this compiled browser verifier as a third opinion

Python 3.9 or newer, standard library only.  node must be on PATH (it runs verify.mjs).
Exit code 0 when every check passes, 1 otherwise.

What is tested
--------------
For every receipt in samples/*.json:

  1. `python3 verify.py <receipt> --json` and `node verify.mjs <receipt> --json` both exit 0
     and reproduce samples/expected/<name>.json: booleans and integers exactly, floats
     within 1e-9 relative or 1e-12 absolute.
  2. The two --json reports agree once parsed: same keys in the same order, booleans and
     integers identical, floats within the tolerance (both scripts print the shortest
     round-trip decimal of each double, so equal doubles give equal text).
  3. Table mode prints the seven checks in order with their plain labels and ends with a
     PASS verdict; --quiet prints the verdict line only; and the two scripts print byte for
     byte the same text in both modes.
  4. When the browser verifier (TypeScript compiled to CommonJS) is present on this
     machine it is run as a third opinion and must agree with the expected report and with
     both ports.  It is not shipped in this repository, so the check is skipped when the
     file is missing.
  5. Tampered copies of every receipt are written to a temporary directory.  Both
     verifiers must reject each one with exit code 1, agree with each other on the whole
     report and on the printed text, name the same failed checks, and agree with the
     browser verifier where it is available.  A truncated file must give exit code 2.
  6. Malformed files (a NaN literal, a UTF-16 file, a missing openings list, an odd-length
     salt, a string where a number belongs) must be refused by both scripts with exit code
     2 and, for shape problems, the same message; a UTF-8 byte-order mark and CRLF line
     endings must be accepted.

One tamper is different by design: changing a single hop of one chain.  DIRECT and
CONNECTED are measured fractions that only fail when *no* challenged pair holds up, so a
single broken chain lowers the passed count by exactly one without flipping the verdict.
The test asserts exactly that, and a second tamper that breaks every chain must exit 1.

A note on the last bit.  The node port runs on the same JavaScript engine as the browser
verifier and must match it bit for bit, always; that is asserted.  The python port carries
V8's fdlibm exp, log and log1p in plain IEEE arithmetic, so its numbers are the same on
every platform, but a JavaScript engine compiled with fused multiply-add (Node on Apple
Silicon, for one) rounds about 1 % of those calls differently in the last bit.  That can
move lowerBound, worstExactErr and the reads band by about 1e-15 relative on some receipts,
never the verdict, the ok flags or any integer.  The cross-port comparison therefore uses
the float tolerance for those fields, and every last-bit difference it tolerates is printed
so that it stays visible.
"""

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_VERIFIER = REPO_ROOT / 'verify.py'
NODE_VERIFIER = REPO_ROOT / 'verify.mjs'
SAMPLES_DIR = REPO_ROOT / 'samples'
EXPECTED_DIR = SAMPLES_DIR / 'expected'

# The browser verifier compiled to CommonJS.  It is the reference implementation and lives
# outside this repository; point --ts-reference or the TS_REFERENCE environment variable at
# it when you have a copy.
DEFAULT_TS_REFERENCE = None  # set --ts-reference or TS_REFERENCE to enable the third opinion

CHECK_NAMES = ['BOUNDED', 'EXACT', 'RANGE', 'HEARDBY', 'DIRECT', 'CONNECTED', 'WEIGHTS']
CHECK_LABELS = {
    'BOUNDED': 'Reads at most B tokens',
    'EXACT': 'Ordinary attention, nothing approximated',
    'RANGE': 'Reaches far back',
    'HEARDBY': 'No token is dropped',
    'DIRECT': 'Read directly (measured fraction)',
    'CONNECTED': 'Linked by a short chain (measured fraction)',
    'WEIGHTS': 'Same model, no retraining',
}
REPORT_FIELDS = ['ok', 'maxDegree', 'worstExactErr', 'meanDegree', 'hoeffding',
                 'readsLower', 'readsUpper', 'readsDense', 'results']
RESULT_FIELDS = ['name', 'ok', 'measured', 'checked', 'passed', 'fraction', 'lowerBound']

RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 1e-12
COMMAND_TIMEOUT_SECONDS = 300


# --------------------------------------------------------------------------
# Bookkeeping: every check lands in the ledger, which prints the summary
# --------------------------------------------------------------------------


class Ledger:
    """Collects PASS / FAIL / SKIP outcomes and prints them as they happen."""

    def __init__(self, verbose):
        self.verbose = verbose
        self.entries = []          # (status, section, check, detail)
        self.section = ''

    def start_section(self, title):
        self.section = title
        print('\n%s' % title)

    def record(self, status, check, detail=''):
        self.entries.append((status, self.section, check, detail))
        line = '  %-4s  %s' % (status, check)
        if detail and (status != 'PASS' or self.verbose):
            line += '\n' + indent_lines(detail)
        print(line)

    def info(self, text):
        """A line of context that is printed but not counted as a check."""
        print('  %-4s  %s' % ('info', text))

    def passed(self, check, detail=''):
        self.record('PASS', check, detail)

    def failed(self, check, detail=''):
        self.record('FAIL', check, detail)

    def skipped(self, check, detail=''):
        self.record('SKIP', check, detail)

    def count(self, status):
        return sum(1 for entry in self.entries if entry[0] == status)

    def print_summary(self):
        failures = [entry for entry in self.entries if entry[0] == 'FAIL']
        print('\n' + '=' * 72)
        print('Summary: %d checks, %d passed, %d failed, %d skipped'
              % (len(self.entries), self.count('PASS'), len(failures), self.count('SKIP')))
        for _, section, check, detail in failures:
            print('  FAIL  %s: %s' % (section, check))
            if detail:
                print(indent_lines(detail, 8))
        print('RESULT: %s' % ('FAIL' if failures else 'PASS'))


def indent_lines(text, spaces=8):
    prefix = ' ' * spaces
    return '\n'.join(prefix + line for line in text.splitlines())


# --------------------------------------------------------------------------
# Running the verifiers
# --------------------------------------------------------------------------


class Verifier:
    """One command-line verifier: a label plus the argv prefix that runs it."""

    def __init__(self, label, prefix):
        self.label = label
        self.prefix = prefix

    def run(self, receipt_path, *flags):
        return run_command(self.prefix + [str(receipt_path)] + list(flags))


def run_command(argv):
    """Run a command with captured output; never raises on a non-zero exit code."""
    return subprocess.run(
        argv, capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=COMMAND_TIMEOUT_SECONDS, cwd=str(REPO_ROOT), check=False,
    )


def describe_run(completed, limit=600):
    """Exit code, stdout and stderr of a finished command, trimmed for a failure message."""
    parts = ['exit code %d' % completed.returncode]
    if completed.stdout.strip():
        parts.append('stdout: ' + completed.stdout.strip()[:limit])
    if completed.stderr.strip():
        parts.append('stderr: ' + completed.stderr.strip()[:limit])
    return '\n'.join(parts)


def parse_report(completed):
    """The --json report of a finished verifier run, or (None, reason)."""
    try:
        return json.loads(completed.stdout), None
    except ValueError as error:
        return None, 'stdout is not JSON (%s)\n%s' % (error, describe_run(completed))


# The browser verifier, run through node.  argv[1] is the compiled module, argv[2] the receipt.
TS_DRIVER = """
const { verify } = require(process.argv[1]);
const text = require('fs').readFileSync(process.argv[2], 'utf8');
process.stdout.write(JSON.stringify(verify(JSON.parse(text))));
"""


class TsReference:
    """The compiled browser verifier as a third opinion; `available` is False when absent."""

    def __init__(self, node, path):
        self.node = node
        self.path = path
        self.available = path is not None and path.is_file()

    def report(self, receipt_path):
        """Projected report of the reference, or (None, reason) when it did not run cleanly."""
        completed = run_command([self.node, '-e', TS_DRIVER, str(self.path), str(receipt_path)])
        if completed.returncode != 0:
            return None, 'reference did not finish\n' + describe_run(completed)
        raw, problem = parse_report(completed)
        if problem:
            return None, problem
        return project_report(raw), None


def project_report(raw):
    """Keep only the public report fields (the browser adds per-check prose and a log)."""
    projected = {field: raw.get(field) for field in REPORT_FIELDS}
    projected['results'] = [
        {field: entry.get(field) for field in RESULT_FIELDS} for entry in raw.get('results', [])
    ]
    return projected


# --------------------------------------------------------------------------
# Comparing parsed JSON reports
# --------------------------------------------------------------------------


def is_bool(value):
    return isinstance(value, bool)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def type_name(value):
    return type(value).__name__


def numbers_close(actual, expected):
    """Integers must match exactly; anything involving a float gets the tolerance."""
    if is_integer(actual) and is_integer(expected):
        return actual == expected
    actual, expected = float(actual), float(expected)
    if math.isnan(actual) or math.isnan(expected):
        return math.isnan(actual) and math.isnan(expected)
    if math.isinf(actual) or math.isinf(expected):
        return actual == expected
    allowed = max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * max(abs(actual), abs(expected)))
    return abs(actual - expected) <= allowed


def numbers_identical(actual, expected):
    """Same type and same value: what two faithful ports must print for the same double."""
    return type(actual) is type(expected) and actual == expected


def compare_values(actual, expected, tolerant, path='report'):
    """Every difference between two parsed JSON values, as human-readable lines.

    tolerant=True applies the float tolerance and ignores key order (comparison against
    samples/expected or the browser reference); tolerant=False demands identical types,
    values and key order (comparison between the two ports).
    """
    if isinstance(expected, dict):
        return compare_objects(actual, expected, tolerant, path)
    if isinstance(expected, list):
        return compare_lists(actual, expected, tolerant, path)
    if is_bool(expected) or is_bool(actual):
        if is_bool(expected) and is_bool(actual) and actual == expected:
            return []
        return ['%s: expected %r, got %r' % (path, expected, actual)]
    if is_number(expected):
        if not is_number(actual):
            return ['%s: expected a number, got %s %r' % (path, type_name(actual), actual)]
        same = numbers_close(actual, expected) if tolerant else numbers_identical(actual, expected)
        return [] if same else ['%s: expected %r, got %r' % (path, expected, actual)]
    if actual != expected:
        return ['%s: expected %r, got %r' % (path, expected, actual)]
    return []


def compare_objects(actual, expected, tolerant, path):
    if not isinstance(actual, dict):
        return ['%s: expected an object, got %s' % (path, type_name(actual))]
    differences = []
    missing = [key for key in expected if key not in actual]
    extra = [key for key in actual if key not in expected]
    if missing:
        differences.append('%s: missing keys %s' % (path, missing))
    if extra:
        differences.append('%s: unexpected keys %s' % (path, extra))
    if not tolerant and not missing and not extra and list(actual) != list(expected):
        differences.append('%s: key order %s, expected %s' % (path, list(actual), list(expected)))
    for key in expected:
        if key in actual:
            differences += compare_values(actual[key], expected[key], tolerant, '%s.%s' % (path, key))
    return differences


def compare_lists(actual, expected, tolerant, path):
    if not isinstance(actual, list):
        return ['%s: expected a list, got %s' % (path, type_name(actual))]
    if len(actual) != len(expected):
        return ['%s: %d entries, expected %d' % (path, len(actual), len(expected))]
    differences = []
    for position, (left, right) in enumerate(zip(actual, expected)):
        differences += compare_values(left, right, tolerant, '%s[%d]' % (path, position))
    return differences


def result_named(report, name):
    """The entry of report.results with this check name, or None."""
    for entry in report.get('results', []):
        if isinstance(entry, dict) and entry.get('name') == name:
            return entry
    return None


# --------------------------------------------------------------------------
# Checks on the untampered samples
# --------------------------------------------------------------------------


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def json_report_or_fail(verifier, receipt_path, ledger, check, expected_exit=0):
    """Run `<verifier> receipt --json`; record a failure and return None when it misbehaves."""
    completed = verifier.run(receipt_path, '--json')
    if completed.returncode != expected_exit:
        ledger.failed(check, 'expected exit code %d\n%s' % (expected_exit, describe_run(completed)))
        return None
    report, problem = parse_report(completed)
    if problem:
        ledger.failed(check, problem)
    return report


def check_against_expected(verifier, receipt_path, expected, ledger):
    """The --json report of one verifier must reproduce samples/expected/<name>.json."""
    check = '%s --json matches samples/expected' % verifier.label
    report = json_report_or_fail(verifier, receipt_path, ledger, check)
    if report is None:
        return None
    differences = compare_values(report, expected, tolerant=True)
    if differences:
        ledger.failed(check, '\n'.join(differences))
    else:
        ledger.passed(check)
    return report


def check_ports_identical(reports, ledger, check='python and node --json identical'):
    """The two ports must print the same report.

    Booleans, integers and key order must be identical.  Floats get the tolerance: the
    python port computes exp, log and log1p with V8's fdlibm algorithms in plain IEEE
    arithmetic, while the node binary at hand may have been compiled with fused
    multiply-add and then rounds some of those calls differently in the last bit.  Any
    such last-bit difference is still printed so that it stays visible.
    """
    python_report, node_report = reports.get('python'), reports.get('node')
    if python_report is None or node_report is None:
        ledger.skipped(check, 'one of the reports is missing')
        return
    exact_differences = compare_values(node_report, python_report, tolerant=False, path='node')
    if not exact_differences:
        ledger.passed(check)
        return
    within_tolerance = compare_values(node_report, python_report, tolerant=True, path='node')
    if within_tolerance:
        ledger.failed(check, '\n'.join(within_tolerance))
        return
    ledger.passed(check + ' within tolerance')
    for difference in exact_differences:
        ledger.info('last-bit difference (engine math): ' + difference)


def check_stdout_identical(verifiers, receipt_path, ledger, check='python and node print the same text'):
    """Table mode and --quiet must give byte for byte the same stdout and the same exit code."""
    problems = []
    for flags, mode in (((), 'table mode'), (('--quiet',), '--quiet')):
        runs = [verifier.run(receipt_path, *flags) for verifier in verifiers]
        exit_codes = [completed.returncode for completed in runs]
        if len(set(exit_codes)) != 1:
            problems.append('%s: exit codes differ: %s' % (mode, exit_codes))
        outputs = [completed.stdout for completed in runs]
        if len(set(outputs)) != 1:
            problems.append('%s: stdout differs\n%s' % (mode, first_stdout_difference(outputs[0], outputs[1])))
    if problems:
        ledger.failed(check, '\n'.join(problems))
    else:
        ledger.passed(check)


def first_stdout_difference(left, right):
    """The first line on which two outputs disagree, for a failure message."""
    left_lines, right_lines = left.splitlines(), right.splitlines()
    for position in range(max(len(left_lines), len(right_lines))):
        a = left_lines[position] if position < len(left_lines) else '<missing>'
        b = right_lines[position] if position < len(right_lines) else '<missing>'
        if a != b:
            return 'line %d:\n  python: %r\n  node:   %r' % (position + 1, a, b)
    return 'outputs differ only in trailing bytes'


def reference_vs_ports(reference_report, reports):
    """node must match the reference bit for bit (same engine, same Math); python within tolerance."""
    differences = []
    if reports.get('node') is not None:
        differences += compare_values(reports['node'], reference_report, tolerant=False, path='node-vs-reference')
    if reports.get('python') is not None:
        differences += compare_values(reports['python'], reference_report, tolerant=True, path='python-vs-reference')
    return differences


def check_table_mode(verifier, receipt_path, ledger):
    """Table mode: seven labelled lines in the fixed order, then a PASS verdict."""
    check = '%s table mode' % verifier.label
    completed = verifier.run(receipt_path)
    lines = completed.stdout.splitlines()
    problems = []
    if completed.returncode != 0:
        problems.append('expected exit code 0\n' + describe_run(completed))
    if len(lines) < len(CHECK_NAMES) + 1:
        problems.append('expected at least %d lines, got %d' % (len(CHECK_NAMES) + 1, len(lines)))
    else:
        problems += table_line_problems(lines)
        verdict = lines[-1].lower()
        if 'pass' not in verdict or 'fail' in verdict:
            problems.append('last line is not a PASS verdict: %r' % lines[-1])
    if problems:
        ledger.failed(check, '\n'.join(problems))
    else:
        ledger.passed(check)


def table_line_problems(lines):
    """The first seven lines must start with the check names and carry the plain labels."""
    problems = []
    for position, name in enumerate(CHECK_NAMES):
        line = lines[position]
        if not line.startswith(name):
            problems.append('line %d should start with %s: %r' % (position + 1, name, line))
        elif CHECK_LABELS[name] not in line:
            problems.append('line %d lacks the label %r: %r' % (position + 1, CHECK_LABELS[name], line))
    return problems


def check_quiet_mode(verifier, receipt_path, ledger, expect_pass=True):
    """--quiet prints exactly one line, the verdict."""
    check = '%s --quiet prints only the verdict' % verifier.label
    completed = verifier.run(receipt_path, '--quiet')
    lines = completed.stdout.splitlines()
    wanted, unwanted = ('pass', 'fail') if expect_pass else ('fail', 'pass')
    expected_exit = 0 if expect_pass else 1
    problems = []
    if completed.returncode != expected_exit:
        problems.append('expected exit code %d\n%s' % (expected_exit, describe_run(completed)))
    if len(lines) != 1:
        problems.append('expected exactly one line, got %d: %r' % (len(lines), lines))
    elif wanted not in lines[0].lower() or unwanted in lines[0].lower():
        problems.append('verdict line should say %s: %r' % (wanted.upper(), lines[0]))
    if problems:
        ledger.failed(check, '\n'.join(problems))
    else:
        ledger.passed(check)


def check_reference_agrees(reference, receipt_path, expected, reports, ledger, check='browser reference agrees'):
    """The compiled browser verifier must reproduce the expected report (when there is
    one) and match both ports: node bit for bit, python within the float tolerance."""
    if not reference.available:
        ledger.skipped(check, 'reference not found at %s' % reference.path)
        return
    if not reports:
        ledger.skipped(check, 'no port reports to compare with')
        return
    report, problem = reference.report(receipt_path)
    if report is None:
        ledger.failed(check, problem)
        return
    differences = []
    if expected is not None:
        differences += compare_values(report, expected, tolerant=True, path='reference')
    differences += reference_vs_ports(report, reports)
    if differences:
        ledger.failed(check, '\n'.join(differences))
    else:
        ledger.passed(check)


def test_sample(sample_path, verifiers, reference, ledger):
    """Every untampered check for one receipt; returns the parsed --json reports by label."""
    expected_path = EXPECTED_DIR / sample_path.name
    ledger.start_section(sample_path.name)
    if not expected_path.is_file():
        ledger.failed('expected report exists', 'missing %s' % expected_path)
        return {}
    expected = load_json(expected_path)
    reports = {}
    for verifier in verifiers:
        reports[verifier.label] = check_against_expected(verifier, sample_path, expected, ledger)
        check_table_mode(verifier, sample_path, ledger)
        check_quiet_mode(verifier, sample_path, ledger)
    check_ports_identical(reports, ledger)
    check_stdout_identical(verifiers, sample_path, ledger)
    check_reference_agrees(reference, sample_path, expected, reports, ledger)
    return reports


# --------------------------------------------------------------------------
# Tampered receipts
# --------------------------------------------------------------------------


class Tamper:
    """One way of altering a receipt and what both verifiers must then do.

    build(receipt) mutates a deep copy in place and returns a short note about what
    changed, or None when the tamper does not apply to this receipt (for example a server
    receipt has no output rows and no chains).  expected_exit is 0, 1 or None; None marks
    the single-hop case whose effect is asserted by check_single_hop_effect instead.
    must_fail lists checks that have to report ok=false; WEIGHTS must always stay ok.
    """

    def __init__(self, name, description, build, expected_exit, must_fail=()):
        self.name = name
        self.description = description
        self.build = build
        self.expected_exit = expected_exit
        self.must_fail = tuple(must_fail)


def opened_rows(receipt):
    return receipt.get('openings', {}).get('rows', [])


def opened_pairs(receipt):
    return receipt.get('openings', {}).get('pairs', [])


def row_label(row):
    return 'layer %s, token %s' % (row.get('layer'), row.get('index'))


def another_token(token, reader):
    """A token index different from `token`, kept at or before `reader` when possible."""
    if token + 1 <= reader:
        return token + 1
    if token > 0:
        return token - 1
    return reader + 1


def first_reading_list_position(data):
    """Position of the first real entry of a reading list payload (-1 is filler)."""
    for position, value in enumerate(data):
        if value >= 0:
            return position
    return None


def first_pair_with_hops(receipt):
    for pair in opened_pairs(receipt):
        if pair.get('hops'):
            return pair
    return None


def tamper_nothing(receipt):
    """A control: the receipt re-serialised by this harness must still verify."""
    return 'no change'


def tamper_nudge_output_value(receipt):
    """Nudge one float32 of an opened output (o) row, well above tol and float32 resolution."""
    rows = opened_rows(receipt)
    if not rows or receipt['statement'].get('exactRows', 0) < 1 or not rows[0].get('o'):
        return None
    row = rows[0]
    data = row['o']['payload']['data']
    old = data[0]
    data[0] = old + max(0.01, 10 * receipt['statement']['tol'])
    return 'row 0 (%s): o[0] %r -> %r' % (row_label(row), old, data[0])


def tamper_reading_list_entry(receipt):
    """Point one entry of an opened reading list at a different token."""
    rows = opened_rows(receipt)
    if not rows:
        return None
    row = rows[0]
    data = row['S']['payload']['data']
    position = first_reading_list_position(data)
    if position is None:
        return None
    old = data[position]
    data[position] = another_token(old, row['index'])
    return 'row 0 (%s): S[%d] %d -> %d' % (row_label(row), position, old, data[position])


def tamper_drop_last_row(receipt):
    """Remove the last opened row: one challenged row then has no opening at all."""
    rows = opened_rows(receipt)
    if not rows:
        return None
    dropped = rows.pop()
    return 'removed row %d (%s)' % (len(rows), row_label(dropped))


def tamper_swap_two_rows(receipt):
    """Exchange two opened rows so neither sits at the position its challenge names."""
    rows = opened_rows(receipt)
    for other in range(1, len(rows)):
        if (rows[other].get('layer'), rows[other].get('index')) != (rows[0].get('layer'), rows[0].get('index')):
            rows[0], rows[other] = rows[other], rows[0]
            return 'swapped rows 0 (%s) and %d (%s)' % (row_label(rows[other]), other, row_label(rows[0]))
    return None


def tamper_change_nonce(receipt):
    """A different nonce picks different rows, tokens and pairs, so the openings no longer match."""
    old = receipt['nonce']
    receipt['nonce'] = old + '-tampered'
    return 'nonce %r -> %r' % (old, receipt['nonce'])


def tamper_change_statement_bound(receipt):
    """Raise statement.B: the statement digest, hence every challenge, changes."""
    old = receipt['statement']['B']
    receipt['statement']['B'] = old + 1
    return 'statement.B %r -> %r' % (old, receipt['statement']['B'])


def corrupt_first_hop(pair):
    """Change the first entry of the reading list opened for the first hop of a chain."""
    hop = pair['hops'][0]
    data = hop['S']['payload']['data']
    position = first_reading_list_position(data)
    if position is None:
        position = 0
    old = data[position]
    data[position] = another_token(old, hop['node'])
    return 'pair (%s -> %s), hop at layer %s node %s: S[%d] %d -> %d' % (
        pair.get('j'), pair.get('i'), hop.get('layer'), hop.get('node'), position, old, data[position])


def tamper_one_hop(receipt):
    """Change one hop's reading list in a single chain (a measured-fraction effect only)."""
    pair = first_pair_with_hops(receipt)
    if pair is None:
        return None
    return corrupt_first_hop(pair)


def tamper_every_chain(receipt):
    """Break the first hop of every chain: DIRECT and CONNECTED then have nothing left."""
    notes = [corrupt_first_hop(pair) for pair in opened_pairs(receipt) if pair.get('hops')]
    if not notes:
        return None
    return '%d chains broken; first: %s' % (len(notes), notes[0])


def tamper_merkle_sibling(receipt):
    """Corrupt one sibling hash in the Merkle path of an opened reading list."""
    rows = opened_rows(receipt)
    if not rows or not rows[0]['S'].get('path'):
        return None
    path = rows[0]['S']['path']
    old = path[0]
    path[0] = ('1' if old[:1] == '0' else '0') + old[1:]
    return 'row 0 (%s): S.path[0] %s... -> %s...' % (row_label(rows[0]), old[:8], path[0][:8])


TAMPERS = [
    Tamper('untouched-copy', 'an unmodified copy written back by this harness', tamper_nothing, 0),
    Tamper('nudge-output-f32', 'one float32 of an opened output row nudged',
           tamper_nudge_output_value, 1, ['EXACT']),
    Tamper('reading-list-entry', 'one entry of an opened reading list points at another token',
           tamper_reading_list_entry, 1, ['BOUNDED']),
    Tamper('drop-last-row', 'the last opened row removed', tamper_drop_last_row, 1, ['BOUNDED']),
    Tamper('swap-two-rows', 'two opened rows exchanged', tamper_swap_two_rows, 1, ['BOUNDED']),
    Tamper('nonce-changed', 'the nonce changed', tamper_change_nonce, 1, ['BOUNDED']),
    Tamper('statement-B-changed', 'statement.B raised by one', tamper_change_statement_bound, 1, ['BOUNDED']),
    Tamper('one-hop-changed', "one hop's reading list changed in a single chain", tamper_one_hop, None),
    Tamper('every-chain-broken', 'the first hop of every chain changed',
           tamper_every_chain, 1, ['DIRECT', 'CONNECTED']),
    Tamper('merkle-sibling-corrupt', 'one sibling hash in a Merkle path corrupted',
           tamper_merkle_sibling, 1, ['BOUNDED']),
]


def write_tampered_receipt(receipt, temp_dir, sample_name, tamper_name):
    path = temp_dir / ('%s.%s.json' % (sample_name, tamper_name))
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle)
    return path


def failed_check_problems(report, tamper):
    """Problems with which checks a tampered receipt's report marks as failed."""
    problems = []
    if report.get('ok') is not False:
        problems.append('report.ok should be false')
    for name in tamper.must_fail:
        entry = result_named(report, name)
        if entry is None or entry.get('ok') is not False:
            problems.append('%s should have failed: %r' % (name, entry))
    weights = result_named(report, 'WEIGHTS')
    if weights is None or weights.get('ok') is not True:
        problems.append('WEIGHTS should still pass: %r' % weights)
    return problems


def check_rejected(tamper, tampered_path, verifiers, ledger):
    """Both verifiers must exit 1, say ok=false and fail the named checks; returns the reports."""
    reports = {}
    for verifier in verifiers:
        check = 'tamper %s: %s rejects it' % (tamper.name, verifier.label)
        report = json_report_or_fail(verifier, tampered_path, ledger, check, expected_exit=1)
        if report is None:
            continue
        problems = failed_check_problems(report, tamper)
        if problems:
            ledger.failed(check, '\n'.join(problems))
        else:
            ledger.passed(check, 'failed checks: %s' % failed_check_names(report))
        reports[verifier.label] = report
        check_quiet_mode(verifier, tampered_path, ledger, expect_pass=False)
    return reports


def failed_check_names(report):
    return ', '.join(entry['name'] for entry in report.get('results', []) if entry.get('ok') is False) or 'none'


def check_still_accepted(tamper, tampered_path, expected, verifiers, ledger):
    """The control copy must verify exactly like the original; returns the reports."""
    reports = {}
    for verifier in verifiers:
        check = 'tamper %s: %s still accepts it' % (tamper.name, verifier.label)
        report = json_report_or_fail(verifier, tampered_path, ledger, check, expected_exit=0)
        if report is None:
            continue
        differences = compare_values(report, expected, tolerant=True)
        if differences:
            ledger.failed(check, '\n'.join(differences))
        else:
            ledger.passed(check)
        reports[verifier.label] = report
    return reports


def single_hop_problems(report, expected, direct_hop):
    """A single broken chain lowers CONNECTED (and DIRECT when the chain was one hop) by one."""
    problems = []
    for name, drop in (('CONNECTED', 1), ('DIRECT', 1 if direct_hop else 0)):
        before = result_named(expected, name)
        after = result_named(report, name)
        if before is None or after is None:
            problems.append('%s missing from a report' % name)
            continue
        if after.get('checked') != before.get('checked'):
            problems.append('%s checked %r, expected %r' % (name, after.get('checked'), before.get('checked')))
        if after.get('passed') != before.get('passed') - drop:
            problems.append('%s passed %r, expected %r' % (name, after.get('passed'), before.get('passed') - drop))
        should_be_ok = before.get('passed') - drop > 0
        if after.get('ok') is not should_be_ok:
            problems.append('%s ok %r, expected %r' % (name, after.get('ok'), should_be_ok))
    return problems


def check_single_hop_effect(tamper, original, tampered_path, expected, verifiers, ledger):
    """The single-hop tamper: exit code follows report.ok, and exactly one chain is lost."""
    pair = first_pair_with_hops(original)
    direct_hop = len(pair['hops']) == 1
    reports = {}
    for verifier in verifiers:
        check = 'tamper %s: %s drops exactly one chain' % (tamper.name, verifier.label)
        completed = verifier.run(tampered_path, '--json')
        report, problem = parse_report(completed)
        if problem:
            ledger.failed(check, problem)
            continue
        problems = single_hop_problems(report, expected, direct_hop)
        wanted_exit = 0 if report.get('ok') else 1
        if completed.returncode != wanted_exit:
            problems.append('exit code %d but report.ok is %r' % (completed.returncode, report.get('ok')))
        if problems:
            ledger.failed(check, '\n'.join(problems))
        else:
            ledger.passed(check, 'verdict %s, exit %d' % ('PASS' if report.get('ok') else 'FAIL', completed.returncode))
        reports[verifier.label] = report
    return reports


def test_one_tamper(tamper, original, expected, sample_name, temp_dir, verifiers, reference, ledger):
    """Build one tampered copy of a receipt and run every assertion on it."""
    receipt = copy.deepcopy(original)
    note = tamper.build(receipt)
    if note is None:
        ledger.skipped('tamper %s' % tamper.name, 'does not apply to this receipt (%s)' % tamper.description)
        return
    tampered_path = write_tampered_receipt(receipt, temp_dir, sample_name, tamper.name)
    ledger.info('tamper %s: %s' % (tamper.name, note))
    if tamper.expected_exit == 0:
        reports = check_still_accepted(tamper, tampered_path, expected, verifiers, ledger)
    elif tamper.expected_exit == 1:
        reports = check_rejected(tamper, tampered_path, verifiers, ledger)
    else:
        reports = check_single_hop_effect(tamper, original, tampered_path, expected, verifiers, ledger)
    check_ports_identical(reports, ledger, 'tamper %s: python and node --json identical' % tamper.name)
    check_stdout_identical(verifiers, tampered_path, ledger, 'tamper %s: python and node print the same text' % tamper.name)
    check_reference_agrees(reference, tampered_path, expected if tamper.expected_exit == 0 else None,
                           reports, ledger, 'tamper %s: browser reference agrees' % tamper.name)


def test_truncated_file(sample_path, temp_dir, verifiers, ledger):
    """Half a receipt is not JSON: both verifiers must exit 2 without a verdict."""
    text = sample_path.read_bytes()
    truncated_path = temp_dir / ('%s.truncated.json' % sample_path.stem)
    truncated_path.write_bytes(text[: len(text) // 2])
    for verifier in verifiers:
        check = 'tamper truncated-file: %s exits 2' % verifier.label
        completed = verifier.run(truncated_path, '--json')
        problems = []
        if completed.returncode != 2:
            problems.append('expected exit code 2\n' + describe_run(completed))
        if completed.stdout.strip():
            problems.append('nothing should be printed on stdout: %r' % completed.stdout[:200])
        if problems:
            ledger.failed(check, '\n'.join(problems))
        else:
            ledger.passed(check)


def test_tampers(sample_path, temp_dir, verifiers, reference, ledger):
    """Every tampered variant of one receipt."""
    ledger.start_section('%s (tampered copies)' % sample_path.name)
    expected_path = EXPECTED_DIR / sample_path.name
    if not expected_path.is_file():
        ledger.skipped('tampers', 'no expected report for this sample')
        return
    original = load_json(sample_path)
    expected = load_json(expected_path)
    for tamper in TAMPERS:
        test_one_tamper(tamper, original, expected, sample_path.stem, temp_dir, verifiers, reference, ledger)
    test_truncated_file(sample_path, temp_dir, verifiers, ledger)


# --------------------------------------------------------------------------
# Checks that do not depend on a sample
# --------------------------------------------------------------------------


def test_missing_file(verifiers, temp_dir, ledger):
    """A file that does not exist must give exit code 2."""
    ledger.start_section('command line')
    missing = temp_dir / 'does-not-exist.json'
    for verifier in verifiers:
        check = '%s exits 2 on a missing file' % verifier.label
        completed = verifier.run(missing, '--quiet')
        if completed.returncode == 2 and not completed.stdout.strip():
            ledger.passed(check)
        else:
            ledger.failed(check, 'expected exit code 2 and no stdout\n' + describe_run(completed))


class Malformed:
    """One way of damaging (or merely re-encoding) a receipt file.

    build(receipt) returns the bytes to write.  message is the exact shape-check message
    both scripts must print after 'error: ' (None when the parsers' own messages differ, as
    they do for invalid JSON), and accepted=True marks a file that must still verify.
    """

    def __init__(self, name, build, message=None, accepted=False):
        self.name = name
        self.build = build
        self.message = message
        self.accepted = accepted


def encoded(receipt, **dump_options):
    return json.dumps(receipt, **dump_options).encode('utf-8')


def malformed_nan_literal(receipt):
    receipt['statement']['tol'] = '__NAN__'
    return json.dumps(receipt).replace('"__NAN__"', 'NaN').encode('utf-8')


def malformed_utf16(receipt):
    return json.dumps(receipt).encode('utf-16')


def malformed_no_pairs_list(receipt):
    del receipt['openings']['pairs']
    return encoded(receipt)


def malformed_odd_salt(receipt):
    opening = receipt['openings']['rows'][0]['S']
    opening['salt'] = opening['salt'][:-1]
    return encoded(receipt)


def malformed_string_n(receipt):
    receipt['statement']['n'] = str(receipt['statement']['n'])
    return encoded(receipt)


def malformed_not_an_object(receipt):
    return b'[]'


def reencoded_with_bom(receipt):
    return b'\xef\xbb\xbf' + encoded(receipt)


def reencoded_with_crlf(receipt):
    return json.dumps(receipt, indent=1).replace('\n', '\r\n').encode('utf-8')


MALFORMED = [
    Malformed('nan-literal', malformed_nan_literal),
    Malformed('utf-16', malformed_utf16),
    Malformed('no-pairs-list', malformed_no_pairs_list, 'receipt.openings.pairs is not an array'),
    Malformed('odd-salt', malformed_odd_salt, 'receipt.openings.rows[0].S.salt is not a hex string'),
    Malformed('string-n', malformed_string_n, 'receipt.statement.n is not a number'),
    Malformed('not-an-object', malformed_not_an_object, 'receipt is not a JSON object'),
    Malformed('utf-8-bom', reencoded_with_bom, accepted=True),
    Malformed('crlf', reencoded_with_crlf, accepted=True),
]


def first_stderr_line(completed):
    return completed.stderr.splitlines()[0] if completed.stderr.strip() else ''


def check_refused(case, path, verifiers, ledger):
    """Both scripts exit 2 with nothing on stdout and, for shape problems, the same message."""
    messages = {}
    for verifier in verifiers:
        check = 'malformed %s: %s refuses it' % (case.name, verifier.label)
        completed = verifier.run(path, '--quiet')
        messages[verifier.label] = first_stderr_line(completed)
        problems = []
        if completed.returncode != 2:
            problems.append('expected exit code 2\n' + describe_run(completed))
        if completed.stdout.strip():
            problems.append('nothing should be printed on stdout: %r' % completed.stdout[:200])
        if case.message and messages[verifier.label] != 'error: ' + case.message:
            problems.append('expected stderr %r, got %r' % ('error: ' + case.message, messages[verifier.label]))
        elif not messages[verifier.label].startswith('error: '):
            problems.append('stderr should start with "error: ": %r' % messages[verifier.label])
        if problems:
            ledger.failed(check, '\n'.join(problems))
        else:
            ledger.passed(check, messages[verifier.label])


def check_accepted_reencoding(case, path, original_report, verifiers, ledger):
    """A re-encoded file must verify exactly like the original."""
    for verifier in verifiers:
        check = 'malformed %s: %s still accepts it' % (case.name, verifier.label)
        report = json_report_or_fail(verifier, path, ledger, check, expected_exit=0)
        if report is None:
            continue
        differences = compare_values(report, original_report, tolerant=False)
        if differences:
            ledger.failed(check, '\n'.join(differences))
        else:
            ledger.passed(check)


def test_malformed_files(sample_path, temp_dir, verifiers, ledger):
    """Damaged and re-encoded copies of one receipt."""
    ledger.start_section('%s (malformed and re-encoded copies)' % sample_path.name)
    original = load_json(sample_path)
    original_report = json_report_or_fail(verifiers[0], sample_path, ledger, 'original verifies')
    for case in MALFORMED:
        path = temp_dir / ('%s.%s.json' % (sample_path.stem, case.name))
        path.write_bytes(case.build(copy.deepcopy(original)))
        if case.accepted:
            if original_report is not None:
                check_accepted_reencoding(case, path, original_report, verifiers, ledger)
        else:
            check_refused(case, path, verifiers, ledger)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(description='End-to-end tests for verify.py and verify.mjs.')
    parser.add_argument('--only', metavar='TEXT', help='run only samples whose file name contains TEXT')
    parser.add_argument('--keep-temp', action='store_true', help='keep the temporary directory with the tampered receipts')
    parser.add_argument('--ts-reference', metavar='PATH', default=os.environ.get('TS_REFERENCE'),
                        help='compiled browser verifier to use as a third opinion (default: $TS_REFERENCE, else skipped)')
    parser.add_argument('--no-tampers', action='store_true', help='skip the tampered-receipt tests')
    parser.add_argument('--verbose', action='store_true', help='print details for passing checks too')
    return parser.parse_args(argv)


def find_samples(only):
    samples = sorted(path for path in SAMPLES_DIR.glob('*.json') if path.is_file())
    if only:
        samples = [path for path in samples if only in path.name]
    return samples


def python_version_text():
    return '%d.%d.%d' % sys.version_info[:3]


def node_version_text(node):
    completed = run_command([node, '--version'])
    return completed.stdout.strip() or '(unknown version)'


def resolve_reference_path(argument):
    if argument:
        return Path(argument).expanduser()
    return DEFAULT_TS_REFERENCE


def print_header(node, reference):
    print('8080.AI linear attention proof verifier tests')
    print('  repository : %s' % REPO_ROOT)
    print('  python     : %s (%s)' % (sys.executable, python_version_text()))
    print('  node       : %s (%s)' % (node, node_version_text(node)))
    state = 'found' if reference.available else 'not found, third-opinion checks will be skipped'
    print('  reference  : %s (%s)' % (reference.path, state))


def main(argv=None):
    args = parse_args(argv)
    node = shutil.which('node')
    if node is None:
        print('error: node is not on PATH; it is needed to run verify.mjs', file=sys.stderr)
        return 1
    for required in (PYTHON_VERIFIER, NODE_VERIFIER):
        if not required.is_file():
            print('error: %s is missing' % required, file=sys.stderr)
            return 1
    samples = find_samples(args.only)
    if not samples:
        print('error: no samples found in %s' % SAMPLES_DIR, file=sys.stderr)
        return 1

    verifiers = [
        Verifier('python', [sys.executable, str(PYTHON_VERIFIER)]),
        Verifier('node', [node, str(NODE_VERIFIER)]),
    ]
    reference = TsReference(node, resolve_reference_path(args.ts_reference))
    print_header(node, reference)

    ledger = Ledger(args.verbose)
    temp_dir = Path(tempfile.mkdtemp(prefix='8080-proof-verifier-tests-'))
    try:
        test_missing_file(verifiers, temp_dir, ledger)
        for sample_path in samples:
            test_sample(sample_path, verifiers, reference, ledger)
        test_malformed_files(min(samples, key=lambda path: path.stat().st_size), temp_dir, verifiers, ledger)
        if not args.no_tampers:
            for sample_path in samples:
                test_tampers(sample_path, temp_dir, verifiers, reference, ledger)
    finally:
        if args.keep_temp:
            print('\ntampered receipts kept in %s' % temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    ledger.print_summary()
    return 1 if ledger.count('FAIL') else 0


if __name__ == '__main__':
    sys.exit(main())
