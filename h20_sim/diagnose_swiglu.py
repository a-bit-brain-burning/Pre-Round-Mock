"""Trace the unchanged SwiGLU PTX and audit cancellation with exact arithmetic."""

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import re
import struct

from bootstrap import DEFAULT_ROOT, HERE, digest, prepare, timestamp, validated_build
from operator_source import source_key
from run_swiglu import (REPO, cases, compile_replay, load_module, run_case, runtime_environment,
                        read_tensor, torch, write_json)


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def value(raw):
    return struct.unpack('<f', struct.pack('<I', raw))[0]


def rounded(number):
    """Nearest FP32 via exact distance, correcting possible FP64 double rounding."""
    middle = bits(float(number))
    candidates = [b for b in (middle - 1, middle, middle + 1) if 0 <= b <= 0xffffffff]
    if isinstance(number, Decimal):
        distance = lambda b: abs(Decimal.from_float(value(b)) - number)
    else:
        distance = lambda b: abs(Fraction(value(b)) - number)
    return value(min(candidates, key=lambda b: (distance(b), b & 1)))


def arithmetic(op, operands):
    a = [Fraction(x) for x in operands]
    if op == 'sub.f32': result = a[0] - a[1]
    elif op == 'add.f32': result = a[0] + a[1]
    elif op == 'mul.f32': result = a[0] * a[1]
    elif op == 'div.full.f32': result = a[0] / a[1]
    elif op == 'fma.rn.f32': result = a[0] * a[1] + a[2]
    elif op == 'ex2.approx.f32':
        with localcontext() as ctx:
            ctx.prec = 100
            return rounded(Decimal(2) ** Decimal.from_float(operands[0]))
    else: raise ValueError(f'Unimplemented diagnostic opcode: {op}')
    return rounded(result)


def parse_trace(text):
    rows = []
    for block in re.split(r'(?=\[thd=\d+\] : )', text):
        header = re.search(r'^\[thd=(\d+)\] : \((.*?):(\d+) - (.*?)\)\n', block)
        if not header:
            continue
        output = re.search(r'Output Registers:\n(.*?)(?:Input Registers:|Register File Contents:|\Z)', block, re.S)
        registers = re.findall(r'(%\w+)\s+\.b32\s+0x([0-9a-fA-F]+)', output.group(1) if output else '')
        rows.append({'instruction': header[4], 'source': header[2], 'line': int(header[3]),
                     'thread_uid': int(header[1]), 'outputs': [(name, int(raw, 16)) for name, raw in registers]})
    return rows


def dependency_path(ptx, traced, a, b, g):
    operations = {}
    for op, dst, args in re.findall(r'^\s*(sub\.f32|add\.f32|mul\.f32|div\.full\.f32|fma\.rn\.f32|ex2\.approx\.f32)\s+(%r\d+),\s*([^;]+);', ptx, re.M):
        operations[dst] = (op, [x.strip() for x in args.split(',')])
    constants = {dst: value(int(raw, 16)) for dst, raw in re.findall(r'mov\.b32\s+(%r\d+),\s*0f([0-9A-Fa-f]{8})', ptx)}
    loads = re.findall(r'@%p1\s+ld\.global\.b32\s+\{\s*(%r\d+)\s*\}', ptx)
    stores = re.findall(r'@%p1\s+st\.global\.b32\s+\[[^\]]+\],\s*\{\s*(%r\d+)\s*\}', ptx)
    if len(loads) != 3 or len(stores) != 2:
        raise ValueError('Original PTX structure changed; inspect before extending the diagnostic')
    inputs = dict(zip(loads, (a, b, g), strict=True))
    observed = {}
    for row in traced:
        m = re.match(r'(\S+)\s+(%r\d+),', row['instruction'])
        if m and m[2] in operations and row['outputs']:
            if len(row['outputs']) != 1:
                raise ValueError('Expected one scalar output in arithmetic trace')
            observed[m[2]] = row
    order = []
    visited = set()
    def visit(token):
        if token not in operations or token in visited:
            return
        op, args = operations[token]
        for arg in args: visit(arg)
        visited.add(token)
        order.append(token)
    for dst in stores: visit(dst)
    if any(dst not in observed for dst in order):
        raise ValueError('Missing arithmetic instructions in the actual trace')

    def evaluate(exponent_ulp=0, division_ulp=0, ideal_sigmoid=None):
        registers = {**constants, **inputs}
        rows = []
        for dst in order:
            op, args = operations[dst]
            operands = [value(int(x[2:], 16)) if x.startswith('0f') else registers[x] for x in args]
            result = arithmetic(op, operands)
            if op == 'ex2.approx.f32': result = value(bits(result) + exponent_ulp)
            if op == 'div.full.f32':
                result = value(bits(result) + division_ulp) if ideal_sigmoid is None else ideal_sigmoid
            registers[dst] = result
            actual = observed[dst]['outputs'][0][1]
            rows.append({'register': dst, 'op': op, 'operands': operands, 'model': result,
                         'model_bits': f'{bits(result):08x}', 'actual': value(actual),
                         'actual_bits': f'{actual:08x}', 'matches': bits(result) == actual,
                         'instruction': observed[dst]['instruction']})
        return [registers[dst] for dst in stores], rows
    return evaluate, stores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = prepare(args.root)
    run = root / 'runs' / f'sim003-swiglu-audit-{timestamp()}'
    run.mkdir()
    source, library, env = runtime_environment(root)
    env.update(PTX_SIM_DEBUG='0', PTX_SIM_DEBUG_THREAD_UID='4')
    exe = compile_replay(run, library, env)
    solution = load_module('sim003_swiglu_solution', REPO / 'SwiGLU_backward/solution.py')
    oracle = load_module('sim003_swiglu_reference', REPO / 'SwiGLU_backward/reference.py')
    report = {'experiment': 'sim003', 'kind': 'swiglu_instruction_audit', 'gpu_execution': False,
              'h20_tested': False, 'run_directory': str(run), 'build': validated_build(root, source, library),
              'trace_method': 'unchanged production PTX; backend per-thread instruction/register dump; no arithmetic replacement',
              'cases': [], 'source_sha256': {source_key(p): digest(p) for p in
                  [Path(__file__), HERE/'run_swiglu.py', HERE/'replay.cpp',
                   REPO/'SwiGLU_backward/solution.py', REPO/'SwiGLU_backward/reference.py', REPO/'common/validation.py']}}
    for seed, dtype, name, inputs in cases('full'):
        if name != 'amplified_cancellation_diagnostic' or dtype != torch.float32:
            continue
        folder = run / str(seed)
        folder.mkdir()
        executed = run_case(solution._backward, oracle, inputs, folder, source, exe, library, env, 180,
                            ['-gpgpu_ptx_inst_debug_to_file 1', '-gpgpu_ptx_inst_debug_thread_uid 4'])
        a, b, g = [x[0, 3].item() for x in inputs]
        expected = oracle.reference(*inputs)[0][0, 3].item()
        tolerance = 2e-5 * (1 + abs(expected))
        trace = parse_trace((folder/'inst_debug.txt').read_text())
        evaluator, stores = dependency_path((folder/'kernel.ptx').read_text(), trace, a, b, g)
        model, stages = evaluator()
        # Distinct checks: exact operations must agree; approximation choices
        # are examined separately against their documented 2-ULP bounds.
        exact_agree = all(r['matches'] for r in stages if r['op'] not in ('ex2.approx.f32', 'div.full.f32'))
        approximate_agree = all(r['matches'] for r in stages if r['op'] in ('ex2.approx.f32', 'div.full.f32'))
        actual = next(r['actual'] for r in stages if r['register'] == stores[0])
        serialized = [read_tensor(folder/f'{n}.out.bin', dtype, inputs[0].shape)[0, 3].item()
                      for n in ('da', 'db')]
        trace_outputs_match = all(bits(x) == bits(next(r['actual'] for r in stages if r['register'] == dst))
                                  for x, dst in zip(serialized, stores, strict=True))
        variations = []
        for ex in range(-2, 3):
            for div in range(-2, 3):
                output, _ = evaluator(ex, div)
                variations.append({'ex2_ulp': ex, 'div_ulp': div, 'da': output[0],
                                   'error_ratio': abs(output[0] - expected) / tolerance})
        with localcontext() as ctx:
            ctx.prec = 100
            da = Decimal.from_float(a)
            sigmoid = Decimal(1) / (Decimal(1) + (-da).exp())
            high_precision = Decimal.from_float(b * g) * (sigmoid + da * sigmoid * (1 - sigmoid))
            ideal, _ = evaluator(ideal_sigmoid=rounded(sigmoid))
            high_value = float(high_precision)
        row = {'seed': seed, 'production': executed, 'a': a, 'b': b, 'g': g,
               'expected_da': expected, 'actual_da': actual, 'exact_ptx_model_da': model[0],
               'allowed_error': tolerance, 'error_ratio': abs(actual - expected) / tolerance,
               'exact_operations_match_trace': exact_agree, 'nearest_approximation_model_matches_trace': approximate_agree,
               'trace_outputs_match_serialized_result': trace_outputs_match,
               'stages': stages, 'approximation_envelope_25_combinations': variations,
               'envelope_pass_count': sum(v['error_ratio'] <= 1 for v in variations),
               'envelope_min_error_ratio': min(v['error_ratio'] for v in variations),
               'ideal_sigmoid_fp32_same_tail_da': ideal[0],
               'ideal_sigmoid_fp32_same_tail_error_ratio': abs(ideal[0] - expected) / tolerance,
               'decimal_100_digit_da': high_value,
               'decimal_matches_original_reference': bits(high_value) == bits(expected),
               'trace_sha256': digest(folder/'inst_debug.txt')}
        write_json(folder/'audit.json', row)
        report['cases'].append(row)
        print(json.dumps({k: row[k] for k in ('seed', 'actual_da', 'error_ratio', 'exact_operations_match_trace',
                                             'nearest_approximation_model_matches_trace', 'envelope_pass_count',
                                             'envelope_min_error_ratio', 'ideal_sigmoid_fp32_same_tail_error_ratio')}), flush=True)
    report['same_pressure_input_for_both_seed_labels'] = report['cases'][0]['production']['input_sha256'] == report['cases'][1]['production']['input_sha256']
    report['status'] = 'AUDIT_COMPLETE' if all(r['exact_operations_match_trace'] and r['nearest_approximation_model_matches_trace']
                         and r['decimal_matches_original_reference'] and r['trace_outputs_match_serialized_result']
                         for r in report['cases']) else 'AUDIT_MISMATCH'
    report['production_status'] = ('PASS' if all(r['production']['status'] == 'FUNCTIONAL_PASS'
                                              for r in report['cases']) else 'HAS_FAILURES')
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected physical CUDA initialization')
    output = HERE / f'results-sim003-swiglu-audit-{timestamp()[:15]}.json'
    write_json(output, report)
    print(json.dumps({'report': str(output), 'status': report['status'], 'production_status': report['production_status']}))
    return 0 if report['status'] == 'AUDIT_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
