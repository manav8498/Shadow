#!/usr/bin/env node
/**
 * Cross-validation harness for the TypeScript LTLf evaluator. Reads a
 * JSON object on stdin:
 *
 *   { formula: Formula, trace: TraceState[] }
 *
 * Calls evalAllPositions and writes the resulting truth-vector
 * (boolean[] of length n+1) to stdout as JSON.
 *
 * Used by python/tests/test_typescript_parity_property.py to assert
 * that the Python and TypeScript LTLf evaluators agree on every
 * (random formula, random trace) pair.
 */

import { readFileSync } from 'node:fs';
import { evalAllPositions } from '../dist/ltl/checker.js';

function main() {
  const stdin = readFileSync(0, 'utf-8');
  const input = JSON.parse(stdin);
  const truthVector = evalAllPositions(input.formula, input.trace ?? []);
  process.stdout.write(JSON.stringify(truthVector));
}

main();
