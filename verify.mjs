#!/usr/bin/env node
/*
 * verify.mjs - check an 8080.AI proof receipt on your own computer.
 *
 * A receipt is the JSON file downloaded from the 8080.AI /proof page.  It commits to one
 * inference run: for every token at every layer, which earlier tokens that token read (its
 * reading list), and - for receipts produced in the browser - the attention inputs and
 * outputs of a few rows.  Each committed table is a salted Merkle tree; only the roots are
 * in the receipt.  The nonce recorded in the receipt, hashed together with the statement and
 * the roots (Fiat-Shamir), decides which rows, tokens and pairs had to be opened; this tool
 * re-derives those picks and ignores openings of anything else.  How the nonce was chosen is
 * not something a receipt can show - this tool takes it as it is.
 *
 * This script re-derives those picks from the receipt alone, checks every opening against
 * its root and evaluates seven checks:
 *
 *   BOUNDED    Reads at most B tokens
 *   EXACT      Ordinary attention, nothing approximated
 *   RANGE      Reaches far back                               (a measured fraction)
 *   HEARDBY    No token is dropped                            (a measured fraction)
 *   DIRECT     Read directly (measured fraction)
 *   CONNECTED  Linked by a short chain (measured fraction)
 *   WEIGHTS    Same model, no retraining: the receipt's modelHash equals its baseModelHash,
 *              two strings the receipt itself carries
 *
 * It is a faithful port of the browser verifier that runs on the /proof page and gives the
 * same answer: the verdict, every ok flag and every integer field exactly; the floating-point
 * statistics to the last bit on the same JavaScript engine build (see "How close" below).
 * Requires Node 18 or newer; uses only Node built-ins and never touches the network.
 *
 * What a PASS means: every check is evaluated only on the entries the nonce picked from this
 * one sealed run.  BOUNDED, EXACT and WEIGHTS held for every opened item; RANGE, HEARDBY,
 * DIRECT and CONNECTED report how often the property held among the opened items, with a
 * one-sided 95 % lower bound for this run.  A receipt says nothing about any other run, about
 * the model behind any particular request, about rows that were not opened beyond that bound,
 * or about the quality or accuracy of the run's output.
 *
 *   node verify.mjs receipt.json            table of checks, a summary, then a verdict line
 *   node verify.mjs receipt.json --json     report as JSON (same fields as samples/expected)
 *   node verify.mjs receipt.json --quiet    verdict line only (--json takes precedence)
 *
 * Exit code 0: every measured check passed.  1: a check failed.  2: file missing, not JSON,
 * or not shaped like a receipt.  verify.py prints exactly the same text.
 *
 * Byte-level rules (the same in the browser, verify.mjs and verify.py)
 *   part encoding     string -> UTF-8; number -> 8-byte big-endian IEEE-754 double; bytes as they are
 *   hashParts(...)    SHA-256 over the parts, each prefixed by its byte length as a 4-byte big-endian word
 *   payload bytes     dtype 'i32': int32 little-endian per value; 'f32': float32 little-endian per value
 *   leaf              hashParts('leaf', domain, index, salt, dtype, length, payloadBytes) with
 *                     domain = tree kind + layer number as JavaScript prints it ('S0', 'q2')
 *   node              hashParts('node', left, right); a level with an odd number of nodes pairs its
 *                     last node with itself
 *   Fiat-Shamir       seed = hashParts('fs', label, statementDigest, nonce, root_1, ..., root_m) with the
 *                     roots in S, q, k, v, o order; pick_c = first 32-bit big-endian word of
 *                     hashParts(seed, c) modulo the modulus (every modulus is far below 2^32)
 *   statement digest  hashParts('statement', JSON of the statement with its keys in UTF-16 code-unit
 *                     order, no whitespace, JavaScript number formatting)
 *
 * How close to the browser the numbers are: lowerBound, worstExactErr, meanDegree, hoeffding,
 * readsLower and readsUpper go through Math.exp, Math.log and Math.log1p.  Every V8 build ships
 * the same fdlibm algorithms for those, but builds compiled with fused multiply-add (Node on
 * Apple Silicon, for one) can differ from other builds in the last bit on about 1 % of inputs,
 * which shows up as a last-digit difference in those fields on some receipts (at most about
 * 1e-14 relative).  Compare --json output as parsed JSON with a small tolerance on those fields,
 * never byte for byte.  Unlike the browser page, this script has no limit on the number of
 * opened rows or the row width (the page's spread calls stop at roughly 120,000 arguments).
 */

import { createHash } from 'node:crypto';
import { readFileSync, realpathSync } from 'node:fs';
import { basename } from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

/* ═══════════════════════════ 1. bytes and hashing ═══════════════════════════ */

const utf8 = new TextEncoder();

/**
 * Bytes of one hash part.  Strings are UTF-8, numbers are 8-byte big-endian IEEE-754
 * doubles (an index such as 7939 is hashed as the double 7939.0, exactly as the browser
 * does) and byte arrays are used as they are.  Nothing else reaches this function once the
 * receipt passed assertReceiptShape (the nonce, the only receipt field hashed as it is, must
 * be a string).
 */
function partBytes(part) {
  if (typeof part === 'string') return utf8.encode(part);
  if (typeof part === 'number') {
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, part, false);
    return bytes;
  }
  if (part instanceof Uint8Array) return part;
  throw new TypeError(`cannot hash a part of type ${typeof part}`);
}

/**
 * SHA-256 over several parts, each prefixed by its byte length as a 4-byte big-endian word.
 * The length prefix removes any ambiguity about where one part ends and the next begins.
 */
function hashParts(...parts) {
  const hash = createHash('sha256');
  for (const part of parts) {
    const bytes = partBytes(part);
    const prefix = new Uint8Array(4);
    new DataView(prefix.buffer).setUint32(0, bytes.length, false);
    hash.update(prefix);
    hash.update(bytes);
  }
  return new Uint8Array(hash.digest());
}

function toHex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/** Bytes of a hex string.  assertReceiptShape guarantees valid, even-length hex before this runs. */
function fromHex(hex) {
  if (typeof hex !== 'string' || hex.length % 2 !== 0) throw new TypeError('expected an even-length hex string');
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i += 1) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

/**
 * Raw bytes of a committed payload: dtype 'i32' is int32 little-endian per element, anything
 * else is float32 little-endian per element.  Values are converted the way a typed array
 * would convert them (int32 wrap-around, float32 rounding), which is a no-op for values that
 * came out of such an array in the first place.
 */
function payloadBytes(payload) {
  const { dtype, data } = payload;
  const bytes = new Uint8Array(data.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < data.length; i += 1) {
    if (dtype === 'i32') view.setInt32(i * 4, data[i], true);
    else view.setFloat32(i * 4, data[i], true);
  }
  return bytes;
}

/* ═══════════════════════════ 2. Merkle openings ═══════════════════════════ */

const LEAF_TAG = 'leaf';
const NODE_TAG = 'node';

/** A leaf commits to its position, a fresh salt, the payload type, its length and its bytes. */
function leafHash(domain, index, salt, payload) {
  return hashParts(LEAF_TAG, domain, index, salt, payload.dtype, payload.data.length, payloadBytes(payload));
}

/**
 * Number of hashing levels above the leaves of a tree with n leaves: ceil(log2(n)), 0 for
 * n <= 1.  A level with an odd number of nodes pairs its last node with itself, so every
 * leaf has exactly this many siblings on its path to the root.
 */
function treeHeight(n) {
  return n > 1 ? Math.ceil(Math.log2(n)) : 0;
}

/**
 * Recompute the root from one opened leaf and its sibling path and compare it with the
 * committed root.  The caller says which position, payload type and payload length it
 * expects, so a valid opening of the wrong leaf is rejected too.
 */
function verifyOpening(rootHex, domain, opening, n, expectIndex, dtype, length) {
  const { index, payload, path } = opening;
  if (index !== expectIndex || index < 0 || index >= n || path.length !== treeHeight(n)) return false;
  if (payload.dtype !== dtype || payload.data.length !== length) return false;
  let node = leafHash(domain, index, fromHex(opening.salt), payload);
  let position = index;
  for (const siblingHex of path) {
    const sibling = fromHex(siblingHex);
    node = position % 2 === 0 ? hashParts(NODE_TAG, node, sibling) : hashParts(NODE_TAG, sibling, node);
    position >>= 1; // an int32 shift: a plain halving here, since index < n < 2^31
  }
  return toHex(node) === rootHex;
}

/** The domain tag of a tree: its kind followed by the layer number as JavaScript prints it ('S0', 'k3'). */
function domainOf(kind, layer) {
  return `${kind}${layer}`;
}

/* ═══════════════════════ 3. statement digest and challenges ═══════════════════════ */

/**
 * Digest of the public statement: SHA-256 of its canonical JSON.  The keys are sorted in
 * JavaScript's default order (by UTF-16 code unit), there is no whitespace, numbers are
 * written the way JavaScript writes them (0.0002 and 0.5 as such), and the replacer array
 * filters keys at every nesting level (which changes nothing today: every statement value
 * is a scalar).
 */
function statementDigest(statement) {
  const canonical = JSON.stringify(statement, Object.keys(statement).sort());
  return hashParts('statement', canonical);
}

/** Every committed root, in the fixed order S, q, k, v, o, as raw bytes. */
function rootsList(roots) {
  return [...roots.S, ...roots.q, ...roots.k, ...roots.v, ...roots.o].map(fromHex);
}

/**
 * Fiat-Shamir challenge derivation.  The seed binds the label, the statement digest, the
 * nonce and then every root (S, q, k, v, o order); each counter value yields one fresh
 * SHA-256 of (seed, counter), of which the first 32-bit big-endian word is reduced modulo
 * `modulus`.  Every modulus used here is far below 2^32, so the bias is negligible, and no
 * value ever exceeds the 53-bit exactness of a double.
 */
function fiatShamir(digest, roots, nonce, label, count, modulus) {
  const seed = hashParts('fs', label, digest, nonce, ...roots);
  const out = [];
  for (let counter = 0; out.length < count; counter += 1) {
    const h = hashParts(seed, counter);
    const word = new DataView(h.buffer, h.byteOffset, h.byteLength).getUint32(0, false);
    out.push(word % modulus);
  }
  return out;
}

/** The challenge values a receipt commits to for one label ('rows', 'tokens' or 'pairs'). */
function challenges(receipt, label, count, modulus) {
  const digest = statementDigest(receipt.statement);
  return fiatShamir(digest, rootsList(receipt.roots), receipt.nonce, label, count, modulus);
}

/** A row challenge in [0, n*L) names one (layer, token) pair. */
function rowChallenge(value, L) {
  return { layer: value % L, index: Math.floor(value / L) };
}

/** A pair challenge in [0, n*n) names two tokens; returned as [earlier, later]. */
function pairChallenge(value, n) {
  const a = Math.floor(value / n);
  const b = value % n;
  return a < b ? [a, b] : [b, a];
}

/* ═══════════════════════════ 4. statistics ═══════════════════════════ */

/** log(exp(a) + exp(b)) without overflow. */
function logAdd(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
}

/** log P[Binomial(n, p) >= k], summed term by term in log space. */
function logBinomTail(n, k, p) {
  if (k <= 0) return 0;
  if (p <= 0) return -Infinity;
  let logTerm = n * Math.log1p(-p);
  let acc = -Infinity;
  for (let x = 0; x <= n; x += 1) {
    if (x >= k) acc = logAdd(acc, logTerm);
    logTerm += Math.log((n - x) / (x + 1)) + Math.log(p) - Math.log1p(-p);
  }
  return acc;
}

/** One-sided 95 % Clopper-Pearson lower bound on a proportion, by 60 bisection steps. */
function clopperPearsonLower(k, n) {
  if (n === 0 || k === 0) return 0;
  let lo = 0;
  let hi = k / n;
  for (let it = 0; it < 60; it += 1) {
    const mid = (lo + hi) / 2;
    if (logBinomTail(n, k, mid) < Math.log(0.05)) lo = mid; else hi = mid;
  }
  return lo;
}

/** 95 % Hoeffding half-width for the mean of t values in [0, B]. */
function hoeffdingHalfWidth(B, t) {
  return B * Math.sqrt(Math.log(2 / 0.05) / (2 * Math.max(t, 1)));
}

/* ═══════════════════════════ 5. the seven checks ═══════════════════════════ */

/** The reading list carried by an opened S row: the entries that are not the -1 filler. */
function readingList(opening) {
  return opening.payload.data.filter((x) => x >= 0);
}

/**
 * A reading list for token i is sound when it is non-empty, has at most B entries, contains
 * i itself, never reaches past i, and is strictly increasing (so every entry is distinct).
 */
function readingListIsSound(list, i, B) {
  return list.length > 0 && list.length <= B
    && list.includes(i) && list[list.length - 1] <= i
    && list.every((x, idx) => idx === 0 || x > list[idx - 1]);
}

/** Open the S row of a challenged (layer, token); returns its reading list, or null when anything is off. */
function verifyRowSet(receipt, row) {
  const { n, B } = receipt.statement;
  const ok = verifyOpening(receipt.roots.S[row.layer], domainOf('S', row.layer), row.S, n, row.index, 'i32', B);
  const list = readingList(row.S);
  return ok && readingListIsSound(list, row.index, B) ? list : null;
}

/**
 * Ordinary softmax attention for one query row over exactly the opened keys and values,
 * computed in double precision from the float32-decoded inputs in a fixed order: per head,
 * scaled dot products, one softmax over every opened key, then the weighted sum of the
 * values.
 */
function softmaxOverOpenedKeys(qRow, kRows, vRows, heads, headDim, count) {
  const width = heads * headDim;
  const scale = 1 / Math.sqrt(headDim);
  const out = [];
  for (let h = 0; h < heads; h += 1) {
    const qi = qRow.subarray(h * headDim, (h + 1) * headDim);
    const scores = [];
    for (let idx = 0; idx < count; idx += 1) {
      const kj = kRows.subarray(idx * width + h * headDim, idx * width + (h + 1) * headDim);
      let dot = 0;
      for (let t = 0; t < headDim; t += 1) dot += qi[t] * kj[t];
      scores.push(dot * scale);
    }
    let top = -Infinity;
    for (const s of scores) top = Math.max(top, s);
    const weights = scores.map((s) => Math.exp(s - top));
    let total = 0;
    for (const w of weights) total += w;
    const acc = new Array(headDim).fill(0);
    for (let idx = 0; idx < count; idx += 1) {
      const vj = vRows.subarray(idx * width + h * headDim, idx * width + (h + 1) * headDim);
      const share = weights[idx] / total;
      for (let t = 0; t < headDim; t += 1) acc[t] += share * vj[t];
    }
    out.push(...acc);
  }
  return out;
}

/**
 * EXACT for one row: open the query, the output and every key/value the reading list names,
 * recompute ordinary attention over exactly those keys and compare with the committed output.
 * Returns the largest absolute difference when it is within `tol`, otherwise -1.
 */
function verifyRowExact(receipt, row, list) {
  const { n, heads, headDim, tol } = receipt.statement;
  const { roots } = receipt;
  const { layer } = row;
  const width = heads * headDim;
  if (!row.q || !row.o || !roots.q[layer] || !roots.o[layer]) return -1;
  const okQ = verifyOpening(roots.q[layer], domainOf('q', layer), row.q, n, row.index, 'f32', width);
  const okO = verifyOpening(roots.o[layer], domainOf('o', layer), row.o, n, row.index, 'f32', width);
  const keysOk = row.keys.length === list.length && row.keys.every((kv, idx) => kv.index === list[idx]
    && verifyOpening(roots.k[layer], domainOf('k', layer), kv.k, n, kv.index, 'f32', width)
    && verifyOpening(roots.v[layer], domainOf('v', layer), kv.v, n, kv.index, 'f32', width));
  if (!okQ || !okO || !keysOk) return -1;

  // Float32Array.set rounds each value to float32, i.e. decodes the committed f32 exactly.
  const kRows = new Float32Array(list.length * width);
  const vRows = new Float32Array(list.length * width);
  row.keys.forEach((kv, idx) => {
    kRows.set(kv.k.payload.data, idx * width);
    vRows.set(kv.v.payload.data, idx * width);
  });
  const qRow = Float32Array.from(row.q.payload.data);
  const oRow = row.o.payload.data;
  const reference = softmaxOverOpenedKeys(qRow, kRows, vRows, heads, headDim, list.length);
  let err = -Infinity;
  for (let t = 0; t < reference.length; t += 1) err = Math.max(err, Math.abs(reference[t] - oRow[t]));
  return err <= tol ? err : -1;
}

/**
 * One line of the report.  A check with nothing challenged (e.g. no pairs in a server
 * receipt) is "not measured" and never a failure.  BOUNDED, EXACT and WEIGHTS (`mustBeAll`)
 * need every challenged item to pass.  RANGE, HEARDBY, DIRECT and CONNECTED are fractions:
 * they are ok whenever at least one item passed, and the informative numbers are `fraction`
 * and `lowerBound` (95 % Clopper-Pearson lower bound for this run), not `ok`.
 */
function checkResult(name, checked, passed, mustBeAll) {
  let ok = true;
  if (checked !== 0) ok = mustBeAll ? passed === checked : passed > 0;
  return {
    name,
    ok,
    measured: checked > 0,
    checked,
    passed,
    fraction: checked ? passed / checked : 0,
    lowerBound: clopperPearsonLower(passed, checked),
  };
}

/** BOUNDED, EXACT and RANGE all come from the challenged rows. */
function verifyRows(receipt) {
  const { n, L, farFrac, rows, exactRows } = receipt.statement;
  const expected = challenges(receipt, 'rows', rows, n * L);
  const sizes = [];
  const count = { bounded: 0, exact: 0, farN: 0, far: 0, worstErr: 0 };

  receipt.openings.rows.forEach((row, idx) => {
    const value = expected[idx];
    if (value === undefined) return;
    const want = rowChallenge(value, L);
    if (row.layer !== want.layer || row.index !== want.index) return;
    const list = verifyRowSet(receipt, row);
    if (!list) return;
    count.bounded += 1;
    sizes.push(list.length);
    if (idx < exactRows) {
      const err = verifyRowExact(receipt, row, list);
      if (err >= 0) {
        count.exact += 1;
        count.worstErr = Math.max(count.worstErr, err);
      }
    }
    // RANGE is only meaningful once a token has some history behind it (position 64 onwards).
    if (row.index >= 64) {
      count.farN += 1;
      if (row.index - list[0] >= farFrac * row.index) count.far += 1;
    }
  });

  return {
    sizes,
    worstErr: count.worstErr,
    results: [
      checkResult('BOUNDED', expected.length, count.bounded, true),
      checkResult('EXACT', Math.min(exactRows, expected.length), count.exact, true),
      checkResult('RANGE', count.farN, count.far, false),
    ],
  };
}

/** HEARDBY: each challenged token must appear in the reading list of some later token. */
function verifyHeardBy(receipt) {
  const { n, B, tokens } = receipt.statement;
  const expected = challenges(receipt, 'tokens', tokens, Math.max(n - 64, 1));
  const byToken = new Map();
  for (const entry of receipt.openings.heardBy) byToken.set(entry.token, entry);
  let passed = 0;

  for (const token of expected) {
    const entry = byToken.get(token);
    if (!entry || entry.reader.index <= token) continue;
    const root = receipt.roots.S[entry.layer];
    if (!verifyOpening(root, domainOf('S', entry.layer), entry.reader, n, entry.reader.index, 'i32', B)) continue;
    if (readingList(entry.reader).includes(token)) passed += 1;
  }

  return checkResult('HEARDBY', expected.length, passed, false);
}

/**
 * Length of a valid chain from pair.j to pair.i: each hop is an opened S row of a later or
 * equal token at a strictly later layer whose reading list contains the previous token, and
 * the last hop is pair.i itself.  Returns 0 when the chain is missing or broken.
 */
function verifyHops(receipt, pair) {
  const { hops } = pair;
  if (!hops || hops.length === 0 || hops[hops.length - 1].node !== pair.i) return 0;
  const { n, B } = receipt.statement;
  let prev = pair.j;
  let lastLayer = -1;

  for (const hop of hops) {
    const ok = hop.layer > lastLayer && prev <= hop.node
      && verifyOpening(receipt.roots.S[hop.layer], domainOf('S', hop.layer), hop.S, n, hop.node, 'i32', B)
      && readingList(hop.S).includes(prev);
    if (!ok) return 0;
    prev = hop.node;
    lastLayer = hop.layer;
  }

  return hops.length;
}

/** DIRECT and CONNECTED come from the challenged pairs (a pair of the same token twice is skipped). */
function verifyPairs(receipt) {
  const { n, pairs } = receipt.statement;
  const expected = challenges(receipt, 'pairs', pairs, n * n);
  const count = { checked: 0, direct: 0, connected: 0 };

  expected.forEach((value, idx) => {
    const pair = receipt.openings.pairs[idx];
    const [j, i] = pairChallenge(value, n);
    if (i === j) return;
    count.checked += 1;
    if (!pair || pair.j !== j || pair.i !== i) return;
    const hops = verifyHops(receipt, pair);
    if (hops >= 1) count.connected += 1;
    if (hops === 1) count.direct += 1;
  });

  return [
    checkResult('DIRECT', count.checked, count.direct, false),
    checkResult('CONNECTED', count.checked, count.connected, false),
  ];
}

/**
 * Verify a parsed receipt that passed assertReceiptShape.  Returns the report with exactly
 * the fields of samples/expected/*.json: ok, maxDegree, worstExactErr, meanDegree, hoeffding,
 * readsLower, readsUpper, readsDense and results (one entry per check, in the fixed order).
 */
export function verify(receipt) {
  const rows = verifyRows(receipt);
  const { n, B, modelHash, baseModelHash } = receipt.statement;
  // WEIGHTS only compares two strings the receipt itself carries (modelHash and baseModelHash);
  // this tool has no list of published model hashes and cannot say which model either names.
  const weights = checkResult('WEIGHTS', 1, modelHash === baseModelHash ? 1 : 0, true);
  const results = [...rows.results, verifyHeardBy(receipt), ...verifyPairs(receipt), weights];

  let sum = 0;
  let longest = 0;
  for (const size of rows.sizes) {
    sum += size;
    longest = Math.max(longest, size);
  }
  const meanDegree = rows.sizes.length ? sum / rows.sizes.length : 0;
  const hoeffding = hoeffdingHalfWidth(B, rows.sizes.length);

  return {
    ok: results.every((r) => r.ok),
    maxDegree: rows.sizes.length ? longest : 0,
    worstExactErr: rows.worstErr,
    meanDegree,
    hoeffding,
    readsLower: n * Math.max(1, meanDegree - hoeffding),
    readsUpper: n * Math.min(B, meanDegree + hoeffding),
    readsDense: (n * (n + 1)) / 2,
    results,
  };
}

/* ═══════════════════════════ 6. receipt shape ═══════════════════════════
 *
 * The browser verifier coerces a few odd value types the way JavaScript does (a numeric
 * string as a layer, null as a payload entry, ...).  These scripts refuse such receipts up
 * front with exit code 2 instead: the /proof page never produces them, and refusing them
 * keeps every later comparison a plain comparison of numbers and strings.  Hex fields must
 * be valid even-length hex too (the browser would report a failed check instead).  verify.py
 * runs the same checks with the same messages.
 */

/** Raised for anything that should end with exit code 2. */
class ReceiptError extends Error {}

const HEX_RE = /^(?:[0-9a-fA-F]{2})*$/;
const STATEMENT_NUMBERS = ['n', 'L', 'B', 'heads', 'headDim', 'tol', 'farFrac', 'rows', 'exactRows', 'pairs', 'tokens'];
const STATEMENT_COUNTS = ['n', 'L', 'B', 'heads', 'headDim', 'rows', 'exactRows', 'pairs', 'tokens'];

function require(condition, message) {
  if (!condition) throw new ReceiptError(message);
}

function isObject(x) {
  return typeof x === 'object' && x !== null && !Array.isArray(x);
}

function isFiniteNumber(x) {
  return typeof x === 'number' && Number.isFinite(x);
}

function isHexString(x) {
  return typeof x === 'string' && HEX_RE.test(x);
}

/** One Merkle opening: index, salt, payload {dtype, data} and sibling path. */
function checkOpening(opening, path) {
  require(isObject(opening), `${path} is not an object`);
  require(isFiniteNumber(opening.index), `${path}.index is not a number`);
  require(isHexString(opening.salt), `${path}.salt is not a hex string`);
  const { payload } = opening;
  require(isObject(payload), `${path}.payload is missing`);
  require(typeof payload.dtype === 'string', `${path}.payload.dtype is not a string`);
  require(Array.isArray(payload.data) && payload.data.every((x) => typeof x === 'number'),
    `${path}.payload.data is not an array of numbers`);
  require(Array.isArray(opening.path) && opening.path.every(isHexString), `${path}.path is not an array of hex strings`);
}

/** One opened row: layer, index, S, optional q and o, and keys (required with q and o). */
function checkRow(row, path) {
  require(isObject(row), `${path} is not an object`);
  require(isFiniteNumber(row.layer), `${path}.layer is not a number`);
  require(isFiniteNumber(row.index), `${path}.index is not a number`);
  checkOpening(row.S, `${path}.S`);
  for (const kind of ['q', 'o']) {
    if (row[kind] !== undefined && row[kind] !== null) checkOpening(row[kind], `${path}.${kind}`);
  }
  const hasQ = row.q !== undefined && row.q !== null;
  const hasO = row.o !== undefined && row.o !== null;
  if ((row.keys !== undefined && row.keys !== null) || (hasQ && hasO)) {
    require(Array.isArray(row.keys), `${path}.keys is not an array`);
    row.keys.forEach((entry, position) => {
      const entryPath = `${path}.keys[${position}]`;
      require(isObject(entry), `${entryPath} is not an object`);
      require(isFiniteNumber(entry.index), `${entryPath}.index is not a number`);
      checkOpening(entry.k, `${entryPath}.k`);
      checkOpening(entry.v, `${entryPath}.v`);
    });
  }
}

function checkHeardByEntry(entry, path) {
  require(isObject(entry), `${path} is not an object`);
  require(isFiniteNumber(entry.token), `${path}.token is not a number`);
  require(isFiniteNumber(entry.layer), `${path}.layer is not a number`);
  checkOpening(entry.reader, `${path}.reader`);
}

function checkPair(pair, path) {
  require(isObject(pair), `${path} is not an object`);
  require(isFiniteNumber(pair.j), `${path}.j is not a number`);
  require(isFiniteNumber(pair.i), `${path}.i is not a number`);
  if (pair.hops === undefined || pair.hops === null) return;
  require(Array.isArray(pair.hops), `${path}.hops is not an array`);
  pair.hops.forEach((hop, position) => {
    const hopPath = `${path}.hops[${position}]`;
    require(isObject(hop), `${hopPath} is not an object`);
    require(isFiniteNumber(hop.layer), `${hopPath}.layer is not a number`);
    require(isFiniteNumber(hop.node), `${hopPath}.node is not a number`);
    checkOpening(hop.S, `${hopPath}.S`);
  });
}

/** Throw a ReceiptError unless the parsed JSON has the shape of a receipt. */
export function assertReceiptShape(receipt) {
  require(isObject(receipt), 'receipt is not a JSON object');
  const { statement, roots, openings, nonce } = receipt;
  require(isObject(statement), 'receipt.statement is missing');
  for (const key of STATEMENT_NUMBERS) {
    require(typeof statement[key] === 'number', `receipt.statement.${key} is not a number`);
    require(Number.isFinite(statement[key]), `receipt.statement.${key} is not a finite number`);
  }
  for (const key of STATEMENT_COUNTS) {
    require(Number.isInteger(statement[key]) && statement[key] >= 0, `receipt.statement.${key} is not a whole number`);
  }
  for (const key of ['modelHash', 'baseModelHash']) {
    require(typeof statement[key] === 'string', `receipt.statement.${key} is not a string`);
  }
  require(typeof nonce === 'string', 'receipt.nonce is not a string');
  require(isObject(roots), 'receipt.roots is missing');
  for (const key of ['S', 'q', 'k', 'v', 'o']) {
    require(Array.isArray(roots[key]), `receipt.roots.${key} is not an array`);
    roots[key].forEach((root, position) => {
      require(isHexString(root), `receipt.roots.${key}[${position}] is not a hex string`);
    });
  }
  require(isObject(openings), 'receipt.openings is missing');
  for (const key of ['rows', 'heardBy', 'pairs']) {
    require(Array.isArray(openings[key]), `receipt.openings.${key} is not an array`);
  }
  openings.rows.forEach((row, position) => checkRow(row, `receipt.openings.rows[${position}]`));
  openings.heardBy.forEach((entry, position) => checkHeardByEntry(entry, `receipt.openings.heardBy[${position}]`));
  openings.pairs.forEach((pair, position) => checkPair(pair, `receipt.openings.pairs[${position}]`));
}

export {
  hashParts, payloadBytes, verifyOpening, statementDigest, fiatShamir, clopperPearsonLower, hoeffdingHalfWidth,
};

/* ═══════════════════════════ 7. command line ═══════════════════════════ */

const CHECK_LABELS = {
  BOUNDED: 'Reads at most B tokens',
  EXACT: 'Ordinary attention, nothing approximated',
  RANGE: 'Reaches far back',
  HEARDBY: 'No token is dropped',
  DIRECT: 'Read directly (measured fraction)',
  CONNECTED: 'Linked by a short chain (measured fraction)',
  WEIGHTS: 'Same model, no retraining',
};

const CANNOT_CHECK_NOTE = 'This tool cannot check these values; compare them with what the /proof page shows, and '
  + 'treat the challenge as sound only if the nonce is one you supplied or that the prover '
  + 'could not have chosen (e.g. a value fixed after the roots were published).';

const USAGE = `Usage: node verify.mjs <receipt.json> [--json] [--quiet]

Checks a proof receipt downloaded from the 8080.AI /proof page.

  --json       print the report as JSON (same fields as samples/expected/*.json)
  --quiet      print only the verdict line
  -h, --help   show this help

A receipt is the JSON file downloaded from the 8080.AI /proof page (or one
someone sent you).  The script re-derives which rows, tokens and pairs the
receipt's nonce picked, checks every opening against its committed root and
evaluates the seven checks on exactly those opened entries.

Exit code 0: every measured check passed.  1: a check failed.  2: file
missing, not JSON, or not shaped like a receipt.
--json takes precedence over --quiet.  "not measured" means the statement
requested nothing of that kind (server receipts open no pairs, tokens or
attention rows); such a check is neither passed nor failed.
Compare --json output with samples/expected/*.json as parsed JSON, with a
small tolerance on the floating-point fields (see the README).`;

function parseArgs(argv) {
  const options = { file: null, json: false, quiet: false, help: false };
  for (const arg of argv) {
    if (arg === '--json') options.json = true;
    else if (arg === '--quiet') options.quiet = true;
    else if (arg === '-h' || arg === '--help') options.help = true;
    else if (arg.startsWith('-')) throw new ReceiptError(`unknown option ${arg}`);
    else if (options.file !== null) throw new ReceiptError('expected exactly one receipt file');
    else options.file = arg;
  }
  return options;
}

function loadReceipt(file) {
  let raw;
  try {
    raw = readFileSync(file);
  } catch (err) {
    throw new ReceiptError(`cannot read ${file}: ${err.message}`);
  }
  if (raw.length >= 2 && ((raw[0] === 0xff && raw[1] === 0xfe) || (raw[0] === 0xfe && raw[1] === 0xff))) {
    throw new ReceiptError(`${file} is UTF-16 encoded - save it as UTF-8 (PowerShell: Set-Content -Encoding utf8, `
      + 'or re-download from the /proof page)');
  }
  let text = raw.toString('utf8');
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // a UTF-8 byte-order mark, accepted as the browser does
  let receipt;
  try {
    receipt = JSON.parse(text);
  } catch (err) {
    throw new ReceiptError(`${file} is not valid JSON: ${err.message}`);
  }
  assertReceiptShape(receipt);
  return receipt;
}

function statusOf(check) {
  if (!check.measured) return 'not measured';
  return check.ok ? 'ok' : 'FAIL';
}

function percent(x) {
  return `${(100 * x).toFixed(1)} %`;
}

/** The last column: the measured fraction and its lower bound. */
function detailOf(check) {
  if (check.name === 'WEIGHTS') return 'deterministic (modelHash compared with baseModelHash)';
  if (!check.measured) return '';
  return `${percent(check.fraction)} passed, 95 % lower bound ${percent(check.lowerBound)}`;
}

function formatCheckLine(check) {
  const name = check.name.padEnd(10);
  const label = (CHECK_LABELS[check.name] ?? check.name).padEnd(44);
  const tally = `${check.passed}/${check.checked}`.padStart(9);
  const status = statusOf(check).padEnd(12);
  return `${name} ${label} ${tally}  ${status} ${detailOf(check)}`.trimEnd();
}

/** Why each unmeasured check has nothing to say. */
function notMeasuredLines(report, statement) {
  const unmeasured = report.results.filter((r) => !r.measured).map((r) => r.name);
  const lines = [];
  const byRequest = ['EXACT', 'HEARDBY', 'DIRECT', 'CONNECTED'].filter((name) => unmeasured.includes(name));
  if (byRequest.length) {
    const fields = [];
    if (byRequest.includes('EXACT')) fields.push(`${statement.exactRows === 0 ? 'exactRows' : 'rows'} = 0`);
    if (byRequest.includes('HEARDBY')) fields.push('tokens = 0');
    if (byRequest.includes('DIRECT')) fields.push(statement.pairs === 0 ? 'pairs = 0' : 'pairs naming the same token twice only');
    lines.push(`Not measured: ${byRequest.join(', ')} - the statement requests ${fields.join(', ')}, so no rows, tokens or `
      + 'pairs were opened for these checks (typical for a receipt produced by the 8080.AI server; browser receipts '
      + 'open all seven). These checks are neither passed nor failed.');
  }
  if (unmeasured.includes('RANGE')) {
    lines.push('Not measured: RANGE - no verified reading list belongs to a token at position 64 or later.');
  }
  if (unmeasured.includes('BOUNDED')) {
    lines.push('Not measured: BOUNDED - the statement requests rows = 0, so no reading list was opened.');
  }
  return lines;
}

/** A receipt string for the summary: as it is when plain printable ASCII, JSON-quoted otherwise. */
function shownText(value) {
  if (typeof value === 'string' && /^[\x20-\x7e]*$/.test(value)) return value;
  return JSON.stringify(value);
}

/** The summary block: receipt parameters, opened rows, the reads band, and the values taken as given. */
function summaryLines(receipt, report) {
  const { statement } = receipt;
  const { n, L, B, tol, farFrac } = statement;
  const opened = report.results[0].passed;
  const exact = report.results[1];
  const lines = [`Receipt: n = ${n} tokens, L = ${L} layers, B = ${B} (reading list cap), tol = ${tol}, farFrac = ${farFrac}`];
  if (opened > 0) {
    let openedLine = `Reading lists opened: ${opened}, mean length ${report.meanDegree.toFixed(2)}, longest ${report.maxDegree}`;
    if (exact.measured) openedLine += `, worst EXACT error ${report.worstExactErr.toExponential(2)} (tol ${tol})`;
    lines.push(openedLine);
    lines.push(`Reads per layer, estimated from the opened rows (95 % band): between ${Math.round(report.readsLower)}`
      + ` and ${Math.round(report.readsUpper)}; ${report.readsDense} if every token read every earlier token`);
  } else {
    lines.push('Reading lists opened: 0');
    lines.push('Reads per layer: not estimated (no reading list verified)');
  }
  lines.push('Values this tool takes as given:');
  for (const key of ['modelHash', 'baseModelHash', 'inputHash', 'ruleCommit']) {
    lines.push(`  ${key.padEnd(14)} ${key in statement ? shownText(statement[key]) : '(absent)'}`);
  }
  lines.push(`  ${'nonce'.padEnd(14)} ${JSON.stringify(receipt.nonce)}`);
  lines.push(CANNOT_CHECK_NOTE);
  return lines;
}

function verdictLine(report) {
  const measured = report.results.filter((r) => r.measured);
  const failed = measured.filter((r) => !r.ok).map((r) => r.name);
  const unmeasured = report.results.length - measured.length;
  let text = failed.length
    ? `Verdict: FAIL - ${failed.length} of ${measured.length} measured checks failed (${failed.join(', ')})`
    : `Verdict: PASS - all ${measured.length} measured checks passed`;
  if (unmeasured) text += `, ${unmeasured} not measured`;
  return text;
}

/** Everything table mode prints before the verdict. */
function reportLines(receipt, report) {
  const lines = report.results.map(formatCheckLine);
  lines.push('');
  const explanations = notMeasuredLines(report, receipt.statement);
  if (explanations.length) {
    lines.push(...explanations);
    lines.push('');
  }
  lines.push(...summaryLines(receipt, report));
  return lines;
}

function printReport(receipt, report, options) {
  if (options.json) {
    console.log(JSON.stringify(report, null, 1));
    return;
  }
  if (!options.quiet) {
    for (const line of reportLines(receipt, report)) console.log(line);
  }
  console.log(verdictLine(report));
}

/** Run the CLI; returns the process exit code. */
export function main(argv) {
  let options;
  try {
    options = parseArgs(argv);
  } catch (err) {
    console.error(`error: ${err.message}\n\n${USAGE}`);
    return 2;
  }
  if (options.help) {
    console.log(USAGE);
    return 0;
  }
  if (options.file === null) {
    console.error(USAGE);
    return 2;
  }

  let receipt;
  let report;
  try {
    receipt = loadReceipt(options.file);
    if (receipt.version !== undefined && receipt.version !== 1) {
      console.error(`warning: receipt version ${JSON.stringify(receipt.version)} is not the version this tool knows (1)`);
    }
    report = verify(receipt);
  } catch (err) {
    // A shape-checked receipt never throws in verify(); anything that does is malformed content.
    const reason = err instanceof ReceiptError ? err.message : `malformed receipt: ${err.name}: ${err.message}`;
    console.error(`error: ${reason}`);
    return 2;
  }

  printReport(receipt, report, options);
  return report.ok ? 0 : 1;
}

/**
 * True when this file is the script node was asked to run (not when it is imported).  If
 * the two paths cannot be compared (an unusual mount, a UNC share), fall back to the file
 * name so that the script never exits 0 without printing anything.
 */
function invokedDirectly() {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return realpathSync(fileURLToPath(import.meta.url)) === realpathSync(entry);
  } catch {
    return basename(entry) === 'verify.mjs';
  }
}

if (invokedDirectly()) {
  // exitCode rather than exit(): lets buffered stdout flush when piped.
  process.exitCode = main(process.argv.slice(2));
}
