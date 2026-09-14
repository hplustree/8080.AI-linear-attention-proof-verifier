# 8080.AI linear attention proof verifier

Check a proof receipt from <https://8080.ai/proof> on your own computer.

## About 8080.AI linear attention

Standard attention re-reads every earlier token for every new token, so the
cost of a context grows with the square of its length. **8080.AI linear
attention** is an attention mechanism in which each token reads a bounded set of
earlier tokens (at most `B` = 288 in the runs behind this page, however long the
text), the output is still the ordinary softmax attention over exactly those
tokens, every token stays reachable through short chains of reads, and the
model weights are the published base model's, unchanged. The cost of a context
then grows in step with its length instead of with its square.

Rather than ask you to take that on trust, 8080.AI publishes it as a
**zero-knowledge proof**: the /proof page seals one run of 8080.AI linear
attention, lets you pick rows with a number only you know, opens exactly those
rows, and checks them in your browser. You learn that the properties hold on the
opened rows; you never learn how the mechanism chose which tokens to read. The
receipt it hands you is a plain JSON file, and it carries only a fingerprint of
the selection rule, never the rule itself.

The two scripts in this repository read that file and repeat every check
offline, so you do not have to trust the page, the server that sealed and opened
the run, or the browser code that checked it.

- `verify.py` runs on Python 3.9 or newer, standard library only.
- `verify.mjs` runs on Node 18 or newer, built-in modules only.
- Neither script opens a network connection or needs anything installed.
- Both give the same verdict, the same ok flags and the same counts as the
  checker on the /proof page, and the same floating-point statistics to about
  1e-14 (see [How close to the browser the numbers are](#how-close-to-the-browser-the-numbers-are)).
- Both print exactly the same text.

## Quick start

```sh
python3 verify.py receipt.json        # macOS, Linux
py verify.py receipt.json             # Windows (or: python verify.py receipt.json)
node verify.mjs receipt.json          # any platform with Node 18+
```

Either command prints one line per check, a summary block and a verdict line:

```
BOUNDED    Reads at most B tokens                           24/24  ok           100.0 % passed, 95 % lower bound 88.3 %
EXACT      Ordinary attention, nothing approximated           6/6  ok           100.0 % passed, 95 % lower bound 60.7 %
RANGE      Reaches far back                                 20/20  ok           100.0 % passed, 95 % lower bound 86.1 %
HEARDBY    No token is dropped                              24/24  ok           100.0 % passed, 95 % lower bound 88.3 %
DIRECT     Read directly (measured fraction)                19/24  ok           79.2 % passed, 95 % lower bound 61.1 %
CONNECTED  Linked by a short chain (measured fraction)      24/24  ok           100.0 % passed, 95 % lower bound 88.3 %
WEIGHTS    Same model, no retraining                          1/1  ok           deterministic (modelHash compared with baseModelHash)

Receipt: n = 512 tokens, L = 4 layers, B = 288 (reading list cap), tol = 0.0002, farFrac = 0.5
Reading lists opened: 24, mean length 144.88, longest 262, worst EXACT error 2.63e-8 (tol 0.0002)
Reads per layer, estimated from the opened rows (95 % band): between 33298 and 115054; 131328 if every token read every earlier token
Values this tool takes as given:
  modelHash      dddb700e84fa5f9fa74e4f76b3daaeb64fda3c0d4d7d2742c8fd8a90260dc00d
  baseModelHash  dddb700e84fa5f9fa74e4f76b3daaeb64fda3c0d4d7d2742c8fd8a90260dc00d
  inputHash      3eb9c4a441dea9be873b5a9bedb975fb6c019d121b28f2be94b89e254b1934b6
  ruleCommit     29d1df1eac54d5765b11515e85bd41268ada70024a0e91c6641b982f5c5a80ad
  nonce          "sample-nonce"
This tool cannot check these values; compare them with what the /proof page shows, and treat the challenge as sound only if the nonce is one you supplied or that the prover could not have chosen (e.g. a value fixed after the roots were published).
Verdict: PASS - all 7 measured checks passed
```

The numbers are `passed/checked`. The status column is `ok`, `FAIL`, or
`not measured` (see [What "not measured" means](#what-not-measured-means));
the last column is the measured fraction with its one-sided 95 %
Clopper-Pearson lower bound. For the four fraction checks (RANGE, HEARDBY,
DIRECT, CONNECTED), `ok` only means that at least one checked item passed;
the number that matters is the fraction and its lower bound. When a check is
not measured, a line after the table says why. The verdict line names the
failed checks and counts the unmeasured ones:

```
Verdict: FAIL - 2 of 2 measured checks failed (BOUNDED, WEIGHTS), 5 not measured
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | every measured check passed |
| 1 | at least one measured check failed |
| 2 | the file is missing, is not JSON, or is not shaped like a receipt (see [What the scripts refuse](#what-the-scripts-refuse-exit-code-2)) |

Options (the same for both scripts):

| Option | Effect |
|--------|--------|
| `--json` | print the full result as JSON instead of the table; the fields are listed under [The JSON result](#the-json-result) |
| `--quiet` | print only the verdict line |

`--json` takes precedence over `--quiet`. The verdict is decided by the exit
code, so scripts can call the verifier without parsing its output.

## How to get a receipt

On <https://8080.ai/proof>:

1. **Seal.** Press *Seal one run* (or *Verify zk proof* at the top of the
   page, which performs all three steps and the growth run for you). The page
   builds a run and seals every per-token table into fingerprints (Merkle
   roots). Nothing has been picked yet.
2. **Your number.** Type any number, or keep the one the page rolled, and press
   *Open the rows my number picks*. The rows, tokens and pairs to open are
   derived from the seal plus your number. On the page the seal comes first
   and the number after; a receipt cannot show that order, so these scripts
   take the nonce as it is (see [What a PASS means](#what-a-pass-means---and-what-it-does-not)).
3. **Verify.** Let the page run the seven checks, then, under *Take the receipt
   with you*, press *Download this receipt (JSON)*. The file is named
   `receipt-<tokens>-<start of your number>.json`.

Receipts made this way are *full* receipts: the run was sealed on the 8080.AI
server with, for every token, its reading list and its attention inputs and
outputs, and every opened row carries what is needed to recompute it. Your
browser only ever checks; nothing that chooses reading lists runs in it.
`samples/full-512-small.json` is a receipt of that kind.

The growth check further down the same page uses runs that were sealed once on
the server at sizes up to 1,048,576 tokens and are opened with your number in
the same way. The larger of those receipts have the same layout but carry
reading lists only: no attention inputs or outputs, no pairs and no heard-by
entries. `samples/server-*.json` are receipts of that kind.

The page can also check a receipt someone sent you (*Check a receipt someone
sent you*). These scripts do the same thing offline.

## The seven checks

Every check is run on the rows, tokens or pairs that your number picked, and
only on those. One sentence each:

| Name | Label | What passes |
|------|-------|-------------|
| BOUNDED | Reads at most B tokens | Every picked row opens correctly against its seal and its reading list has at most `B` distinct earlier tokens, in increasing order, including the token itself. |
| EXACT | Ordinary attention, nothing approximated | For the first `exactRows` picked rows, recomputing ordinary softmax attention over exactly the opened keys and values reproduces the sealed output row within `tol`. |
| RANGE | Reaches far back | A picked row at position 64 or later read a token at least `farFrac` of the way back to the start of the text (a measured fraction). |
| HEARDBY | No token is dropped | A picked token appears in the reading list of some later token at some layer, and that reading list opens correctly against its seal (a measured fraction). |
| DIRECT | Read directly (measured fraction) | For a picked pair of tokens, the later token read the earlier one directly at some layer. |
| CONNECTED | Linked by a short chain (measured fraction) | For a picked pair of tokens, a chain of reads through layers in increasing order links the earlier token to the later one (a direct read is a chain of length one). |
| WEIGHTS | Same model, no retraining | `statement.modelHash` equals `statement.baseModelHash`. Both are strings the receipt itself carries; the scripts have no list of published model hashes and cannot say which model either one names. |

BOUNDED, EXACT and WEIGHTS are all-or-nothing: one failing item fails the
check. RANGE, HEARDBY, DIRECT and CONNECTED are reported as fractions: the
scripts count how many of the picked items pass, and the result carries that
fraction together with a one-sided 95 % Clopper-Pearson lower bound on it.
These four checks are reported `ok` whenever at least one picked item passed,
so read the fraction and the bound, not the `ok` flag. A pair whose chain is
missing or broken counts as a failure for that pair, never as a skip.

### What "not measured" means

A check with nothing to check (`checked = 0`) is reported as *not measured*.
It is neither a pass nor a failure and never changes the verdict. This happens
when the statement asks for zero items of that kind: server-sealed receipts
have `exactRows`, `pairs` and `tokens` set to 0, so EXACT, HEARDBY, DIRECT and
CONNECTED are not measured on them, and RANGE is not measured when none of the
picked rows sits at position 64 or later.

## What a PASS means - and what it does not

A receipt commits to one sealed run: for every token at every layer, which earlier tokens that token read (its reading list), plus, for full receipts, the attention inputs and outputs of a few rows. The nonce in the receipt picks which rows, tokens and pairs had to be opened; this tool re-derives those picks, checks every opening against its committed root, and evaluates the seven checks on exactly those opened entries.

A PASS therefore tells you:

- BOUNDED, EXACT and WEIGHTS held for every opened item.
- RANGE, HEARDBY, DIRECT and CONNECTED held for the printed fraction of opened items; the printed lower bound is a one-sided 95 % Clopper-Pearson bound on how often the property holds across that run's rows or pairs, assuming the picks were random. These four checks are reported ok whenever at least one item passed - read the fraction and the bound, not the ok flag.

A PASS does not tell you:

- anything about rows that were not opened beyond the statistical bound above;
- anything about any other run, or about the model that served any particular request;
- what model the hashes name: WEIGHTS only compares two strings carried by the receipt (modelHash and baseModelHash), and this tool has no list of published hashes;
- how the nonce was chosen - the tool takes the receipt's nonce as it is;
- anything about the quality, accuracy or content of the run's output.

The receipt does not say, and this tool does not know, how each token's reading list came about; the checks only use what was opened.

## Receipt format and hashing rules

This section is precise enough to re-implement the verifier. When it and the
scripts disagree, the scripts (and the browser checker they mirror) are right.

### Layout

```
{
  "version": 1,          // both scripts warn on stderr when present and not 1, then continue
  "statement": {
    "n": 512,            // tokens in the run
    "L": 4,              // layers
    "B": 288,            // reading-list cap: an S row has exactly B slots
    "heads": 2,          // attention heads
    "headDim": 16,       // values per head; a q/k/v/o row has heads * headDim values
    "tol": 0.0002,       // EXACT tolerance
    "farFrac": 0.5,      // RANGE threshold
    "rows": 24,          // rows to open
    "exactRows": 6,      // how many of those rows also open q, o and their keys
    "pairs": 24,         // pairs to open
    "tokens": 24,        // heard-by tokens to open
    "inputHash": "…",    // 64 hex characters
    "modelHash": "…",    // 64 hex characters
    "baseModelHash": "…",// 64 hex characters
    "ruleCommit": "…"    // 64 hex characters
  },
  "nonce": "…",          // your number, as the string you typed
  "roots": { "S": [...], "q": [...], "k": [...], "v": [...], "o": [...] },
  "openings": {
    "rows":    [ { "layer", "index", "S", "q"?, "o"?, "keys": [ { "index", "k", "v" } ] } ],
    "heardBy": [ { "token", "layer", "reader" } ],
    "pairs":   [ { "j", "i", "hops": [ { "layer", "node", "S" } ] | null } ]
  }
}
```

`inputHash` and `ruleCommit` take part in the statement digest and in nothing
else.

Each entry of `roots.S` is the root of one sealed table per layer: `roots.S[l]`
seals the reading lists of layer `l`. Table `S_l` has `n` leaves; leaf `i` is
an `i32` array of exactly `B` values holding the reading list of token `i` at
layer `l` in increasing order, padded with `-1`. Full receipts also carry
`roots.q`, `roots.k`, `roots.v` and `roots.o` (one root per layer, `n` leaves
each; every leaf is an `f32` array of `heads * headDim` values: the query, key,
value and output rows of a token). Server-sealed receipts leave those four
lists empty. A root is 32 bytes written as 64 lowercase hex characters.

An **opening** is one leaf together with what is needed to recompute a root:

```
{ "index": 503, "salt": "<64 hex>", "payload": { "dtype": "i32" | "f32", "data": [...] }, "path": ["<64 hex>", ...] }
```

- `openings.rows[c]` answers the `c`-th row pick. `S` is the opening of the
  reading list, `q` and `o` (present for the first `exactRows` rows) are the
  openings of the query and output rows, and `keys` holds one `{ index, k, v }`
  per reading-list entry, in reading-list order, with the openings of that
  token's key and value rows at the same layer.
- `openings.heardBy` holds one entry per picked token, in any order: the
  `layer`, and the opening `reader` of the reading list of a later token at
  that layer.
- `openings.pairs[c]` answers the `c`-th pair pick. `hops` is `null` when the
  prover has no chain to show; otherwise each hop opens the reading list `S` of
  token `node` at layer `layer`.

### Hashing primitive

Everything is SHA-256. `hashParts(part_1, …, part_m)` is SHA-256 over the
concatenation of the parts, each preceded by its byte length as a 4-byte
big-endian unsigned integer. A part is encoded as:

| Part | Bytes |
|------|-------|
| string | UTF-8 |
| number | 8-byte big-endian IEEE-754 double (an index such as `7939` is hashed as the double `7939.0`) |
| raw bytes | as they are (a root, a salt or a hash from an earlier step) |

The length prefix means no two different part lists can produce the same
input bytes.

### Payload bytes

The bytes of a payload are its values laid out one after another,
little-endian: `i32` values as 4-byte signed integers, `f32` values as 4-byte
IEEE-754 single precision. The receipt writes the values as ordinary decimal
numbers; an honest receipt's `f32` values are exactly representable as
single precision, so encoding and decoding them loses nothing.

### Merkle leaf and node

```
leaf = hashParts("leaf", domain, index, salt, dtype, length, payloadBytes)
node = hashParts("node", left, right)
```

- `domain` is the table's kind followed by the layer number: `"S0"`, `"q2"`,
  `"k1"`, `"v1"`, `"o3"`.
- `index` is the leaf position and `length` the number of values in the
  payload, both hashed as numbers (doubles).
- `salt` is the 32-byte value from the opening (decoded from its 64 hex
  characters). Each leaf has its own salt, so a root reveals nothing about the
  rows that were not opened.
- `dtype` is the string `"i32"` or `"f32"`.

Leaves are paired left to right and hashed into the level above; when a level
has an odd number of nodes, the last node is paired with itself. The root is
the single node at the top. The height of a table with `n` leaves is
`ceil(log2(n))` for `n > 1` and `0` for `n = 1`.

Verifying an opening against a root, expecting position `p`, type `dtype` and
`length` values:

1. Reject unless `opening.index == p`, `0 <= p < n`, `path.length` equals the
   tree height, `payload.dtype == dtype` and `payload.data.length == length`.
2. Compute `node = leaf` from the opening.
3. For each sibling hash in `path`, bottom up: if the current position is
   even, `node = hashParts("node", node, sibling)`, otherwise
   `node = hashParts("node", sibling, node)`; then halve the position
   (dropping the remainder).
4. Accept when the hex of `node` equals the root.

### Statement digest

```
digest = hashParts("statement", canonicalJSON)
```

`canonicalJSON` is the statement serialised as JSON with its keys in sorted
order (plain code-unit order, so for the fields above: `B`, `L`,
`baseModelHash`, `exactRows`, `farFrac`, `headDim`, `heads`, `inputHash`,
`modelHash`, `n`, `pairs`, `rows`, `ruleCommit`, `tokens`, `tol`), no
whitespace, strings quoted as JSON strings, and numbers written the way
JavaScript writes them: the shortest decimal that reads back to the same
double, with no decimal point for whole numbers, and exponent notation only
for magnitudes below 1e-6 or at 1e21 and above. So `288` is `288`, `0.5` is
`0.5` and `0.0002` is `0.0002`. For example, a server-sealed statement
serialises as

```
{"B":288,"L":2,"baseModelHash":"1ea3…0389","exactRows":0,"farFrac":0.5,"headDim":16,"heads":2,"inputHash":"3f45…67bd","modelHash":"1ea3…0389","n":8192,"pairs":0,"rows":100,"ruleCommit":"b311…fc96","tokens":0,"tol":0.0002}
```

with the hashes written out in full.

### Deriving the picks from the digest, the roots and your number

The verifier does not trust the receipt's word about which rows were picked.
It derives the picks itself:

```
roots  = every root, as 32 raw bytes, in the order roots.S, roots.q, roots.k, roots.v, roots.o
seed   = hashParts("fs", label, digest, nonce, root_1, …, root_m)
pick_c = (first 4 bytes of hashParts(seed, c), read as a big-endian unsigned 32-bit integer) mod modulus
         for c = 0, 1, 2, … until `count` picks exist
```

`label`, `count` and `modulus` depend on what is being picked:

| label | count | modulus | pick → what |
|-------|-------|---------|-------------|
| `"rows"` | `statement.rows` | `n * L` | `layer = pick mod L`, `token = floor(pick / L)`; `openings.rows[c]` must name that layer and token |
| `"tokens"` | `statement.tokens` | `max(n - 64, 1)` | `token = pick`; matched against `openings.heardBy` by token |
| `"pairs"` | `statement.pairs` | `n * n` | `a = floor(pick / n)`, `b = pick mod n`, `j = min(a, b)`, `i = max(a, b)`; `openings.pairs[c]` must have that `j` and `i`; a pick with `i == j` is skipped and not counted |

The `nonce` is your number as a string, and `c` is hashed as a number. The
moduli are far below 2^32, so the reduction bias is negligible.

Why the picks are hard to steer: the seed is a hash of the statement, of every
root and of the nonce, so changing anything in a table changes its root and
with it every pick, and changing one character of the nonce changes the whole
pick list. What the scripts cannot know is when the nonce was chosen: an
offline verifier only sees that the openings match the nonce written in the
file, not whether that nonce was fixed after the roots were. Treat the
challenge as sound only if the nonce is one you supplied on the /proof page
or one the prover could not have chosen (for example a value fixed after the
roots were published).

### The checks in full

Let `width = heads * headDim`. "Opens against" means the opening verifies
against the named root with the given position, type and length, as above.

**BOUNDED.** For each row pick `c` with `(layer, token)`: `openings.rows[c]`
must carry the same `layer` and `index`; its `S` opens against
`roots.S[layer]` at position `token` as `i32` of length `B`; and its reading
list (the values that are not `-1`) is non-empty, has at most `B` entries, is
strictly increasing (so every entry is distinct), contains `token` itself, and
ends at or before `token`. `checked = rows`; a missing or mismatched row
counts as a failure. Rows opened beyond the number of picks are ignored.

**EXACT.** For the first `min(exactRows, rows)` picks, in addition to
BOUNDED: `q` opens against `roots.q[layer]` and `o` against `roots.o[layer]`,
both at position `token` as `f32` of length `width`; `keys` has one entry per
reading-list element, in the same order, with `keys[m].index` equal to the
`m`-th element, `k` opening against `roots.k[layer]` and `v` against
`roots.v[layer]` at that position as `f32` of length `width`. Then the output
row is recomputed in double precision from the single-precision values, per
head `h` with `d = headDim`:

```
q_h    = q[h*d : (h+1)*d]
s_m    = (sum over t = 0..d-1, in order, of q_h[t] * k_m[h*d + t]) * (1 / sqrt(d))    for each opened key m
top    = max over m of s_m
e_m    = exp(s_m - top)
den    = sum over m, in order, of e_m
out[h*d + t] = sum over m, in order, of (e_m / den) * v_m[h*d + t]
```

The row passes when `max over t of |out[t] - o[t]| <= tol`, comparing against
the `o` values as written in the receipt. `worstExactErr` in the result is the
largest such difference among passing rows. A row that failed BOUNDED fails
EXACT as well.

**RANGE.** Among rows that passed BOUNDED and have `token >= 64`: the row
passes when `token - first >= farFrac * token`, where `first` is the smallest
entry of the reading list. `checked` is the number of such rows.

**HEARDBY.** For each token pick `t`: the entry of `openings.heardBy` with
`token == t` must have `reader.index > t`, `reader` must open against
`roots.S[layer]` at position `reader.index` as `i32` of length `B`, and the
reader's reading list must contain `t`. `checked = tokens`.

**DIRECT and CONNECTED.** For each pair pick with `j < i` (equal endpoints are
skipped): the chain in `openings.pairs[c].hops` is valid when it is non-empty,
its last hop has `node == i`, and walking it from `prev = j` with
`lastLayer = -1`, every hop has `layer > lastLayer` and `node >= prev`, its `S`
opens against `roots.S[layer]` at position `node` as `i32` of length `B`, and
that reading list contains `prev`; after each hop `prev = node` and
`lastLayer = layer`. CONNECTED passes when the chain is valid; DIRECT passes
when the valid chain has exactly one hop. `checked` is the number of picks
with `j != i` for both.

**WEIGHTS.** Passes when `statement.modelHash == statement.baseModelHash`.
`checked = 1`.

### Statistics

For every check, `fraction = passed / checked` (0 when nothing was checked)
and `lowerBound` is the one-sided 95 % Clopper-Pearson lower bound: the
largest `p` such that `P[Binomial(checked, p) >= passed] < 0.05`, found by 60
steps of bisection on `[0, passed / checked]` using the binomial tail summed in
log space; it is 0 when `checked` or `passed` is 0. (For 1 of 1, as in
WEIGHTS, the formula gives 0.05; that is a property of the bound, not a doubt
about the comparison.)

Let `t` be the number of rows that passed BOUNDED and `meanDegree` the mean of
their reading-list lengths (`maxDegree` the longest). Then

```
hoeffding  = B * sqrt(ln(2 / 0.05) / (2 * max(t, 1)))
readsLower = n * max(1, meanDegree - hoeffding)
readsUpper = n * min(B, meanDegree + hoeffding)
readsDense = n * (n + 1) / 2
```

`readsLower` and `readsUpper` are a 95 % band for the number of reads per
layer, extrapolated from the opened rows to all `n` tokens (`n` times the mean
reading-list length, plus or minus the Hoeffding half-width); `readsDense` is
what every token reading every earlier token would cost per layer. The band is
only printed when at least one reading list verified.

### The JSON result

`--json` prints an object with exactly these fields, in this order:

| Field | Meaning |
|-------|---------|
| `ok` | `true` when every check is ok (not measured counts as ok) |
| `maxDegree` | longest reading list among rows that passed BOUNDED |
| `worstExactErr` | largest EXACT difference among passing rows (0 when none) |
| `meanDegree` | mean reading-list length among rows that passed BOUNDED |
| `hoeffding` | the half-width above |
| `readsLower`, `readsUpper`, `readsDense` | as above |
| `results` | seven objects, in the order BOUNDED, EXACT, RANGE, HEARDBY, DIRECT, CONNECTED, WEIGHTS, each with `name`, `ok`, `measured`, `checked`, `passed`, `fraction`, `lowerBound` |

`measured` is `checked > 0`. `ok` is `true` when nothing was checked; when
something was, BOUNDED, EXACT and WEIGHTS need `passed == checked` and the
other four need `passed > 0`.

## What the scripts refuse (exit code 2)

Before any hashing, both scripts check that the file has the shape of a
receipt, and refuse it with `error: <what is wrong>` on stderr and exit code 2
otherwise. The browser checker is more lenient in a few places - it coerces
some odd value types the way JavaScript does - and these scripts deliberately
do not follow it there, because the /proof page never produces such files and
refusing them keeps every later comparison a plain comparison of numbers and
strings. The rules:

- The file must be UTF-8 JSON. A UTF-8 byte-order mark and CRLF line endings
  are fine; a UTF-16 file (PowerShell's default redirection) is refused with a
  hint; the non-JSON literals `NaN`, `Infinity` and `-Infinity` are refused
  (the browser refuses them too).
- Every statement field in the layout above is required. `n`, `L`, `B`,
  `heads`, `headDim`, `rows`, `exactRows`, `pairs` and `tokens` must be whole
  numbers, `tol` and `farFrac` finite numbers, `modelHash` and `baseModelHash`
  strings (the browser would pass WEIGHTS with both absent). A numeric string
  such as `"512"` is refused, not converted.
- `nonce` must be a string; `roots.S`, `.q`, `.k`, `.v`, `.o` and
  `openings.rows`, `.heardBy`, `.pairs` must all be present as arrays, even
  when empty (server receipts carry `"pairs": []` and `"heardBy": []`).
- Every opening in the file must be well formed: a numeric `index`, a `salt`
  and `path` entries made of an even number of hex digits, a `payload` with a
  string `dtype` and a `data` array of numbers (`null`, `true` or a string in
  a payload is refused; the browser would coerce it). Roots must be hex strings
  too. A malformed hex string is reported as a malformed receipt, where the
  browser would report a failed check.
- `layer`, `index`, `token`, `node`, `j` and `i` must be numbers (`"1"` is
  refused, where the browser would read it as 1). `hops` may be `null` or
  absent, otherwise it must be an array of hops.

Nothing that the /proof page downloads trips any of these rules. The scripts
also have no limit on the number of opened rows or the row width, so they
verify receipts larger than the browser page can render.

## How close to the browser the numbers are

The verdict, every `ok` flag and every integer field (`checked`, `passed`,
`maxDegree`, `readsDense`) are exact: the two scripts and the browser checker
always agree on them. `lowerBound`, `worstExactErr`, `meanDegree`,
`hoeffding`, `readsLower` and `readsUpper` go through `exp`, `log` and
`log1p`, and there the last bit depends on the engine:

- V8 (Chrome, Edge, Node) does not use the platform's libm for these; it ships
  its own copies of the fdlibm routines. `verify.py` carries those same
  routines, step by step, in plain IEEE double arithmetic, so its numbers are
  the same on every platform and never depend on the C library.
- A V8 binary compiled with fused multiply-add (Node on Apple Silicon, for one)
  rounds about 1 % of those calls differently in the last bit, and different
  builds fuse at different places. Because the Clopper-Pearson bisection
  converges onto the point where the log tail equals `log(0.05)`, such a
  difference can move `lowerBound` by one or two units in the last decimal
  digit on some `(passed, checked)` pairs (about 1e-15 relative; at most about
  1e-14). `worstExactErr` can move by a similar amount. Verdicts are not
  affected unless an EXACT error sits within about 1e-19 of `tol`.
- Safari and Firefox use other math libraries and differ from V8 in the same
  way.

So compare `--json` output as parsed JSON with a small tolerance on those six
fields (1e-9 relative is generous), never byte for byte, and read printed
lower bounds as what they are: a 95 % bound quoted to one decimal.
`verify.mjs` matches the browser bit for bit whenever both run on the same V8
build, and both scripts print exactly the same table (percentages are rounded
to one decimal, where a last-bit difference cannot show).

## Samples

`samples/` holds three real receipts and, under `samples/expected/`, the result
the reference checker produced for each one (the same fields as `--json`):

| Receipt | Sealed | Tokens | Opened |
|---------|--------|--------|--------|
| `full-512-small.json` | server, full seal | 512 | 24 rows (6 with attention inputs and outputs), 24 tokens, 24 pairs |
| `server-8192.json` | server, reading lists | 8,192 | 100 rows, reading lists only |
| `server-1048576.json` | server, reading lists | 1,048,576 | 100 rows, reading lists only |

Try them straight away:

```sh
python3 verify.py samples/server-1048576.json
node verify.mjs samples/full-512-small.json
```

The full test suite (`python3 tests/run_tests.py`, needs `node` on PATH)
checks both scripts against these files, checks that the two scripts print
the same text, and rejects tampered and malformed copies. The short form of
the same comparison, done on parsed JSON with a tolerance on the six
floating-point fields (see [How close to the browser the numbers are](#how-close-to-the-browser-the-numbers-are)):

```sh
python3 - <<'EOF'
import glob, json, math, os, subprocess

def same(actual, expected):
    if isinstance(expected, dict):
        return list(actual) == list(expected) and all(same(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float) or isinstance(actual, float):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    return actual == expected

for receipt in sorted(glob.glob('samples/*.json')):
    expected_path = os.path.join('samples', 'expected', os.path.basename(receipt))
    with open(expected_path) as handle:
        expected = json.load(handle)
    for script in (['python3', 'verify.py'], ['node', 'verify.mjs']):
        output = subprocess.run(script + [receipt, '--json'], capture_output=True, text=True, check=True).stdout
        print('ok' if same(json.loads(output), expected) else 'MISMATCH', script[1], receipt)
EOF
```

Every line should read `ok`. Both scripts print `--json` with one space of
indentation and a final newline, so on the shipped samples
`diff <(node verify.mjs samples/server-8192.json --json) samples/expected/server-8192.json`
happens to be clean too; do not rely on that for other receipts or other
machines. The full receipt is over a megabyte because it carries the opened
keys and values; the 1,048,576-token receipt is small because it carries
reading lists only.

To see a failure, change one value in an opened row of any sample and run the
verifier again: the row's opening no longer matches its root, and the checks
that depend on it report `FAIL` with exit code 1.

## Questions and reports

If a receipt downloaded from <https://8080.ai/proof> fails in these scripts,
or the scripts and the page disagree, open an issue on this repository with
the receipt attached (it contains nothing personal: a synthetic run, hashes,
opened rows and the number you typed) and the exact command and output. The
verifier here is the reference for what the receipt format means; the page's
checker mirrors it.

## License

MIT. See [LICENSE](LICENSE).
