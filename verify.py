#!/usr/bin/env python3
"""
verify.py - check a proof receipt downloaded from the 8080.AI /proof page.

Python 3.9 or newer, standard library only, no network access.

    python3 verify.py receipt.json            one line per check, a summary, then a verdict
    python3 verify.py receipt.json --json     the verification result as JSON
    python3 verify.py receipt.json --quiet    only the verdict line

    (macOS / Linux: python3 verify.py ...   Windows: py verify.py ... or python verify.py ...)

Exit code 0 when every measured check passes, 1 when any measured check fails,
2 when the file is missing, is not JSON, or is not shaped like a receipt.
--json takes precedence over --quiet.

What a receipt is
-----------------
A receipt describes one attention run over n tokens and L layers.  For every
token at every layer the prover committed to the token's *reading list*: which
earlier tokens that token read.  For browser receipts the prover also committed
to the per-token query rows (q), key rows (k), value rows (v) and output rows
(o).  Each committed list is a salted Merkle tree; the receipt carries only the
roots plus the handful of entries the challenge picked.

The challenge is derived from the receipt itself (statement digest, roots and
the nonce) by Fiat-Shamir hashing, so the verifier can recompute which rows,
tokens and pairs the prover had to open.  Nothing in the receipt says how the
reading lists were chosen; the checks below only use what was opened.

What a PASS means
-----------------
Every check is evaluated only on the entries the nonce picked from this one
sealed run.  BOUNDED, EXACT and WEIGHTS held for every opened item; the four
measured-fraction checks report how often the property held among the opened
items, with a one-sided 95 % Clopper-Pearson lower bound for this run.  A
receipt says nothing about any other run, about the model behind any
particular request, about rows that were not opened beyond that bound, or
about the quality or accuracy of the run's output.  How the nonce was chosen
is not something a receipt can show; this tool takes it as it is.

The seven checks
----------------
    BOUNDED    every challenged reading list holds at most B distinct tokens,
               all at or before the reader, in increasing order
    EXACT      the opened output row equals ordinary softmax attention over
               exactly the opened keys and values (browser receipts only)
    RANGE      a challenged token (position >= 64) read something at least
               farFrac x its own position back (a measured fraction)
    HEARDBY    a challenged token appears in the reading list of some later
               token at some layer (a measured fraction)
    DIRECT     for a challenged pair, the later token read the earlier one
               directly (a measured fraction)
    CONNECTED  for a challenged pair, a chain of reads through strictly
               increasing layers links the two (a measured fraction)
    WEIGHTS    the receipt's modelHash equals its baseModelHash.  Both are strings
               the receipt itself carries; this tool has no list of published
               model hashes, so it cannot say which model either one names.

BOUNDED, EXACT and WEIGHTS are all-or-nothing: one failing item fails the
check.  The four measured-fraction checks are reported ok whenever at least
one challenged item passed; what they actually tell you is the fraction and
its 95 % Clopper-Pearson lower bound, so read those numbers, not the ok flag.

A check with nothing to measure (e.g. no pairs in a server receipt) reports
ok=true, measured=false, checked=0: it is "not measured", neither passed nor
failed, and never changes the verdict.

This file is a port of the browser verifier, which is the reference.  Every
byte that is hashed, every arithmetic step of the EXACT recompute and every
statistic is reproduced in the same order.  Each section below names the
reference routine it mirrors.

Byte-level rules (the same in the browser, verify.mjs and verify.py)
--------------------------------------------------------------------
  part encoding     str -> UTF-8; number -> 8-byte big-endian IEEE-754 double;
                    bytes -> as they are
  hashParts(...)    SHA-256 over the parts, each prefixed by its byte length as
                    a 4-byte big-endian word
  payload bytes     dtype 'i32': int32 little-endian per value; 'f32': float32
                    little-endian per value
  leaf              hashParts('leaf', domain, index, salt, dtype, length,
                    payloadBytes) with domain = tree kind + layer number as
                    JavaScript prints it ('S0', 'q2')
  node              hashParts('node', left, right); a level with an odd number
                    of nodes pairs its last node with itself
  Fiat-Shamir       seed = hashParts('fs', label, statementDigest, nonce,
                    root_1, ..., root_m) with the roots in S, q, k, v, o order;
                    pick_c = first 32-bit big-endian word of hashParts(seed, c),
                    reduced modulo the modulus (every modulus is far below 2^32)
  statement digest  hashParts('statement', JSON of the statement with its keys
                    in UTF-16 code-unit order, no whitespace, JavaScript number
                    formatting)

How close to the browser the numbers are
----------------------------------------
The verdict, every ok flag and every integer field (checked, passed, maxDegree,
readsDense) are exact.  lowerBound, worstExactErr, meanDegree, hoeffding,
readsLower and readsUpper go through exp, log and log1p.  This file carries
V8's own implementations of those functions (fdlibm), in plain IEEE double
arithmetic, so its output is the same on every platform.  JavaScript engines
compiled with fused multiply-add (Node on Apple Silicon, for one) can differ
from that in the last bit on about 1 % of inputs, which shows up as a last-digit
difference in those fields on some receipts (at most about 1e-14 relative).
Compare --json output as parsed JSON with a small tolerance on those fields,
never byte for byte.

CLI output format (verify.mjs prints exactly the same text)
------------------------------------------------------------
Table mode prints one line per check:
    <name padded to 10> <label padded to 44> <passed/checked right-aligned 9>  <status padded to 12> <detail>
where status is "ok", "FAIL" or "not measured" and detail is the measured
fraction with its 95 % lower bound (WEIGHTS: "deterministic ...").  Then a
blank line, an explanation of every check that is not measured, a summary
block (receipt parameters, opened reading lists, the reads band, and the
hashes and nonce the tool takes as given), and the verdict:
    Verdict: PASS - all M measured checks passed[, U not measured]
    Verdict: FAIL - F of M measured checks failed (NAMES)[, U not measured]
A receipt whose "version" field is present and not 1 gets a warning on stderr
but is still verified with these rules.
"""

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from decimal import Decimal, ROUND_HALF_UP

# --------------------------------------------------------------------------
# Small JavaScript-compatibility helpers
#
# The reference runs in a browser, so a few JavaScript conventions leak into
# the bytes that get hashed and the text that gets printed (number formatting,
# integer wrap-around, toFixed rounding).  These helpers reproduce them.
# --------------------------------------------------------------------------


def is_number(value):
    """True for int or float, but not for bool (which Python treats as int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def same_number(a, b):
    """JavaScript strict equality between two receipt numbers."""
    return is_number(a) and is_number(b) and a == b


def js_mod(a, b):
    """JavaScript's % operator on two numbers.

    Both operands are taken as doubles, as JavaScript does; the result keeps
    the sign of the dividend and is NaN for a zero divisor.  A whole-number
    result comes back as an int so that challenge picks stay integers.
    """
    a, b = float(a), float(b)
    if b == 0 or math.isnan(a) or math.isnan(b) or math.isinf(a):
        return math.nan
    result = a if math.isinf(b) else math.fmod(a, b)
    return int(result) if result.is_integer() else result


def js_div(a, b):
    """JavaScript's / operator (division by zero gives +-Infinity or NaN)."""
    a, b = float(a), float(b)
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def js_floor(value):
    """Math.floor that passes NaN and infinities through instead of raising."""
    value = float(value)
    if not math.isfinite(value):
        return value
    return math.floor(value)


def js_round(value):
    """Math.round: the nearest integer, halves going towards +Infinity."""
    value = float(value)
    if not math.isfinite(value):
        return value
    lower = math.floor(value)
    return lower + 1 if value - lower >= 0.5 else lower


def js_max(values):
    """Math.max(...values): -Infinity when empty, NaN when any value is NaN."""
    best = -math.inf
    for value in values:
        if math.isnan(value):
            return math.nan
        if value > best:
            best = value
    return best


def to_int32(value):
    """JavaScript's ToInt32: truncate, wrap modulo 2**32, reinterpret as signed."""
    value = float(value)
    if not math.isfinite(value):
        return 0
    wrapped = int(value) & 0xFFFFFFFF
    return wrapped - (1 << 32) if wrapped >= (1 << 31) else wrapped


def shortest_decimal(positive):
    """Shortest round-trip digits of a positive float.

    Returns (digits, n) with digits a string without leading or trailing
    zeros and n the decimal exponent such that positive == 0.digits x 10**n.
    This is the (s, n, k) decomposition used by JavaScript's Number toString.
    """
    text = repr(positive)                       # e.g. '2.63e-08', '0.0002', '512.0'
    mantissa, _, exponent_text = text.partition('e')
    exponent = int(exponent_text) if exponent_text else 0
    int_part, _, frac_part = mantissa.partition('.')
    digits = (int_part + frac_part).lstrip('0')
    exponent -= len(frac_part)                  # value == int(digits) x 10**exponent
    stripped = digits.rstrip('0')
    exponent += len(digits) - len(stripped)
    return stripped, exponent + len(stripped)


def js_number_to_string(value):
    """Format a number exactly as JavaScript's Number.prototype.toString does."""
    number = float(value)
    if math.isnan(number):
        return 'NaN'
    if math.isinf(number):
        return 'Infinity' if number > 0 else '-Infinity'
    if number == 0:
        return '0'
    digits, n = shortest_decimal(abs(number))
    k = len(digits)
    if k <= n <= 21:
        body = digits + '0' * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + '.' + digits[n:]
    elif -6 < n <= 0:
        body = '0.' + '0' * (-n) + digits
    else:
        mantissa = digits if k == 1 else digits[0] + '.' + digits[1:]
        exponent_sign = '+' if n - 1 >= 0 else '-'
        body = mantissa + 'e' + exponent_sign + str(abs(n - 1))
    return ('-' if number < 0 else '') + body


def js_to_fixed(value, digits):
    """Number.prototype.toFixed(digits): decimal rounding of the exact double, halves away from zero."""
    number = float(value)
    if not math.isfinite(number):
        return js_number_to_string(number)
    if number == 0:
        number = 0.0                            # toFixed prints -0 as 0
    exact = Decimal(number)                     # the exact binary value, as JavaScript sees it
    rounded = exact.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return str(rounded)


def js_to_exponential(value, digits):
    """Number.prototype.toExponential(digits), e.g. 3.66e-9 (no zero padding of the exponent)."""
    number = float(value)
    if not math.isfinite(number):
        return js_number_to_string(number)
    if number == 0:
        return '0.' + '0' * digits + 'e+0' if digits else '0e+0'
    exact = Decimal(abs(number))
    exponent = exact.adjusted()                 # number == mantissa x 10**exponent, 1 <= mantissa < 10
    mantissa = exact.scaleb(-exponent).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    if mantissa >= 10:
        mantissa = mantissa.scaleb(-1).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
        exponent += 1
    sign = '-' if number < 0 else ''
    return '%s%se%s%d' % (sign, mantissa, '+' if exponent >= 0 else '-', abs(exponent))


def utf8_bytes(text):
    """UTF-8 like TextEncoder: a lone surrogate becomes U+FFFD, never an error."""
    try:
        return text.encode('utf-8')
    except UnicodeEncodeError:
        return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'replace').encode('utf-8')


def js_json_string(text):
    """Quote a string the way JSON.stringify does (lowercase \\u escapes)."""
    short_escapes = {'"': '\\"', '\\': '\\\\', '\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
    pieces = ['"']
    for char in text:
        code = ord(char)
        if char in short_escapes:
            pieces.append(short_escapes[char])
        elif code < 0x20 or 0xD800 <= code <= 0xDFFF:
            pieces.append('\\u%04x' % code)
        else:
            pieces.append(char)
    pieces.append('"')
    return ''.join(pieces)


def js_json(value, key_list=None):
    """JSON.stringify(value, key_list) without indentation.

    When key_list is given it plays the role of JSON.stringify's replacer
    array: only those keys are emitted, in that order, at every nesting level.
    """
    if value is None:
        return 'null'
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if is_number(value):
        return js_number_to_string(value) if math.isfinite(value) else 'null'
    if isinstance(value, str):
        return js_json_string(value)
    if isinstance(value, list):
        return '[' + ','.join(js_json(item, key_list) for item in value) + ']'
    if isinstance(value, dict):
        keys = list(value.keys()) if key_list is None else [k for k in key_list if k in value]
        entries = [js_json_string(k) + ':' + js_json(value[k], key_list) for k in keys]
        return '{' + ','.join(entries) + '}'
    raise TypeError('value cannot be serialised: %r' % type(value).__name__)


def utf16_sort_key(text):
    """Sort key matching JavaScript's default (UTF-16 code unit) string order."""
    return text.encode('utf-16-be', 'surrogatepass')


def from_hex(text):
    """Bytes of a hex string.  The receipt-shape check guarantees valid, even-length hex."""
    return bytes.fromhex(text)


def to_hex(data):
    """Mirror of toHex: lowercase hex."""
    return data.hex()


# --------------------------------------------------------------------------
# exp, log and log1p exactly as V8 computes Math.exp, Math.log and Math.log1p
#
# V8 does not call the platform's libm for these; it ships its own copies of
# the fdlibm routines (src/base/ieee754.cc, from FreeBSD's msun).  They are
# reproduced here step by step in plain double arithmetic so that the
# Clopper-Pearson bisection and the EXACT softmax take the same path as the
# browser.  Every operation below is a basic IEEE-754 operation (+ - * /) or a
# bit manipulation of the double's two 32-bit words, and Python floats are
# IEEE-754 doubles, so the results are the same on every platform.  Engines
# compiled with fused multiply-add can differ in the last bit; see the header.
# --------------------------------------------------------------------------


def high_word(value):
    """The upper 32 bits of a double, as an unsigned int (fdlibm's __HI)."""
    return struct.unpack('>II', struct.pack('>d', value))[0]


def low_word(value):
    """The lower 32 bits of a double, as an unsigned int (fdlibm's __LO)."""
    return struct.unpack('>II', struct.pack('>d', value))[1]


def signed32(word):
    """Reinterpret an unsigned 32-bit word as a signed C int32."""
    word &= 0xFFFFFFFF
    return word - (1 << 32) if word >= (1 << 31) else word


def with_high_word(value, word):
    """The double whose upper 32 bits are `word` and whose lower 32 bits are those of `value`."""
    return struct.unpack('>d', struct.pack('>II', word & 0xFFFFFFFF, low_word(value)))[0]


def from_words(high, low):
    """The double with these upper and lower 32-bit words."""
    return struct.unpack('>d', struct.pack('>II', high & 0xFFFFFFFF, low & 0xFFFFFFFF))[0]


def c_int(value):
    """C's (int) cast of a double: truncation toward zero."""
    return int(value)


EXP_HALF = (0.5, -0.5)
EXP_HUGE = 1.0e+300
EXP_TWOM1000 = 9.33263618503218878990e-302     # 2**-1000
EXP_TWO1023 = 8.988465674311579539e307         # 2**1023
EXP_O_THRESHOLD = 7.09782712893383973096e+02
EXP_U_THRESHOLD = -7.45133219101941108420e+02
EXP_LN2HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)
EXP_LN2LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)
EXP_INVLN2 = 1.44269504088896338700e+00
EXP_P1 = 1.66666666666666019037e-01
EXP_P2 = -2.77777777770155933842e-03
EXP_P3 = 6.61375632143793436117e-05
EXP_P4 = -1.65339022054652515390e-06
EXP_P5 = 4.13813679705723846039e-08
EXP_E = 2.718281828459045                      # V8 returns this for exp(1) exactly


def js_exp(x):
    """Math.exp as V8 computes it (fdlibm e_exp.c)."""
    x = float(x)
    hx = high_word(x)
    xsb = (hx >> 31) & 1                        # sign bit of x
    hx &= 0x7FFFFFFF                            # high word of |x|

    if hx >= 0x40862E42:                        # |x| >= 709.78...
        if hx >= 0x7FF00000:                    # infinity or NaN
            if ((hx & 0xFFFFF) | low_word(x)) != 0:
                return x + x                    # NaN
            return x if xsb == 0 else 0.0       # exp(+inf) = inf, exp(-inf) = 0
        if x > EXP_O_THRESHOLD:
            return EXP_HUGE * EXP_HUGE          # overflow
        if x < EXP_U_THRESHOLD:
            return EXP_TWOM1000 * EXP_TWOM1000  # underflow

    if x == 1.0:
        return EXP_E

    hi = 0.0
    lo = 0.0
    k = 0
    if hx > 0x3FD62E42:                         # |x| > 0.5 ln2: argument reduction
        if hx < 0x3FF0A2B2:                     # and |x| < 1.5 ln2
            hi = x - EXP_LN2HI[xsb]
            lo = EXP_LN2LO[xsb]
            k = 1 - xsb - xsb
        else:
            k = c_int(EXP_INVLN2 * x + EXP_HALF[xsb])
            t = float(k)
            hi = x - t * EXP_LN2HI[0]           # t*ln2HI is exact here
            lo = t * EXP_LN2LO[0]
        x = hi - lo
    elif hx < 0x3E300000:                       # |x| < 2**-28
        if EXP_HUGE + x > 1.0:
            return 1.0 + x
    else:
        k = 0

    t = x * x                                   # x is now in the primary range
    if k >= -1021:
        twopk = from_words(0x3FF00000 + (k << 20), 0)
    else:
        twopk = from_words(0x3FF00000 + ((k + 1000) << 20), 0)
    c = x - t * (EXP_P1 + t * (EXP_P2 + t * (EXP_P3 + t * (EXP_P4 + t * EXP_P5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= -1021:
        if k == 1024:
            return y * 2.0 * EXP_TWO1023
        return y * twopk
    return y * twopk * EXP_TWOM1000


LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10
TWO54 = 1.80143985094819840000e+16
LG1 = 6.666666666666735130e-01
LG2 = 3.999999999940941908e-01
LG3 = 2.857142874366239149e-01
LG4 = 2.222219843214978396e-01
LG5 = 1.818357216161805012e-01
LG6 = 1.531383769920937332e-01
LG7 = 1.479819860511658591e-01


def js_log(x):
    """Math.log as V8 computes it (fdlibm e_log.c): -Infinity at 0, NaN for negatives."""
    x = float(x)
    hx = signed32(high_word(x))
    lx = low_word(x)

    k = 0
    if hx < 0x00100000:                         # x < 2**-1022
        if ((hx & 0x7FFFFFFF) | lx) == 0:
            return -math.inf                    # log(+-0) = -inf
        if hx < 0:
            return math.nan                     # log of a negative number
        k -= 54
        x *= TWO54                              # subnormal number: scale up
        hx = signed32(high_word(x))
    if hx >= 0x7FF00000:
        return x + x                            # infinity or NaN
    k += (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = with_high_word(x, hx | (i ^ 0x3FF00000))    # normalize x or x/2
    k += i >> 20
    f = x - 1.0
    if (0x000FFFFF & (2 + hx)) < 3:             # -2**-20 <= f < 2**-20
        if f == 0.0:
            if k == 0:
                return 0.0
            dk = float(k)
            return dk * LN2_HI + dk * LN2_LO
        R = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0:
            return f - R
        dk = float(k)
        return dk * LN2_HI - ((R - dk * LN2_LO) - f)
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147A
    w = z * z
    j = 0x6B851 - hx
    t1 = w * (LG2 + w * (LG4 + w * LG6))
    t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)))
    i |= j
    R = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + R))
        return dk * LN2_HI - ((hfsq - (s * (hfsq + R) + dk * LN2_LO)) - f)
    if k == 0:
        return f - s * (f - R)
    return dk * LN2_HI - ((s * (f - R) - dk * LN2_LO) - f)


LP1 = 6.666666666666735130e-01
LP2 = 3.999999999940941908e-01
LP3 = 2.857142874366239149e-01
LP4 = 2.222219843214978396e-01
LP5 = 1.818357216161805012e-01
LP6 = 1.531383769920937332e-01
LP7 = 1.479819860511658591e-01


def js_log1p(x):
    """Math.log1p as V8 computes it (fdlibm s_log1p.c): -Infinity at -1, NaN below."""
    x = float(x)
    hx = signed32(high_word(x))
    ax = hx & 0x7FFFFFFF

    k = 1
    f = 0.0
    hu = 0
    c = 0.0
    if hx < 0x3FDA827A:                         # 1+x < sqrt(2)+
        if ax >= 0x3FF00000:                    # x <= -1.0
            return -math.inf if x == -1.0 else math.nan
        if ax < 0x3E200000:                     # |x| < 2**-29
            if TWO54 + x > 0.0 and ax < 0x3C900000:     # |x| < 2**-54
                return x
            return x - x * x * 0.5
        if hx > 0 or hx <= signed32(0xBFD2BEC4):        # sqrt(2)/2- <= 1+x < sqrt(2)+
            k = 0
            f = x
            hu = 1
    if hx >= 0x7FF00000:
        return x + x                            # infinity or NaN
    if k != 0:
        if hx < 0x43400000:
            u = 1.0 + x
            hu = signed32(high_word(u))
            k = (hu >> 20) - 1023
            c = (1.0 - (u - x)) if k > 0 else (x - (u - 1.0))    # correction term
            c /= u
        else:
            u = x
            hu = signed32(high_word(u))
            k = (hu >> 20) - 1023
            c = 0.0
        hu &= 0x000FFFFF
        if hu < 0x6A09E:                        # u ~< sqrt(2)
            u = with_high_word(u, hu | 0x3FF00000)      # normalize u
        else:
            k += 1
            u = with_high_word(u, hu | 0x3FE00000)      # normalize u/2
            hu = (0x00100000 - hu) >> 2
        f = u - 1.0
    hfsq = 0.5 * f * f
    if hu == 0:                                 # |f| < 2**-20
        if f == 0.0:
            if k == 0:
                return 0.0
            c += k * LN2_LO
            return k * LN2_HI + c
        R = hfsq * (1.0 - 0.66666666666666666 * f)
        if k == 0:
            return f - R
        return k * LN2_HI - ((R - (k * LN2_LO + c)) - f)
    s = f / (2.0 + f)
    z = s * s
    R = z * (LP1 + z * (LP2 + z * (LP3 + z * (LP4 + z * (LP5 + z * (LP6 + z * LP7))))))
    if k == 0:
        return f - (hfsq - s * (hfsq + R))
    return k * LN2_HI - ((hfsq - (s * (hfsq + R) + (k * LN2_LO + c))) - f)


# --------------------------------------------------------------------------
# Hashing and byte encoding (mirror of hashParts / payloadBytes)
# --------------------------------------------------------------------------


def part_bytes(part):
    """Encode one hash part: str -> UTF-8, number -> float64 big-endian, bytes as is."""
    if isinstance(part, (bytes, bytearray)):
        return bytes(part)
    if isinstance(part, str):
        return utf8_bytes(part)
    if is_number(part):
        return struct.pack('>d', float(part))
    raise TypeError('unsupported hash part: %r' % type(part).__name__)


def hash_parts(*parts):
    """SHA-256 over the parts, each prefixed by its 4-byte big-endian length."""
    encoded = [part_bytes(part) for part in parts]
    message = b''.join(struct.pack('>I', len(chunk)) + chunk for chunk in encoded)
    return hashlib.sha256(message).digest()


def f32_values(data):
    """Round every element to the nearest float32 and return them as doubles.

    This is what Float32Array.from does; values beyond the float32 range become
    +-Infinity rather than an error.
    """
    rounded = []
    for value in data:
        number = float(value)
        try:
            rounded.append(struct.unpack('<f', struct.pack('<f', number))[0])
        except OverflowError:
            rounded.append(math.copysign(math.inf, number))
    return rounded


def payload_bytes(payload):
    """Raw little-endian bytes of a payload: int32 for 'i32', float32 otherwise."""
    data = payload['data']
    if payload['dtype'] == 'i32':
        return struct.pack('<%di' % len(data), *[to_int32(x) for x in data])
    return struct.pack('<%df' % len(data), *f32_values(data))


# --------------------------------------------------------------------------
# Merkle opening verification (mirror of leafHash / treeHeight / verifyOpening)
# --------------------------------------------------------------------------

LEAF = 'leaf'
NODE = 'node'


def leaf_hash(domain, index, salt, payload):
    """Hash of one committed entry: domain-tagged, salted, dtype and length prefixed."""
    return hash_parts(LEAF, domain, index, salt, payload['dtype'], len(payload['data']), payload_bytes(payload))


def tree_height(n):
    """Number of hashing levels above the leaves of a tree with n leaves: ceil(log2(n)), 0 for n <= 1.

    A level with an odd number of nodes pairs its last node with itself, so
    every leaf has exactly this many siblings on its path to the root.
    """
    if not n > 1:
        return 0
    if float(n).is_integer():
        return (int(n) - 1).bit_length()
    return math.ceil(math.log2(n))


def verify_opening(root_hex, domain, opening, n, expect_index, dtype, length):
    """Recompute the root from one opened entry and its sibling path."""
    if root_hex is None:
        return False
    index = opening.get('index')
    payload = opening.get('payload')
    path = opening.get('path')
    if not same_number(index, expect_index) or index < 0 or index >= n or len(path) != tree_height(n):
        return False
    if payload.get('dtype') != dtype or len(payload['data']) != length:
        return False
    node = leaf_hash(domain, index, from_hex(opening.get('salt')), payload)
    position = index
    for sibling_hex in path:
        sibling = from_hex(sibling_hex)
        if js_mod(position, 2) == 0:
            node = hash_parts(NODE, node, sibling)
        else:
            node = hash_parts(NODE, sibling, node)
        # JavaScript's `position >>= 1`: an int32 shift, which is a plain halving here since index < n < 2^31.
        position = to_int32(position) >> 1
    return to_hex(node) == root_hex


def root_at(roots, layer):
    """roots[layer] with JavaScript's tolerance: any invalid position gives None."""
    if not is_number(layer) or not float(layer).is_integer():
        return None
    position = int(layer)
    if 0 <= position < len(roots):
        return roots[position]
    return None


def verify_layer_opening(receipt, kind, layer, opening, expect_index, dtype, length):
    """Verify an opening against the root of tree `kind` (S, q, k, v or o) at `layer`."""
    root_hex = root_at(receipt['roots'][kind], layer)
    if root_hex is None:
        return False
    # The domain tag is the tree kind followed by the layer number as JavaScript prints it: 'S0', 'k3'.
    domain = kind + js_number_to_string(layer)
    return verify_opening(root_hex, domain, opening, whole(receipt['statement'], 'n'), expect_index, dtype, length)


# --------------------------------------------------------------------------
# Statement digest and Fiat-Shamir challenges
# --------------------------------------------------------------------------


def canonical_statement_json(statement):
    """JSON.stringify(statement, Object.keys(statement).sort()).

    The keys are sorted in JavaScript's default order (by UTF-16 code unit),
    emitted without whitespace, and the replacer array filters keys at every
    nesting level (which changes nothing today: every statement value is a
    scalar).
    """
    sorted_keys = sorted(statement.keys(), key=utf16_sort_key)
    return js_json(statement, sorted_keys)


def statement_digest(statement):
    """Mirror of statementDigest."""
    return hash_parts('statement', canonical_statement_json(statement))


def roots_list(roots):
    """All roots in the fixed order S, q, k, v, o, as bytes."""
    hex_roots = roots['S'] + roots['q'] + roots['k'] + roots['v'] + roots['o']
    return [from_hex(root) for root in hex_roots]


def fiat_shamir(digest, roots, nonce, label, count, modulus):
    """Mirror of fiatShamir.

    The seed binds the label, the statement digest, the nonce and then every
    root (S, q, k, v, o order).  Each counter value yields one fresh SHA-256 of
    (seed, counter), of which the first 32-bit big-endian word is reduced
    modulo `modulus`.  Every modulus used here is far below 2^32, so the bias
    of the reduction is negligible and no value exceeds the 53-bit exactness of
    a double.
    """
    seed = hash_parts('fs', label, digest, nonce, *roots)
    picks = []
    counter = 0
    while len(picks) < count:
        word = struct.unpack('>I', hash_parts(seed, counter)[:4])[0]
        picks.append(js_mod(word, modulus))
        counter += 1
    return picks


def challenges(receipt, label, count, modulus):
    """The challenge values the nonce picks for `label` ('rows', 'pairs' or 'tokens')."""
    digest = statement_digest(receipt['statement'])
    return fiat_shamir(digest, roots_list(receipt['roots']), receipt['nonce'], label, count, modulus)


def row_from_challenge(value, L):
    """A row challenge encodes (layer, token index) as value = index * L + layer."""
    return js_mod(value, L), js_floor(js_div(value, L))


def pair_from_challenge(value, n):
    """A pair challenge encodes two token positions; returned as (earlier, later)."""
    a = js_floor(js_div(value, n))
    b = js_mod(value, n)
    return (a, b) if a < b else (b, a)


# --------------------------------------------------------------------------
# Statistics (mirror of clopperPearsonLower / hoeffdingHalfWidth)
# --------------------------------------------------------------------------


def log_add(a, b):
    """log(exp(a) + exp(b)) computed stably."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + js_log1p(js_exp(b - a))
    return b + js_log1p(js_exp(a - b))


def log_binom_tail(n, k, p):
    """log P[Binomial(n, p) >= k], summed term by term in log space."""
    if k <= 0:
        return 0
    if p <= 0:
        return -math.inf
    log_term = n * js_log1p(-p)
    acc = -math.inf
    x = 0
    while x <= n:
        if x >= k:
            acc = log_add(acc, log_term)
        log_term += js_log(js_div(n - x, x + 1)) + js_log(p) - js_log1p(-p)
        x += 1
    return acc


def clopper_pearson_lower(k, n):
    """One-sided 95 % Clopper-Pearson lower bound, by 60 bisection steps."""
    if n == 0 or k == 0:
        return 0
    lo = 0
    hi = k / n
    threshold = js_log(0.05)
    for _ in range(60):
        mid = (lo + hi) / 2
        if log_binom_tail(n, k, mid) < threshold:
            lo = mid
        else:
            hi = mid
    return lo


def hoeffding_half_width(B, t):
    """95 % Hoeffding half-width for a mean of t values in [0, B]."""
    return B * math.sqrt(js_log(2 / 0.05) / (2 * max(t, 1)))


# --------------------------------------------------------------------------
# The seven checks
# --------------------------------------------------------------------------


def whole(statement, key):
    """A count field of the statement as an int (the shape check guarantees a whole number)."""
    return int(statement[key])


def active_set(opening):
    """The reading list stored in an S opening: the non-negative entries (-1 is padding)."""
    return [x for x in opening['payload']['data'] if x >= 0]


def set_is_sound(reading_list, i, B):
    """Non-empty, at most B entries, contains i itself, ends at or before i, strictly increasing."""
    if not 0 < len(reading_list) <= B:
        return False
    if i not in reading_list or reading_list[-1] > i:
        return False
    return all(reading_list[idx] > reading_list[idx - 1] for idx in range(1, len(reading_list)))


def verify_row_set(receipt, row):
    """Open the reading list of a challenged row; None if the opening or the list is unsound."""
    B = whole(receipt['statement'], 'B')
    ok = verify_layer_opening(receipt, 'S', row.get('layer'), row['S'], row.get('index'), 'i32', B)
    reading_list = active_set(row['S'])
    if ok and set_is_sound(reading_list, row.get('index'), B):
        return reading_list
    return None


def softmax_over_opened_keys(q_row, k_rows, v_rows, heads, d):
    """Ordinary softmax attention for one query over exactly the opened keys and values.

    One softmax per head.  Inputs are float32 values (as doubles); all arithmetic
    is double precision, in the same order as the browser verifier.
    """
    out = []
    scale = js_div(1, math.sqrt(d)) if d >= 0 else math.nan
    for h in range(heads):
        start, stop = h * d, (h + 1) * d
        qi = q_row[start:stop]
        scores = []
        for k_row in k_rows:
            kj = k_row[start:stop]
            dot = 0.0
            for t in range(d):
                dot += qi[t] * kj[t]
            scores.append(dot * scale)
        top = js_max(scores)
        exps = [js_exp(s - top) for s in scores]
        denominator = 0.0
        for value in exps:
            denominator += value
        acc = [0.0] * d
        for idx, v_row in enumerate(v_rows):
            vj = v_row[start:stop]
            weight = js_div(exps[idx], denominator)
            for t in range(d):
                acc[t] += weight * vj[t]
        out.extend(acc)
    return out


def verify_row_exact(receipt, row, reading_list):
    """Recompute the output row from the opened q, k, v; return the max error or -1 on failure."""
    statement = receipt['statement']
    heads, head_dim, tol = whole(statement, 'heads'), whole(statement, 'headDim'), statement['tol']
    width = heads * head_dim
    layer, index = row.get('layer'), row.get('index')
    q_opening, o_opening = row.get('q'), row.get('o')
    if q_opening is None or o_opening is None:
        return -1
    if root_at(receipt['roots']['q'], layer) is None or root_at(receipt['roots']['o'], layer) is None:
        return -1
    ok_q = verify_layer_opening(receipt, 'q', layer, q_opening, index, 'f32', width)
    ok_o = verify_layer_opening(receipt, 'o', layer, o_opening, index, 'f32', width)
    keys = row['keys']
    keys_ok = len(keys) == len(reading_list) and all(
        same_number(kv.get('index'), reading_list[idx])
        and verify_layer_opening(receipt, 'k', layer, kv['k'], kv.get('index'), 'f32', width)
        and verify_layer_opening(receipt, 'v', layer, kv['v'], kv.get('index'), 'f32', width)
        for idx, kv in enumerate(keys)
    )
    if not (ok_q and ok_o and keys_ok):
        return -1
    k_rows = [f32_values(kv['k']['payload']['data']) for kv in keys]
    v_rows = [f32_values(kv['v']['payload']['data']) for kv in keys]
    q_row = f32_values(q_opening['payload']['data'])
    o_row = o_opening['payload']['data']          # compared as the plain numbers in the receipt
    reference = softmax_over_opened_keys(q_row, k_rows, v_rows, heads, head_dim)
    err = js_max([abs(value - o_row[t]) for t, value in enumerate(reference)])
    return err if err <= tol else -1


def check_result(name, checked, passed, must_be_all):
    """Summarise one check.

    Nothing challenged means "not measured", never a failure.  BOUNDED, EXACT
    and WEIGHTS (must_be_all) need every challenged item to pass.  RANGE,
    HEARDBY, DIRECT and CONNECTED are fractions: they are ok whenever at least
    one item passed, and the informative numbers are `fraction` and
    `lowerBound` (the 95 % Clopper-Pearson lower bound for this run), not `ok`.
    """
    if checked == 0:
        ok = True
    elif must_be_all:
        ok = passed == checked
    else:
        ok = passed > 0
    return {
        'name': name,
        'ok': ok,
        'measured': checked > 0,
        'checked': checked,
        'passed': passed,
        'fraction': passed / checked if checked else 0,
        'lowerBound': clopper_pearson_lower(passed, checked),
    }


def verify_rows(receipt):
    """BOUNDED, EXACT and RANGE over the challenged rows."""
    statement = receipt['statement']
    n, L, far_frac = whole(statement, 'n'), whole(statement, 'L'), statement['farFrac']
    exact_rows = whole(statement, 'exactRows')
    # The products are formed in double arithmetic, as JavaScript does (they only differ above 2^53).
    expected = challenges(receipt, 'rows', whole(statement, 'rows'), float(n) * float(L))
    sizes = []
    bounded = exact = far_candidates = far = 0
    worst_err = 0
    for idx, row in enumerate(receipt['openings']['rows']):
        if idx >= len(expected):
            continue
        layer, index = row_from_challenge(expected[idx], L)
        if not same_number(row.get('layer'), layer) or not same_number(row.get('index'), index):
            continue
        reading_list = verify_row_set(receipt, row)
        if reading_list is None:
            continue
        bounded += 1
        sizes.append(len(reading_list))
        if idx < exact_rows:
            err = verify_row_exact(receipt, row, reading_list)
            if err >= 0:
                exact += 1
                worst_err = max(worst_err, err)
        # RANGE is only meaningful once a token has some history behind it (position 64 onwards).
        if index >= 64:
            far_candidates += 1
            if index - reading_list[0] >= far_frac * index:
                far += 1
    results = [
        check_result('BOUNDED', len(expected), bounded, True),
        check_result('EXACT', min(exact_rows, len(expected)), exact, True),
        check_result('RANGE', far_candidates, far, False),
    ]
    return results, sizes, worst_err


def verify_heard_by(receipt):
    """HEARDBY: each challenged token appears in the reading list of a later token."""
    statement = receipt['statement']
    n, B = whole(statement, 'n'), whole(statement, 'B')
    expected = challenges(receipt, 'tokens', whole(statement, 'tokens'), max(n - 64, 1))
    by_token = {entry.get('token'): entry for entry in receipt['openings']['heardBy']}
    passed = 0
    for token in expected:
        entry = by_token.get(token)
        if entry is None or entry['reader'].get('index') <= token:
            continue
        reader = entry['reader']
        if not verify_layer_opening(receipt, 'S', entry.get('layer'), reader, reader.get('index'), 'i32', B):
            continue
        if token in active_set(reader):
            passed += 1
    return check_result('HEARDBY', len(expected), passed, False)


def verify_hops(receipt, pair):
    """Length of the verified chain from pair['j'] to pair['i'], or 0 if it does not hold up."""
    hops = pair.get('hops')
    if hops is None or len(hops) == 0 or not same_number(hops[-1].get('node'), pair.get('i')):
        return 0
    B = whole(receipt['statement'], 'B')
    previous = pair.get('j')
    last_layer = -1
    for hop in hops:
        layer, node = hop.get('layer'), hop.get('node')
        ok = (
            layer > last_layer
            and previous <= node
            and verify_layer_opening(receipt, 'S', layer, hop['S'], node, 'i32', B)
            and previous in active_set(hop['S'])
        )
        if not ok:
            return 0
        previous = node
        last_layer = layer
    return len(hops)


def verify_pairs(receipt):
    """DIRECT and CONNECTED over the challenged pairs."""
    statement = receipt['statement']
    n = whole(statement, 'n')
    expected = challenges(receipt, 'pairs', whole(statement, 'pairs'), float(n) * float(n))
    opened = receipt['openings']['pairs']
    checked = direct = connected = 0
    for idx, value in enumerate(expected):
        j, i = pair_from_challenge(value, n)
        if i == j:
            continue
        checked += 1
        pair = opened[idx] if idx < len(opened) else None
        if pair is None or not same_number(pair.get('j'), j) or not same_number(pair.get('i'), i):
            continue
        hops = verify_hops(receipt, pair)
        if hops >= 1:
            connected += 1
        if hops == 1:
            direct += 1
    return [
        check_result('DIRECT', checked, direct, False),
        check_result('CONNECTED', checked, connected, False),
    ]


def verify(receipt):
    """Verify a receipt that passed assert_receipt_shape; returns the same fields as the browser verifier."""
    statement = receipt['statement']
    n, B = whole(statement, 'n'), whole(statement, 'B')
    row_results, sizes, worst_err = verify_rows(receipt)
    # WEIGHTS only compares two strings the receipt itself carries (modelHash and baseModelHash);
    # this tool has no list of published model hashes and cannot say which model either names.
    # Both are strings here (shape check), so Python's == is JavaScript's === on them.
    weights_ok = 1 if statement['modelHash'] == statement['baseModelHash'] else 0
    weights = check_result('WEIGHTS', 1, weights_ok, True)
    results = row_results + [verify_heard_by(receipt)] + verify_pairs(receipt) + [weights]
    total = 0
    for size in sizes:
        total += size
    mean_degree = total / len(sizes) if sizes else 0
    hoeffding = hoeffding_half_width(B, len(sizes))
    return {
        'ok': all(result['ok'] for result in results),
        'maxDegree': max(sizes) if sizes else 0,
        'worstExactErr': worst_err,
        'meanDegree': mean_degree,
        'hoeffding': hoeffding,
        'readsLower': n * max(1, mean_degree - hoeffding),
        'readsUpper': n * min(B, mean_degree + hoeffding),
        'readsDense': (float(n) * (float(n) + 1)) / 2,
        'results': results,
    }


# --------------------------------------------------------------------------
# Receipt shape (the same checks and messages as verify.mjs)
#
# The browser verifier coerces a few odd value types the way JavaScript does
# (a numeric string as a layer, null as a payload entry, ...).  These scripts
# refuse such receipts up front with exit code 2 instead: the /proof page
# never produces them, and refusing them keeps every later comparison a plain
# comparison of numbers and strings.  Hex fields must be valid even-length hex
# too (the browser would report a failed check instead).
# --------------------------------------------------------------------------


class ReceiptError(ValueError):
    """Anything that should end with exit code 2."""


HEX_RE = re.compile(r'^(?:[0-9a-fA-F]{2})*$')
STATEMENT_NUMBERS = ['n', 'L', 'B', 'heads', 'headDim', 'tol', 'farFrac', 'rows', 'exactRows', 'pairs', 'tokens']
STATEMENT_COUNTS = ['n', 'L', 'B', 'heads', 'headDim', 'rows', 'exactRows', 'pairs', 'tokens']


def require(condition, message):
    if not condition:
        raise ReceiptError(message)


def is_finite_number(value):
    return is_number(value) and math.isfinite(value)


def is_hex_string(value):
    return isinstance(value, str) and HEX_RE.match(value) is not None


def check_opening(opening, path):
    """One Merkle opening: index, salt, payload {dtype, data} and sibling path."""
    require(isinstance(opening, dict), path + ' is not an object')
    require(is_finite_number(opening.get('index')), path + '.index is not a number')
    require(is_hex_string(opening.get('salt')), path + '.salt is not a hex string')
    payload = opening.get('payload')
    require(isinstance(payload, dict), path + '.payload is missing')
    require(isinstance(payload.get('dtype'), str), path + '.payload.dtype is not a string')
    data = payload.get('data')
    require(isinstance(data, list) and all(is_number(x) for x in data), path + '.payload.data is not an array of numbers')
    siblings = opening.get('path')
    require(isinstance(siblings, list) and all(is_hex_string(x) for x in siblings), path + '.path is not an array of hex strings')


def check_row(row, path):
    """One opened row: layer, index, S, optional q and o, and keys (required with q and o)."""
    require(isinstance(row, dict), path + ' is not an object')
    require(is_finite_number(row.get('layer')), path + '.layer is not a number')
    require(is_finite_number(row.get('index')), path + '.index is not a number')
    check_opening(row.get('S'), path + '.S')
    for kind in ('q', 'o'):
        if row.get(kind) is not None:
            check_opening(row[kind], path + '.' + kind)
    keys = row.get('keys')
    if keys is not None or (row.get('q') is not None and row.get('o') is not None):
        require(isinstance(keys, list), path + '.keys is not an array')
        for position, entry in enumerate(keys):
            entry_path = '%s.keys[%d]' % (path, position)
            require(isinstance(entry, dict), entry_path + ' is not an object')
            require(is_finite_number(entry.get('index')), entry_path + '.index is not a number')
            check_opening(entry.get('k'), entry_path + '.k')
            check_opening(entry.get('v'), entry_path + '.v')


def check_heard_by_entry(entry, path):
    require(isinstance(entry, dict), path + ' is not an object')
    require(is_finite_number(entry.get('token')), path + '.token is not a number')
    require(is_finite_number(entry.get('layer')), path + '.layer is not a number')
    check_opening(entry.get('reader'), path + '.reader')


def check_pair(pair, path):
    require(isinstance(pair, dict), path + ' is not an object')
    require(is_finite_number(pair.get('j')), path + '.j is not a number')
    require(is_finite_number(pair.get('i')), path + '.i is not a number')
    hops = pair.get('hops')
    if hops is None:
        return
    require(isinstance(hops, list), path + '.hops is not an array')
    for position, hop in enumerate(hops):
        hop_path = '%s.hops[%d]' % (path, position)
        require(isinstance(hop, dict), hop_path + ' is not an object')
        require(is_finite_number(hop.get('layer')), hop_path + '.layer is not a number')
        require(is_finite_number(hop.get('node')), hop_path + '.node is not a number')
        check_opening(hop.get('S'), hop_path + '.S')


def assert_receipt_shape(receipt):
    """Raise ReceiptError unless the parsed JSON has the shape of a receipt."""
    require(isinstance(receipt, dict), 'receipt is not a JSON object')
    statement = receipt.get('statement')
    require(isinstance(statement, dict), 'receipt.statement is missing')
    for key in STATEMENT_NUMBERS:
        require(is_number(statement.get(key)), 'receipt.statement.%s is not a number' % key)
        require(math.isfinite(statement[key]), 'receipt.statement.%s is not a finite number' % key)
    for key in STATEMENT_COUNTS:
        require(float(statement[key]).is_integer() and statement[key] >= 0, 'receipt.statement.%s is not a whole number' % key)
    for key in ('modelHash', 'baseModelHash'):
        require(isinstance(statement.get(key), str), 'receipt.statement.%s is not a string' % key)
    require(isinstance(receipt.get('nonce'), str), 'receipt.nonce is not a string')
    roots = receipt.get('roots')
    require(isinstance(roots, dict), 'receipt.roots is missing')
    for key in ('S', 'q', 'k', 'v', 'o'):
        require(isinstance(roots.get(key), list), 'receipt.roots.%s is not an array' % key)
        for position, root in enumerate(roots[key]):
            require(is_hex_string(root), 'receipt.roots.%s[%d] is not a hex string' % (key, position))
    openings = receipt.get('openings')
    require(isinstance(openings, dict), 'receipt.openings is missing')
    for key in ('rows', 'heardBy', 'pairs'):
        require(isinstance(openings.get(key), list), 'receipt.openings.%s is not an array' % key)
    for position, row in enumerate(openings['rows']):
        check_row(row, 'receipt.openings.rows[%d]' % position)
    for position, entry in enumerate(openings['heardBy']):
        check_heard_by_entry(entry, 'receipt.openings.heardBy[%d]' % position)
    for position, pair in enumerate(openings['pairs']):
        check_pair(pair, 'receipt.openings.pairs[%d]' % position)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

LABELS = {
    'BOUNDED': 'Reads at most B tokens',
    'EXACT': 'Ordinary attention, nothing approximated',
    'RANGE': 'Reaches far back',
    'HEARDBY': 'No token is dropped',
    'DIRECT': 'Read directly (measured fraction)',
    'CONNECTED': 'Linked by a short chain (measured fraction)',
    'WEIGHTS': 'Same model, no retraining',
}

CANNOT_CHECK_NOTE = (
    'This tool cannot check these values; compare them with what the /proof page shows, and '
    'treat the challenge as sound only if the nonce is one you supplied or that the prover '
    'could not have chosen (e.g. a value fixed after the roots were published).'
)

USAGE_EPILOG = """\
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
small tolerance on the floating-point fields (see the README)."""


def status_word(result):
    """The status column of the table."""
    if not result['measured']:
        return 'not measured'
    return 'ok' if result['ok'] else 'FAIL'


def percent(value):
    """A fraction as JavaScript prints (100 * x).toFixed(1) + ' %'."""
    return js_to_fixed(100 * value, 1) + ' %'


def detail_text(result):
    """The last column: the measured fraction and its lower bound."""
    if result['name'] == 'WEIGHTS':
        return 'deterministic (modelHash compared with baseModelHash)'
    if not result['measured']:
        return ''
    return '%s passed, 95 %% lower bound %s' % (percent(result['fraction']), percent(result['lowerBound']))


def check_line(result):
    """One line of the table."""
    tally = js_number_to_string(result['passed']) + '/' + js_number_to_string(result['checked'])
    label = LABELS.get(result['name'], result['name'])
    line = '%-10s %-44s %9s  %-12s %s' % (result['name'], label, tally, status_word(result), detail_text(result))
    return line.rstrip()


def not_measured_lines(verification, statement):
    """Why each unmeasured check has nothing to say."""
    unmeasured = [r['name'] for r in verification['results'] if not r['measured']]
    lines = []
    by_request = [name for name in ('EXACT', 'HEARDBY', 'DIRECT', 'CONNECTED') if name in unmeasured]
    if by_request:
        fields = []
        if 'EXACT' in by_request:
            fields.append(('exactRows' if whole(statement, 'exactRows') == 0 else 'rows') + ' = 0')
        if 'HEARDBY' in by_request:
            fields.append('tokens = 0')
        if 'DIRECT' in by_request:
            fields.append('pairs = 0' if whole(statement, 'pairs') == 0 else 'pairs naming the same token twice only')
        lines.append(
            'Not measured: %s - the statement requests %s, so no rows, tokens or pairs were opened for '
            'these checks (typical for a receipt produced by the 8080.AI server; browser receipts open all '
            'seven). These checks are neither passed nor failed.' % (', '.join(by_request), ', '.join(fields)))
    if 'RANGE' in unmeasured:
        lines.append('Not measured: RANGE - no verified reading list belongs to a token at position 64 or later.')
    if 'BOUNDED' in unmeasured:
        lines.append('Not measured: BOUNDED - the statement requests rows = 0, so no reading list was opened.')
    return lines


def shown_text(value):
    """A receipt string for the summary: as it is when plain printable ASCII, JSON-quoted otherwise."""
    if isinstance(value, str) and all(0x20 <= ord(char) <= 0x7E for char in value):
        return value
    return js_json(value)


def summary_lines(receipt, verification):
    """The summary block: receipt parameters, opened rows, the reads band, and the values taken as given."""
    statement = receipt['statement']
    opened = verification['results'][0]['passed']
    exact = verification['results'][1]
    lines = ['Receipt: n = %s tokens, L = %s layers, B = %s (reading list cap), tol = %s, farFrac = %s' % (
        js_number_to_string(statement['n']), js_number_to_string(statement['L']), js_number_to_string(statement['B']),
        js_number_to_string(statement['tol']), js_number_to_string(statement['farFrac']))]
    if opened > 0:
        opened_line = 'Reading lists opened: %s, mean length %s, longest %s' % (
            js_number_to_string(opened), js_to_fixed(verification['meanDegree'], 2),
            js_number_to_string(verification['maxDegree']))
        if exact['measured']:
            opened_line += ', worst EXACT error %s (tol %s)' % (
                js_to_exponential(verification['worstExactErr'], 2), js_number_to_string(statement['tol']))
        lines.append(opened_line)
        lines.append('Reads per layer, estimated from the opened rows (95 %% band): between %s and %s; '
                     '%s if every token read every earlier token' % (
                         js_number_to_string(js_round(verification['readsLower'])),
                         js_number_to_string(js_round(verification['readsUpper'])),
                         js_number_to_string(verification['readsDense'])))
    else:
        lines.append('Reading lists opened: 0')
        lines.append('Reads per layer: not estimated (no reading list verified)')
    lines.append('Values this tool takes as given:')
    for key in ('modelHash', 'baseModelHash', 'inputHash', 'ruleCommit'):
        lines.append('  %-14s %s' % (key, shown_text(statement.get(key)) if key in statement else '(absent)'))
    lines.append('  %-14s %s' % ('nonce', js_json(receipt['nonce'])))
    lines.append(CANNOT_CHECK_NOTE)
    return lines


def verdict_line(verification):
    """The final line, shared by every output mode."""
    measured = [r for r in verification['results'] if r['measured']]
    failed = [r['name'] for r in measured if not r['ok']]
    unmeasured = len(verification['results']) - len(measured)
    if failed:
        text = 'Verdict: FAIL - %d of %d measured checks failed (%s)' % (len(failed), len(measured), ', '.join(failed))
    else:
        text = 'Verdict: PASS - all %d measured checks passed' % len(measured)
    if unmeasured:
        text += ', %d not measured' % unmeasured
    return text


def report_lines(receipt, verification):
    """Everything table mode prints before the verdict."""
    lines = [check_line(result) for result in verification['results']]
    lines.append('')
    explanations = not_measured_lines(verification, receipt['statement'])
    if explanations:
        lines += explanations
        lines.append('')
    lines += summary_lines(receipt, verification)
    return lines


def json_text(value, depth=0):
    """JSON.stringify(value, null, 1): one space per nesting level, JavaScript number format."""
    pad = ' ' * (depth + 1)
    if isinstance(value, dict):
        if not value:
            return '{}'
        entries = [pad + js_json_string(k) + ': ' + json_text(v, depth + 1) for k, v in value.items()]
        return '{\n' + ',\n'.join(entries) + '\n' + ' ' * depth + '}'
    if isinstance(value, list):
        if not value:
            return '[]'
        entries = [pad + json_text(v, depth + 1) for v in value]
        return '[\n' + ',\n'.join(entries) + '\n' + ' ' * depth + ']'
    return js_json(value)


def reject_constant(name):
    """JSON.parse has no NaN or Infinity literals; neither do we."""
    raise ValueError('invalid JSON literal %s' % name)


def load_receipt(path):
    """Read, parse and shape-check the receipt; raises ReceiptError."""
    try:
        with open(path, 'rb') as handle:
            raw = handle.read()
    except OSError as error:
        raise ReceiptError('cannot read %s: %s' % (path, error)) from error
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        raise ReceiptError('%s is UTF-16 encoded - save it as UTF-8 (PowerShell: Set-Content -Encoding utf8, '
                           'or re-download from the /proof page)' % path)
    try:
        text = raw.decode('utf-8-sig')          # accepts a UTF-8 byte-order mark, as the browser does
    except UnicodeDecodeError as error:
        raise ReceiptError('%s is not UTF-8 text: %s' % (path, error)) from error
    try:
        receipt = json.loads(text, parse_constant=reject_constant)
    except ValueError as error:
        raise ReceiptError('%s is not valid JSON: %s' % (path, error)) from error
    assert_receipt_shape(receipt)
    return receipt


def build_parser():
    parser = argparse.ArgumentParser(
        description='Verify a proof receipt downloaded from the 8080.AI /proof page.',
        epilog=USAGE_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('receipt', help='path to the receipt JSON file')
    parser.add_argument('--json', action='store_true',
                        help='print the report as JSON (same fields as samples/expected/*.json)')
    parser.add_argument('--quiet', action='store_true', help='print only the verdict line')
    return parser


def use_unix_newlines():
    """Print LF line endings everywhere so the output matches verify.mjs byte for byte."""
    try:
        sys.stdout.reconfigure(newline='\n')
    except AttributeError:
        pass


def main(argv=None):
    use_unix_newlines()
    args = build_parser().parse_args(argv)

    try:
        receipt = load_receipt(args.receipt)
    except ReceiptError as error:
        print('error: %s' % error, file=sys.stderr)
        return 2
    if 'version' in receipt and receipt['version'] != 1:
        print('warning: receipt version %s is not the version this tool knows (1)' % js_json(receipt['version']),
              file=sys.stderr)
    try:
        verification = verify(receipt)
    except Exception as error:  # a shape-checked receipt never gets here; anything that does is malformed
        print('error: malformed receipt: %s: %s' % (type(error).__name__, error), file=sys.stderr)
        return 2

    if args.json:
        print(json_text(verification))
    elif args.quiet:
        print(verdict_line(verification))
    else:
        for line in report_lines(receipt, verification):
            print(line)
        print(verdict_line(verification))
    return 0 if verification['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
