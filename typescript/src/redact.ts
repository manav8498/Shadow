/**
 * Default PII/secret redactor — parity with python/src/shadow/redact.
 *
 * Applied at record boundaries before canonical JSON + content-id, so
 * redacted content is what ends up hashed and persisted. Covers:
 *
 *   - OpenAI keys (sk-, sk-proj-, sk-svcacct-, sk-admin-)
 *   - Anthropic keys (sk-ant-)
 *   - AWS access key IDs (AKIA/ASIA/AIDA/AROA + 16 uppercase alnum)
 *   - GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_)
 *   - PEM private keys
 *   - JWT tokens
 *   - Email addresses
 *   - E.164 phone numbers
 *   - Credit-card PANs (Luhn-validated, 13-19 digits, Amex layout)
 *
 * Recursive over nested objects/arrays. Per-key allowlist bypass.
 * Luhn validation rejects non-card numeric lookalikes.
 */

export interface RedactionPattern {
  readonly name: string;
  readonly regex: RegExp;
  readonly replacement: string;
}

// NOTE: ordering matters. More-specific prefixes first so e.g. `sk-ant-`
// isn't consumed by the generic `sk-` rule.
export const DEFAULT_PATTERNS: readonly RedactionPattern[] = [
  {
    name: 'private_key',
    regex: /-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY-----[\s\S]{20,}?-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY-----/g,
    replacement: '[REDACTED:private_key]',
  },
  {
    name: 'jwt',
    regex: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{20,}\b/g,
    replacement: '[REDACTED:jwt]',
  },
  {
    name: 'anthropic_api_key',
    regex: /sk-ant-[A-Za-z0-9_-]{20,}/g,
    replacement: '[REDACTED:anthropic_api_key]',
  },
  {
    name: 'openai_api_key',
    regex: /sk-(?!ant-)(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}/g,
    replacement: '[REDACTED:openai_api_key]',
  },
  {
    name: 'aws_access_key_id',
    regex: /\b(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}\b/g,
    replacement: '[REDACTED:aws_access_key_id]',
  },
  {
    name: 'github_token',
    regex: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,251}\b/g,
    replacement: '[REDACTED:github_token]',
  },
  {
    name: 'email',
    regex: /[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}/g,
    replacement: '[REDACTED:email]',
  },
  {
    name: 'phone',
    regex: /\+[1-9]\d{9,14}(?!\d)/g,
    replacement: '[REDACTED:phone]',
  },
];

// Credit card: handled separately so we can Luhn-validate each match.
const CREDIT_CARD_RE =
  /(?<![0-9])(?:\d{13,19}|\d{4}(?:[\s\-]\d{4}){2,3}\d{0,3}|\d{4}[\s\-]\d{6}[\s\-]\d{5})(?![0-9])/g;

/** Standard Luhn check, 13-19 digit range (ISO/IEC 7812). */
export function luhnValid(digits: string): boolean {
  if (digits.length < 13 || digits.length > 19) return false;
  let total = 0;
  for (let i = 0; i < digits.length; i++) {
    const ch = digits[digits.length - 1 - i];
    if (ch < '0' || ch > '9') return false;
    let n = ch.charCodeAt(0) - 48;
    if (i % 2 === 1) {
      n *= 2;
      if (n > 9) n -= 9;
    }
    total += n;
  }
  return total % 10 === 0;
}

export interface RedactorOptions {
  /** Keys in this set bypass redaction for their value subtrees. */
  allowlistKeys?: ReadonlySet<string>;
  /** Override the default pattern set. */
  patterns?: readonly RedactionPattern[];
}

export class Redactor {
  private readonly patterns: readonly RedactionPattern[];
  private readonly allowlist: ReadonlySet<string>;

  constructor(opts: RedactorOptions = {}) {
    this.patterns = opts.patterns ?? DEFAULT_PATTERNS;
    this.allowlist = opts.allowlistKeys ?? new Set();
  }

  /** Redact a JSON-shaped value in place. Returns a new value. */
  redactValue<T>(value: T): T {
    return this.walk(value, null) as T;
  }

  private walk(value: unknown, parentKey: string | null): unknown {
    if (typeof value === 'string') {
      return this.redactText(value);
    }
    if (Array.isArray(value)) {
      return value.map((v) => this.walk(v, parentKey));
    }
    if (value !== null && typeof value === 'object') {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(value)) {
        if (this.allowlist.has(k)) {
          out[k] = v;
        } else {
          out[k] = this.walk(v, k);
        }
      }
      return out;
    }
    return value;
  }

  private redactText(text: string): string {
    let out = text;
    for (const p of this.patterns) {
      // `regex` is /g; reset lastIndex implicitly by using replace.
      out = out.replace(p.regex, p.replacement);
    }
    // Credit cards: regex + Luhn.
    out = out.replace(CREDIT_CARD_RE, (match) => {
      const digitsOnly = match.replace(/[^0-9]/g, '');
      return luhnValid(digitsOnly) ? '[REDACTED:credit_card]' : match;
    });
    return out;
  }
}
