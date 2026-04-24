# Schema watch

Proactive tool-schema change detection — runs *before* replay, takes
~20 ms, catches the most common silent regressions.

```bash
shadow schema-watch config_a.yaml config_b.yaml
```

## What it detects

Eleven classes of change, classified into four severity tiers:

### Breaking

- Tool removed
- Required parameter added
- Parameter removed
- **Parameter renamed** (detected via type + required-status matching — see below)
- Type changed
- Required flipped (non-required → required)
- Enum narrowed

### Risky

- Description edited to remove imperative verbs ("ONLY", "MUST", "BEFORE")
- Required flipped (required → non-required)

### Additive

- Tool added
- Parameter added (optional)
- Enum broadened

### Neutral

- Description edited (cosmetic only)

## Rename detection

A removed parameter and an added parameter on the same tool with the
**same type** and **same required status** (threshold 0.6 confidence)
are flagged as a **rename** rather than two separate changes. This
catches the most common silent breaking change in practice:

```yaml
# Before
tools:
  - name: execute_sql
    input_schema:
      properties: { database: {type: string}, query: {type: string} }
      required: [database, query]

# After
tools:
  - name: execute_sql
    input_schema:
      properties: { db: {type: string}, query: {type: string} }  # renamed!
      required: [db, query]
```

Schema-watch emits:

```
✖ BREAKING  execute_sql: parameter renamed `database` → `db`
```

Instead of the greedy remove+add:

```
✖ BREAKING  execute_sql: parameter `database` removed
+ ADDITIVE  execute_sql: parameter `db` added (required)
```

## Output formats

- `shadow schema-watch ... --format terminal` (default): rich console
- `shadow schema-watch ... --format markdown`: GitHub-flavoured table
  + expandable rationale
- `shadow schema-watch ... --format json`: machine-readable

## CI gate

Exit code 1 on any breaking change by default. Tune with `--fail-on`:

```bash
# Fail only on breaking:
shadow schema-watch cfg_a.yaml cfg_b.yaml --fail-on breaking

# Fail on anything risky or worse:
shadow schema-watch cfg_a.yaml cfg_b.yaml --fail-on risky

# Never fail, always exit 0:
shadow schema-watch cfg_a.yaml cfg_b.yaml --fail-on none
```
