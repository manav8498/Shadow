# Streaming replay

The `chunk` record kind captures one chunk of a streaming model response so a candidate run can replay the original chunk timing instead of fabricating its own.

## Why absolute time, not relative

Each `chunk` payload carries an absolute `time_unix_nano`, not a delta from the previous chunk. Cumulative `sleep(delta)` accumulates rounding error on long streams (a 30-second response with 200 chunks can drift hundreds of milliseconds on a typical event loop). Storing absolute timestamps lets the replay engine compute deadlines monotonically, so each chunk is yielded at the correct wall-clock moment regardless of host clock skew or replay speed multiplier.

## Recording

```python
from shadow.sdk import Session
from shadow.v02_records import record_chunk

with Session(output_path="trace.agentlog") as s:
    for i, chunk in enumerate(provider_stream):
        record_chunk(
            s,
            chunk_index=i,
            delta=chunk.delta_dict(),
            is_final=(i == last_index),
        )
```

`delta` is a passthrough of the provider's per-chunk delta — Anthropic `text_delta` / `input_json_delta` / `thinking_delta`, OpenAI `{content?, tool_calls?[]}`. Shadow doesn't interpret it; only the per-provider streaming aggregator at recording time and the differ at comparison time look inside.

## Replay

```python
from shadow.v02_records import replay_chunks_async

async def yielder(chunk_payload):
    print(chunk_payload["delta"])

await replay_chunks_async(chunks, yielder, speed=1.0)
```

`speed=2.0` halves the wait between chunks; `speed=0` plays as fast as the loop can run. The loop uses a monotonic deadline so long streams don't drift.

## Limitations

- One stream per recording. Interleaved chunks from concurrent streams aren't supported in v0.2; if you have that, capture each stream as its own session and join later.
- The `delta` schema is provider-shaped. A diff between an Anthropic stream and an OpenAI stream of "the same" response works structurally but not field-for-field.
