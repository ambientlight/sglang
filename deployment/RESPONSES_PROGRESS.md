# V4.1 Responses progress and reasoning

The V4.1 Responses adapter now supplies assistant-message phases for Codex
Remote. Previously a preamble completed without a phase before its tool call,
so a client could close the working indicator while the turn continued.

V4.1 does not generate a native phase token. Streaming messages therefore start
as `commentary` and deliver text immediately. A message followed by a tool call,
more reasoning, truncation, or failure stays commentary. When generation stops
normally without tool calls, the completed message becomes `final_answer`.
Clients must apply the completed item's phase to the same item ID. The final
response snapshot and stored response contain that resolved phase. Non-streaming
responses use the same tool/stop classification. Other model families keep their
existing behavior.

Plaintext reasoning remains a separate reasoning item; it is never appended to
the answer. With `reasoning.summary` requested, the adapter mirrors the trace
into summary events for clients' expandable work details and retains the full
plaintext in `content`. Without that request, raw reasoning delta events are
emitted instead. Both modes retain the trace in the completed reasoning item.

Codex's DeepSeek profile already uses `model_reasoning_summary = "auto"`, and
app-server exposes reasoning summaries and raw `content` separately. Setting
`model_reasoning_summary = "none"` requests the raw delta path instead. The
`show_raw_agent_reasoning` setting controls legacy/TUI output; it is not a new
ChatGPT Remote UI toggle. Expansion controls are owned by the connected client.

Qualification uses real V4.1 parsers with CPU generator fixtures: preambles,
final answers, one-byte chunk boundaries, tool suffixes, incomplete/failed
streams, immediate text delivery, stored-response consistency, and opt-in
reasoning details. The full pinned-image API regression run passed 1,177 tests
and 430 subtests.

The deployment replaces only `serving_responses.py` using a frozen read-only
bind on the current image. The existing rollout watcher requires 60 continuous
idle seconds, verifies the image, configuration and source hashes, and checks
health, cache allocation, and capture services after recreation. Its private
state directory retains `rollback.compose.json` and the original source. No
Codex binary or conversation-history migration is required.
