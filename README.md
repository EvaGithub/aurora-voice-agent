# Aurora Hotel Voice Agent

Aurora is a hotel-reservations voice agent: a caller speaks, the agent answers policy questions from a grounded knowledge base, checks live availability, books a room, switches language mid-call, and hands off to a human when asked.

The core cascade is:

```text
caller audio -> VAD and endpointing -> STT -> AgentRouter -> LLM -> RAG and tools -> TTS
```

## Attribution And Scope

The staged cascade, hotel tools, retrieval, language routing, telemetry, evaluation suites, and browser demo come from the [FDE workshop scaffold](https://github.com/hamzafarooq/multi-agent-course) by Hamza Farooq. This fork adds:

- **A room-native LiveKit agent worker** (`livekit/agent_worker.py`). The scaffold used the room for participant identity while the audio itself travelled over HTTP, which its README called out as the remaining gap. The worker joins the room as a real participant, subscribes to the caller's audio track, endpoints turns server-side, and publishes its own synthesized speech as a LiveKit track. See [Room-Native Agent Worker](#room-native-agent-worker).
- **A scriptable simulated caller** (`livekit/sim_caller.py`) that publishes speech into the room and verifies the reply that comes back, so the room path is testable without a microphone, including barge-in.
- **A publishable system-voice path** (`synthesize_wav` in `pipeline/providers.py`). The scaffold's system TTS played to the machine's speakers and returned nothing, which a worker cannot publish.
- **Unit coverage for the worker's audio logic** (`livekit/test_agent_worker.py`, 16 tests). These caught a real defect: the minimum-utterance check measured the whole buffer including pre-roll and endpoint silence, so an 80 ms blip cleared a 300 ms threshold and would bill a transcription request on every cough.
- **Recovery from Groq `tool_use_failed` 400s** (`pipeline/providers.py`), which previously dropped the tool call and hung up on the caller.

## Capabilities

- Hotel availability and mock booking tools
- Hotel-only conversational guardrails
- Local policy RAG using SQLite FTS5
- English and Spanish session routing
- Mock, OpenAI, and Groq provider modes
- Local microphone capture with WebRTC VAD
- Browser VAD with adaptive noise calibration and playback barge-in
- Per-turn structured telemetry and a browser trace timeline
- Local LiveKit room with caller and agent participants
- Room-native agent worker that subscribes to and publishes real LiveKit audio tracks
- Scriptable simulated caller for testing the room path without a microphone
- Deterministic task evaluation and red-team suites
- Zero-cost capacity calculator for DAU and concurrency planning
- SIP and IVR simulations for telephony mapping

## Project Structure

```text
aurora-voice-agent/
|-- README.md
|-- RUNBOOK.md
|-- knowledge/
|   `-- hotel_policies.md
|-- evals/
|   |-- core.json
|   |-- red_team.json
|   `-- run_evals.py
|-- pipeline/
|   |-- agent.py
|   |-- knowledge.py
|   |-- providers.py
|   |-- router.py
|   |-- scale_check.py
|   |-- telemetry.py
|   |-- test_features.py
|   `-- voice_loop.py
|-- livekit/
|   |-- start_local_server.sh
|   |-- create_room.py
|   |-- talk_server.py
|   |-- agent_worker.py
|   |-- sim_caller.py
|   |-- test_agent_worker.py
|   `-- web/
`-- mocks/
    |-- demo_call.py
    |-- ivr_menu_mock.py
    `-- sip-ivr-call-flow.md
```

## Quick Start Without An API Key

The complete agent, tool, RAG, routing, evaluation, and scale paths run without network access or paid requests.

```bash
cd pipeline
python3 smoke_test.py
python3 -m unittest -v test_features.py
PROVIDER=mock python3 voice_loop.py --text
```

Try these turns:

```text
What is the weather?
What is the cancellation policy?
I need a room from August 12 to August 14 for two guests.
Please speak Spanish.
¿Cuál es la política de mascotas?
Connect me to the front desk.
```

## OpenAI Setup

```bash
cd pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.env .env
```

Set the following values in `pipeline/.env`:

```env
PROVIDER=openai
OPENAI_API_KEY=your_key_here
TTS_BACKEND=system
TELEMETRY_JSONL=../logs/voice-events.jsonl
```

Verify the live model before adding audio:

```bash
python voice_loop.py --text
```

Run the local microphone cascade:

```bash
python voice_loop.py
```

The terminal reports capture, STT, routing, retrieval, LLM, tool, TTS, and total turn timing. `TTS_BACKEND=system` uses the macOS voice and avoids cloud TTS cost during rehearsal.

Set `TTS_BACKEND=provider` to use the selected provider's configured TTS model and voice. Provider TTS incurs audio-generation cost.

## Groq Setup

The provider adapter uses the same tool-calling interface for OpenAI and Groq.

```env
PROVIDER=groq
GROQ_API_KEY=your_key_here
TTS_BACKEND=system
```

The commands remain the same.

## Local LiveKit Demo

Install the room demo once:

```bash
cd livekit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
```

Use three terminals.

Terminal 1 starts the self-contained LiveKit development server:

```bash
cd livekit
./start_local_server.sh
```

Terminal 2 creates the room and starts the browser application:

```bash
cd livekit
source .venv/bin/activate
python create_room.py
python talk_server.py
```

Open `http://localhost:5173`, click **Start call**, allow microphone access, and speak naturally. The browser automatically joins the caller and Aurora participants, detects caller turns, displays grounding sources, and shows stage telemetry.

The LiveKit bridge honors `TTS_BACKEND` from `pipeline/.env`. With `provider`, the server synthesizes WAV audio using `TTS_MODEL` and `TTS_VOICE`, and the UI labels the response with the selected voice. With `system` or `mock`, the browser uses its installed speech voice.

The browser exposes two workshop controls:

- **Endpoint silence** changes how long a pause must be before a turn is committed.
- **Speech sensitivity** changes the adaptive speech threshold relative to the measured noise floor.

Speak while Aurora is playing a response to demonstrate playback barge-in. The browser cancels speech output, records the interruption, and opens the next caller turn.

### LiveKit Boundary

`talk_server.py` is the workshop bridge. The caller and Aurora identities are real room participants, but the audio itself never crosses the room: the browser captures, endpoints, and posts a completed recording to `/voice-agent` over HTTP, then plays the answer locally. Remove LiveKit from that path and the demo behaves the same, because the room carries identity rather than media.

`agent_worker.py` closes that gap. See **Room-Native Agent Worker** below.

Persistent session storage and SIP dispatch remain out of scope.

## Room-Native Agent Worker

`agent_worker.py` joins the room as a real participant instead of answering over HTTP. It subscribes to the caller's published audio track, endpoints turns server-side, and publishes its own synthesized speech as a LiveKit track:

```text
caller mic -> LiveKit room -> worker subscribes -> VAD -> STT -> Agent
          -> TTS -> worker publishes track -> LiveKit room -> caller
```

The room becomes the transport rather than an identity label, which is the property that makes SIP viable: a phone caller joins as an ordinary participant publishing an audio track, so the worker needs no separate telephony path.

Start the server, then the worker:

```bash
cd livekit
livekit-server --dev --bind 127.0.0.1     # or ./start_local_server.sh for Docker
python agent_worker.py                    # PROVIDER comes from pipeline/.env
```

Drive it without a microphone using the simulated caller, which publishes system-voice speech into the room and verifies the reply that comes back:

```bash
python sim_caller.py --say "What is the cancellation policy?" --listen 12
python sim_caller.py --say "I need a room for two guests." \
    --interrupt "Actually, connect me to a person." --interrupt-after 2 --listen 10
```

Check connectivity alone before debugging the cascade:

```bash
python agent_worker.py --probe --seconds 15
```

### What Changes Against The Browser Bridge

| Concern | `talk_server.py` bridge | `agent_worker.py` |
|---------|------------------------|-------------------|
| Audio transport | HTTP POST of a recorded blob | Subscribed and published LiveKit tracks |
| Endpointing | Browser JavaScript VAD | Server-side `webrtcvad` on 20 ms frames |
| STT input | Compressed webm, codec inferred from a filename | Raw PCM through `provider.transcribe` |
| Barge-in | Cancel local speech synthesis | `AudioSource.clear_queue()` on the published track |
| Echo handling | Transcript blocklist heuristic | Not needed; the worker never subscribes to itself |
| Reach | One browser tab | Any participant, including a future SIP caller |

Requesting `AudioStream(sample_rate=16000, num_channels=1, frame_size_ms=20)` makes the SDK deliver exactly one `webrtcvad` window per frame, so no resampling or re-buffering layer is needed. Because the worker owns the published track, it also drops the transcript-matching echo heuristic the browser path needed to guess whether it had heard its own playback.

### System Voice As Publishable Audio

`Provider.synthesize` returns `None` under `TTS_BACKEND=system` because the system command plays to the machine's speakers, which a worker cannot publish. `synthesize_wav` renders the same free local voice to a file instead, so the room-native path stays zero-cost during rehearsal. `synthesize` is unchanged, so `voice_loop.py` and `talk_server.py` keep their existing behavior.

## Grounding And Tools

Aurora uses different boundaries for different kinds of truth:

| Information | Mechanism | Reason |
|-------------|-----------|--------|
| Policies, parking, pets, breakfast, accessibility | Local RAG | Read-oriented knowledge with source evidence |
| Availability and room rates | Tool call | Dynamic operational truth |
| Booking creation | Tool call | Auditable state mutation |
| Language switching | `set_language` control tool | Validated session state and matching TTS locale |
| Transfer and hangup | Control action | Runtime and telephony behavior |

The local retriever indexes Markdown sections with SQLite FTS5. It includes English and Spanish query expansion while keeping the source document unchanged.

Aurora uses hybrid tool routing. High-confidence policy and amenity phrases select `search_hotel_knowledge` in application code before the first model call. Other tool decisions remain automatic. This keeps retrieval reliable after interruptions or off-topic turns without routing a request such as `cancel my reservation` into policy search.

## Telemetry

Each turn carries a session ID, turn ID, trace ID, provider, model, language, stage timings, tool arguments, tool results, sources, action, and ordered runtime events.

Raw transcript and response content are omitted by default, and sensitive tool fields such as guest name and contact details are redacted. Set `TELEMETRY_INCLUDE_CONTENT=true` only for controlled local debugging with non-sensitive data.

The LiveKit server writes JSONL traces to:

```text
logs/voice-events.jsonl
```

The path is ignored by Git. Set `TELEMETRY_JSONL` to change or disable the destination.

Important production measures include endpoint delay, STT latency, LLM time to first token, tool latency, TTS time to first audio, end-of-turn to first audio, interruption latency, task completion, critical entity accuracy, transfer rate, and cost per successful outcome.

## Evaluation And Red Teaming

Run all deterministic scenarios:

```bash
cd evals
python3 run_evals.py --suite all
```

Run one suite with conversation details:

```bash
python3 run_evals.py --suite core --verbose
python3 run_evals.py --suite red-team --verbose
```

The suites verify expected tools, actions, languages, sources, allowed text, and forbidden text. The red-team set covers prompt injection, policy fabrication, privacy, structured tool input, and guardrails after a language switch.

The room-native worker's audio logic has its own unit suite, which needs no server or network:

```bash
cd livekit
python -m unittest -v test_agent_worker.py
```

It covers endpointing behavior (onset debounce, pre-roll retention, mid-sentence pauses, noise rejection) and the WAV to PCM conversion that feeds the published track. The noise-rejection case caught a real defect: measuring the minimum utterance against the whole buffer counted pre-roll and endpoint silence, so an 80 ms blip cleared a 300 ms threshold and billed a transcription request. The endpointer now measures speech frames alone.

## Scale Check

The calculator converts product assumptions into peak concurrency and service demand without calling a provider:

```bash
cd pipeline
python3 scale_check.py --dau 1000000
```

Default assumptions are 0.25 calls per DAU, four minutes per call, three turns per minute, an 8x peak factor, 40 sessions per worker, and 30 percent headroom. Change every assumption before using the result as a capacity plan.

Example with a blended variable cost:

```bash
python3 scale_check.py --dau 1000000 --cost-per-minute 0.035
```

## Telephony Mapping

```text
PSTN caller -> carrier -> SIP trunk -> SBC or SIP edge -> LiveKit room -> agent -> tools
```

Run the local signaling demonstrations:

```bash
cd mocks
python3 demo_call.py
python3 demo_call.py --transfer
python3 ivr_menu_mock.py
```

The mock maps booking completion to SIP BYE and human escalation to SIP REFER. A real phone deployment also requires a carrier or telephony provider, an internet-reachable SIP edge, codec and media negotiation, security policy, dispatch rules, and a room-native agent worker.

## Safety And Cost

- Keep `.env`, virtual environments, telemetry logs, and private workshop materials out of Git.
- Do not enable raw telemetry content for real customer conversations without an approved privacy and retention policy.
- Use mock mode for rehearsal, evaluation, and scale exercises.
- Use system TTS while developing to avoid cloud TTS charges.
- Treat booking tools as mock systems until authentication, validation, idempotency, persistence, and audit controls are added.
