# PLANS.md

The detailed Execution Plan (`PLANS.md`) is a **living document** and the **memory** that helps Codex steer toward a completed project. Fel mentioned his actual `plans.md` file was about **160 lines** in length, expanded to approximate the detail required for a major project, such as the 15,000-line change to the JSON parser for streaming tool calls.

## 1. Big Picture / Goal

- **Objective:** To execute a core refactor of the existing streaming JSON parser architecture to seamlessly integrate the specialized `ToolCall_V2` library, enabling advanced, concurrent tool call processing and maintaining robust performance characteristics suitable for the "AI age". This refactor must minimize latency introduced during intermediate stream buffering.
- **Architectural Goal:** Transition the core tokenization and parsing logic from synchronous, block-based handling to a fully asynchronous, state-machine-driven model, specifically targeting non-blocking tool call detection within the stream.
- **Success Criteria (Mandatory):**
    - All existing unit, property, and fuzzing tests must pass successfully post-refactor.
    - New comprehensive integration tests must be written and passed to fully validate `ToolCall_V2` library functionality and streaming integration.
    - Performance benchmarks must demonstrate no more than a 5% regression in parsing speed under high-concurrency streaming loads.
    - The `plans.md` document must be fully updated upon completion, serving as the executive summary of the work accomplished.
    - A high-quality summary and documentation updates (e.g., Readme, API guides) reflecting the new architecture must be generated and committed.

## 2. To-Do List (High Level)

- [ ] **Spike 1:** Comprehensive research and PoC for `ToolCall_V2` integration points.
- [ ] **Refactor Core:** Implement the new asynchronous state machine for streaming tokenization.
- [ ] **Feature A:** Implement the parsing hook necessary to detect `ToolCall_V2` structures mid-stream.
- [ ] **Feature B:** Develop the compatibility layer (shim) for backward support of legacy tool call formats.
- [ ] **Testing:** Write extensive property tests specifically targeting concurrency and error handling around tool calls.
- [ ] **Documentation:** Update all internal and external documentation, including `README.md` and inline comments.

## 3. Plan Details (Spikes & Features)

### Spike 1: Research `ToolCall_V2` Integration

- **Action:** Investigate the API signature of the `ToolCall_V2` library, focusing on its memory allocation strategies and compatibility with the current Rust asynchronous ecosystem (Tokio/Async-std). Determine if vendoring or a simple dependency inclusion is required.
- **Steps:**
    1. Analyze `ToolCall_V2` source code to understand its core dependencies and threading requirements.
    2. Create a minimal proof-of-concept (PoC) file to test basic instantiation and serialization/deserialization flow.
    3. Benchmark PoC for initial overhead costs compared to the previous custom parser logic.
- **Expected Outcome:** A clear architectural recommendation regarding dependency management and an understanding of necessary low-level code modifications.

### Refactor Core: Asynchronous State Machine Implementation

- **Goal:** Replace the synchronous `ChunkProcessor` with a `StreamParser` that utilizes an internal state enum (e.g., START, KEY, VALUE, TOOL_CALL_INIT, TOOL_CALL_BODY).
- **Steps:**
    1. Define the new `StreamParser` trait and associated state structures.
    2. Migrate existing buffer management to use asynchronous channels/queues where appropriate.
    3. Refactor token emission logic to be non-blocking.
    4. Ensure all existing `panic!` points are converted to recoverable `Result` types for robust streaming.

### Feature A: `ToolCall_V2` Stream Hook

- **Goal:** Inject logic into the `StreamParser` to identify the start of a tool call structure (e.g., specific JSON key sequence) and hand control to the `ToolCall_V2` handler without blocking the main parser thread.
- **Steps:**
    1. Implement the `ParseState::TOOL_CALL_INIT` state.
    2. Write the bridging code that streams raw bytes/tokens directly into the `ToolCall_V2` library's parser.
    3. Handle the return of control to the main parser stream once the tool call object is fully constructed.
    4. Verify that subsequent JSON data (after the tool call structure) is processed correctly.

### Feature B: Legacy Tool Call Compatibility Shim

- **Goal:** Create a compatibility wrapper that translates incoming legacy tool call formats into the structures expected by the new `ToolCall_V2` processor, ensuring backward compatibility.
- **Steps:**
    1. Identify all legacy parsing endpoints that still utilize the old format.
    2. Implement a `LegacyToolCallAdapter` struct to wrap the old format.
    3. Test the adapter against a suite of known legacy inputs.

### Testing Phase

- **Goal:** Achieve 100% test passing rate and add specific coverage for the new feature.
- **Steps:**
    1. Run the complete existing test suite to ensure the core refactor has not caused regressions.
    2. Implement new property tests focused on interleaved data streams: standard JSON data mixed with large, complex `ToolCall_V2` objects.
    3. Integrate and run the fuzzing tests against the new `StreamParser`.

## 4. Progress (Living Document Section)

_(This section is regularly updated by Codex, acting as its memory, showing items completed and current status)._

|Date|Time|Item Completed / Status Update|Resulting Changes (LOC/Commit)|
|:--|:--|:--|:--|
|2023-11-01|09:30|Plan initialized. Began research on Spike 1.|Initial `plans.md` committed.|
|2023-11-01|11:45|Completed Spike 1 research. Decision made to vendor/fork `ToolCall_V2`.|Research notes added to Decision Log.|
|2023-11-01|14:00|Defined `StreamParser` trait and core state enum structures.|Initial ~500 lines of refactor boilerplate.|
|2023-11-01|17:15|Migrated synchronous buffer logic to non-blocking approach. Core tests failing (expected).|~2,500 LOC modified in `core/parser_engine.rs`.|
|2023-11-02|10:30|Completed implementation of Feature A (Tool Call Stream Hook).|New `tool_call_handler.rs` module committed.|
|2023-11-02|13:45|Wrote initial suite of integration tests for Feature A. Tests now intermittently passing.|~600 LOC of new test code.|
|2023-11-02|15:50|Implemented Feature B (Legacy Shim). All existing unit tests pass again.|Code change finalized. Total PR delta now > 4,200 LOC.|
|2023-11-02|16:20|Documentation updates for `README.md` completed and committed.|Documentation finalized.|
|**Current Status:**|**[Timestamp]**|Tests are stable, clean-up phase initiated. Ready for final review and PR submission.|All checks complete.|

## 5. Surprises and Discoveries

_(Unexpected technical issues or findings that influence the overall plan)._

1. **Threading Conflict:** The `ToolCall_V2` library uses an internal thread pool which conflicts with the parent process's executor configuration, necessitating extensive use of `tokio::task::spawn_blocking` wrappers instead of direct calls.
2. **Vendoring Requirement:** Due to a subtle memory leak identified in `ToolCall_V2`'s error handling path when processing incomplete streams, the decision was made to **vendor in** (fork and patch) the library to implement a necessary hotfix.
3. **JSON Format Edge Case:** Discovery of an obscure edge case where the streaming parser incorrectly handles immediately nested tool calls, requiring an adjustment to the `TOOL_CALL_INIT` state machine logic.

## 6. Decision Log

_(Key implementation decisions made during the execution of the plan)._

| Date       | Decision                                                            | Rationale                                                                                                                                          |
| :--------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023-11-01 | Chosen Language/Framework: Rust and Tokio.                          | Maintain consistency with established project codebase.                                                                                            |
| 2023-11-01 | Dependency Strategy: Vendoring/Forking `ToolCall_V2` library.       | Provides greater control over critical memory management and allows for immediate patching of stream-related bugs.                                 |
| 2023-11-02 | Error Handling: Adopted custom `ParserError` enum for all failures. | Standardized error reporting across the new asynchronous streams, preventing unexpected panics in production.                                      |
| 2023-11-02 | Testing Priority: Exhaustive Property Tests.                        | Given the complexity of the core refactor, property tests were prioritized over simple unit tests to maximize confidence in the 15,000 LOC change. |