# Failure modes

Draft taxonomy for the tool-surface traces, grouped from the reviewer's notes on
the two 2026-09-26 runs (40 traces, 11 fails). The reviewer's own words stay in
each run's `annotations.json`. Each run's `patterns.json` holds this grouping by
reference, so the review app's Progress view shows it.

The modes are split by where the fix belongs: the assistant's behavior, the
market data, or a Maverick tool.

| Mode | Fails | Where the fix belongs | Evaluator |
| --- | --- | --- | --- |
| Acts on a guess instead of asking | q09, q14, b18, b20 | Assistant behavior, helped by tool design | Model judge (`judges/acts-on-a-guess.md`) |
| Market data missing or unreliable | q04, q07, b06, b15 | Market data and the eval fixtures | None yet: fix the causes first |
| Tool result misleading or unusable | q02, b05, q15 | Maverick tool code | None: each defect gets a unit test when fixed |

The known causes behind the second and third modes, and b18's missing
`entry_date`, are rows in `docs/exec-plans/tech-debt-tracker.md`.

## Acts on a guess instead of asking

The request leaves out something the action or answer depends on, such as which
stock, which date or year, or the user's goals. The assistant then records data
or gives a specific answer using a value it assumed, instead of asking first.
Saying what it assumed does not make it a pass.

- q09: no year was given; the assistant recorded the purchase as 2026.
- q14: no stock was named; the assistant sized the position anyway.
- b18: no date was given; the trade was logged with today's date.
  `portfolio_journal_add_trade` does not expose the `entry_date` that
  `JournalService.add_trade` already accepts, so the tool could not backdate it.
- b20: no goals or risk tolerance were given; the assistant drafted a
  portfolio with an assumed moderate risk level.

## Market data missing or unreliable

The method is sound, but the local market data behind the answer is empty,
stale, or does not cover enough history. A figure is wrong, or the question
cannot be answered.

- q04: the reviewer judged the RSI and MACD values wrong and suspected stale
  data. The traces fetch live history into a fresh database, so this needs a
  reference value to confirm.
- q07: in the `empty` data state the screener has no symbols to screen
  (`symbols_screened: 0`), so there is no bullish list to return.
- b06: the seeded screening snapshot uses fixture closes (for example NVDA at
  181.20) that disagree with live quotes. This comes from `seed.py`, not from
  Maverick.
- b15: the portfolio backtests covered 2 of the 5 holdings.

## Tool result misleading or unusable

A Maverick tool reports success for input it could not serve, rejects valid
input with a misleading error, or returns a result too large for the client to
read.

- q02: `market_data_get_quote` returns `status: success` with price 0 for
  delisted TWTR. `MarketDataService.get_quote` builds the quote without
  checking that a price exists.
- b05: `technical_get_support_resistance` rejects `BRK.B` with an
  "insufficient price history" error; Yahoo spells it `BRK-B`. A blanket dot to
  dash rewrite would break exchange suffixes such as `7203.T`.
- q15: the backtest result was about 105,000 characters, mostly the daily
  equity curve and drawdown series, so the client saved it to a file and the
  model never read it.

## Judge status

`judges/acts-on-a-guess.md` was run on 2026-09-27 by in-session Opus 5.5
subagents over the 37 reviewed traces that are not its few-shot examples
(q14, q10, b10). Against the reviewer's labels it passed 32 of 34 passing
traces (TPR 0.94) and failed 3 of 3 failing ones (TNR 1.00). The judgments
are in `judges/results/2026-09-27-acts-on-a-guess-dev.json`.

This is a first read on a development set, not a validation. Three failing
traces cannot measure TNR, and there is no held-out test set. Validating the
judge needs about 20 or more labeled failures of this mode, which means a
new batch of cases where the request leaves out something the result depends
on.

## For the reviewer

- q09: the note says March 3 was a Sunday. That holds for 2024; the assistant
  recorded 2026, when it was a Tuesday. The trace still fits this mode, because
  the year was a guess.
- b11: the verdict is pass, but the note says the assistant should have asked
  before recording a Labor Day purchase. That is the same principle as q09's
  fail. It is kept as pass here, and the judge prompt does not use it as an
  example.
- q15: the note asks for backfilled data, but the trace shows the model never
  read the result at all. It is filed under tool results for that reason.
- The judge's two disagreements are label questions, so the prompt was not
  tuned to them:
  - b11: the judge fails it for recording the Labor Day purchase before
    checking it, which is what the b11 note says. If the note is the rule,
    b11's verdict should be fail.
  - b15: labeled fail for its data, the judge also fails it for this mode:
    "best strategy for my portfolio" gave no goals or risk tolerance, and
    the answer still recommends specific trims and sectors. If that holds,
    b15 belongs to both modes.
