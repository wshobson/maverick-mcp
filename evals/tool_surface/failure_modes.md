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
`entry_date`, were fixed on 2026-09-27, apart from q04 (unconfirmed) and
q07. The quote now errors without a price (q02). Dotted class shares retry
with a dash (b05). Tool responses cut equity and drawdown series to 60
points (q15). The journal tool takes an `entry_date` (b18). Portfolio and
comparison backtests name what failed (b15). The seed registers a screening
universe instead of fixture prices (b06). For q07, `screening_run_screens`
now says how to add symbols; a built-in default universe is still an open
product decision in `docs/exec-plans/tech-debt-tracker.md`. The traces above
record what the server did before these fixes.

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

The same 37 traces were also put to Jev (`jev-1.13.0`, through `jev.py`) as
one yes/no question each, with the judge's definitions and examples as its
criteria. At a 0.5 cutoff it passed 24 of 34 passing traces (TPR 0.71) and
failed 2 of 3 failing ones (TNR 0.67), missing b18 at 0.27. It cost $0.0027
for about 63,000 input tokens. Tuning the cutoff on three failures would only
fit noise, so the subagent judge stays the evaluator for this mode. The run is
in `judges/results/2026-09-27-acts-on-a-guess-jev-dev.json`.

## Re-run on 2026-09-27

After the fixes, the seven affected cases were re-run through the subagent, in
`runs/20260927T152805Z-claude-opus-5-5/`:

- q02: the quote now returns an error ("No quote data for TWTR ...").
- q07: `screening_run_screens` now says how to add symbols. The assistant
  declined to pick its own list, so the request still has no answer while the
  default universe stays an open decision.
- q15: the model read the backtest result (largest tool result about 9,200
  characters).
- b05: `BRK.B` works as typed.
- b06: the screens ran on live data for the nine seeded symbols, with no
  fixture prices.
- b15: the assistant asked what "strategy" meant and about the horizon.
- b18: the journal now takes `entry_date`, but the assistant still logged
  today's date without asking. So this mode is the assistant's behavior, not
  only a missing tool parameter.

The re-runs and batch 3 also surfaced new defects, now rows in
`docs/exec-plans/tech-debt-tracker.md`:

- finviz movers (fixed in #300)
- synthetic support and resistance levels
- `risk_adjusted_analysis` sizing
- annualized metrics on a 365-day year
- profit factor 0 with no losing trades
- seed cost bases far below real prices
- a price-bar insert race under parallel calls
- no tool that lists watchlists
- no dividend fields

A suspected correlation bug was checked: an independent yfinance calculation
reproduced the low values, so they are real for the last year. The separate
off-diagonal selection issue in `correlation_analysis` stays open in the
tracker.

## Batch 3

`cases_batch3.json` (c01 to c20) aims at the "acts on a guess" mode. Sixteen
cases leave out or get wrong a detail the result depends on (the `gap` field
names which), and four are controls (`gap: none`). c03 names its trade by tag
but gives no sale date, and c15 names no watchlist while no tool lists them,
so both are gap cases. The traces are in
`runs/20260927T153817Z-claude-opus-5-5/` and wait for the reviewer's labels.
The judge is scored on them only after labeling, so its verdicts cannot bias
the review.

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
