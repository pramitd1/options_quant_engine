# India Macro Schedule Notes

This file documents the assumptions behind [india_macro_schedule.json](india_macro_schedule.json).

## Scope

The Stage 1 scheduled-event filter is intentionally conservative and only uses high-impact India macro releases that can materially affect:

- index option implied volatility
- short-dated premium buying decisions
- trade blocking / watchlist downgrades near event windows

The current file includes:

- India CPI
- India IIP
- India GDP

## Sources

The dates in the schedule were built from official MoSPI advance release calendars:

- `ARC_2025-26.pdf`
- `ARC_2026-27.pdf`

RBI policy dates were not added because no later official RBI MPC schedule was available beyond the last confirmed February 2026 meeting at the time this file was prepared. The engine should not rely on guessed policy dates.

## Time Assumptions

MoSPI release calendars provide official release dates, but they do not consistently provide machine-friendly intraday timestamps suitable for the engine.

For Stage 1, macro releases in [india_macro_schedule.json](india_macro_schedule.json) use a conservative normalized timestamp:

- `16:00 Asia/Kolkata`

This convention is used to keep the event-risk filter deterministic and interpretable until a more precise timestamp source is added.

## Holiday / Weekend Handling

Some official calendar dates fall on weekends or holidays. The MoSPI calendar note indicates that such releases shift to the next working day.

Where that applied, the schedule file moves the event to the next working day and documents the adjustment in the event `notes` field.

Examples:

- `India IIP` listed for `2026-03-28` was moved to `2026-03-30`
- `India CPI` listed for `2026-04-12` was moved to `2026-04-13`
- `India IIP` listed for `2026-06-28` was moved to `2026-06-29`

## Why Historical RBI Context Was Not Added

Adding the last confirmed RBI event as a historical record inside the live schedule is not the best operational choice for this engine.

Reasons:

- Stage 1 uses the schedule as a forward-looking risk-control input, not as an event history database
- stale past events add noise to the config and can confuse operators reviewing the next-event state
- keeping the schedule forward-looking is cleaner for replay/live consistency

If historical event study is needed later, it should live in a separate research dataset rather than the live production schedule.

## Future Upgrade Path

When the macro layer expands, the better production path is:

1. keep `india_macro_schedule.json` as the forward event calendar
2. add a separate historical event dataset for replay/research
3. replace assumed timestamps with provider-specific exact release times where available
