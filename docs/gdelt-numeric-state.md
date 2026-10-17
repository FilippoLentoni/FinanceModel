# Numeric organization news state

`gdelt_organization_tone_v1` uses released GDELT GKG v1 machine metadata,
version `gdelt-gkg-v1-organization-tone-v1`. It is distinct from paid headline
lexicon metadata. Each asset contributes five frozen inputs, in this order:

1. Whole-document organization tone divided by 100; absent tone remains zero
   with its separate missingness mask.
2. `log1p(article_count) / log1p(10000)`, capped at one.
3. `log1p(cluster_count) / log1p(10000)`, capped at one.
4. Hours since source availability divided by 48.
5. Missing-tone mask. An observed zero mention count remains an observed zero;
   a missing, corrupt, unavailable or stale archive is a readiness failure.

The source retains exact organization aliases and disclosed coverage limits.
Tone describes the entire document, not a calibrated company return signal.
VOO uses Vanguard organization mentions, not constituent-weighted ETF sentiment.
The daily archive describes the preceding UTC day. It does not include breaking
news from later that same morning; an intraday source requires a separate
feature version and a separately trained actor.

Source documents retain archive checksums, provider Last-Modified, retrieval,
local availability, actual central publication, rights evidence and attribution.
`historical_source_proxy` reconstructs availability from the conservative maximum
of provider Last-Modified and the following day at noon UTC. It is explicitly
distinct from actual system first-seen history. Live inference uses `platform`
mode and additionally requires central publication and retrieval by the decision
cutoff. Source age cannot exceed 48 hours in either mode.

Source identity uses the producer's Python JSON encoding. Model artifacts use
RFC8785, which can rewrite `1.0` to `1`. The internal frozen corpus therefore
retains the exact producer canonical JSON string alongside each source checksum
and semantic document, verifying both before accepting an artifact round trip.
It never mints a replacement source identity merely because encoding changed.

The released `recursive_ppo_gdelt_morning` profile retains the 2025 training,
2026 H1 validation and 2026 development windows, five seeds and 30,000 requested
PPO steps. The producer checks its budget first, reads at most 31 source days
per request, requires complete requested coverage, and freezes the corpus as an
immutable `run_artifact`. Only its opaque reference/checksum enters DynamoDB.
The training container verifies the artifact and every source document before
creating a learner. Previously observed 2026 periods are explicitly development
even if a different dataset identity has zero recorded accesses; promotion
cannot pass on that evidence.

Training decisions use preceding completed prices at the next actual exchange
session's 09:00 America/New_York cutoff. Calendar sessions and DST determine
that next session; weekdays are never guessed. Execution remains its 09:30 open.
The last price row is an unused terminal close mark when no following session
is supplied. The common evaluator and learner receive the same cutoffs, state
extractor, allocation transform, fees and constraints.

Serving requires a complete five-member exported ensemble with the exact GDELT
feature version and training corpus checksum. It reads the current central
numeric source, records price `as_of` separately from `decision_at`, and retains
the legacy `decision_timing` constant. Optional `news_timing` identifies current
eligible news with completed prices. Frozen paper decisions store the actual
source bytes, state, member outputs and constraints for replay. Explicit
historical serving without a morning cutoff is refused; existing frozen
decision replay remains available.

The isolated eight-asset classical profile also consumes the current numeric
source. Its `news_required: false` disables the mandatory paid-headline path;
the released numeric adapter remains required. This adapter is a declared
research sensitivity assumption: at most five basis points of mean shift at
unit tone and at most 25% volatility scaling at unit log mention intensity.
Centered scenarios receive the same mean and scale changes as the covariance
and expected-return inputs. Coefficients are **not** estimated from future
returns or presented as a learned/calibrated news forecast. Source coverage
intensity can reflect company popularity. Every solver records original and
transformed checksums, exact features and assumptions; improved performance is
not claimed. Instrument objective Shapley is bounded to nine assets (512 cheap
objective evaluations), while four-group plan comparison keeps sixteen
optimization hybrids.

Manual user-authorized full fits can name the released daily-history or numeric
GDELT objective with the existing `model_selection` job type and its 3000-second
maximum. The producer still freezes the profile; callers cannot inject a custom
protocol. Its existing price, category/project budget, approval and lease gates
remain mandatory. The weekly research-controller role is still forbidden from
using this longer job, and its 900-second recursive limit remains unchanged.

The [GDELT terms](https://gdeltproject.org/about.html#termsofuse) allow use of
released datasets and require attribution. This adapter uses only released
machine metadata and includes that attribution; it does not acquire article text.
