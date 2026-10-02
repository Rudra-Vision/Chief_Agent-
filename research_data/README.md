# research_data

Human-readable mirrors of the research artefacts. The database is authoritative;
these files exist so the research history can be inspected, diffed and backed up
as plain text.

```
journal/       one JSON file per completed trade (entry context, MAE/MFE, outcome)
experiments/   one JSON file per experiment (hypothesis, change, all validation stages)
strategies/    the immutable published configuration of every strategy version
candidates/    challengers awaiting validation
promoted/      versions that became champion, with their promotion evidence
rejected/      versions that failed the promotion gate, with the reason
```

Files are written by the research engine when it runs. Nothing here is required
for the system to operate, and nothing here is ever used as an input to a trading
decision - it is a durable, human-readable record.
