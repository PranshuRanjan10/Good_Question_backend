-- ONE-TIME, run BEFORE schema.sql. Do not re-run once live data exists.
--
-- Removes the 5 tables from the first Supabase draft. They were keyed on (device_id, local_id),
-- but Render wipes the local SQLite file on every restart, so local ids start again at 1 and new
-- rows would overwrite old ones in the cloud. The new tables are keyed on a uid that never repeats.
-- All 5 were checked empty (0 rows) on 2026-09-24 before this was written.
drop table if exists public.alerts;
drop table if exists public.incidents;
drop table if exists public.readiness_snapshots;
drop table if exists public.task_records;
drop table if exists public.training_completions;
