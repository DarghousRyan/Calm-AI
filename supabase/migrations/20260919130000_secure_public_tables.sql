-- Calm AI database security hardening.
--
-- Calm AI currently reads and writes these tables through the FastAPI backend
-- using DATABASE_URL. The browser/client does not query these tables directly.
-- Keep the tables unavailable through the Supabase Data API until explicit,
-- user-scoped policies are added for a future native client.

begin;

alter table if exists public.users enable row level security;
alter table if exists public.chat_messages enable row level security;
alter table if exists public.checkin_logs enable row level security;
alter table if exists public.daily_logs enable row level security;
alter table if exists public.predictions enable row level security;

-- RLS alone prevents access when no policy exists. Revoking the client-role
-- grants as well makes the intended deny-by-default posture explicit.
revoke all on table public.users from anon, authenticated;
revoke all on table public.chat_messages from anon, authenticated;
revoke all on table public.checkin_logs from anon, authenticated;
revoke all on table public.daily_logs from anon, authenticated;
revoke all on table public.predictions from anon, authenticated;

commit;
