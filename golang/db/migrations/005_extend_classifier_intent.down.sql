-- PostgreSQL cannot safely remove enum values without rebuilding the enum type,
-- which would risk existing classification rows that already use the new intents.
-- Keep this rollback as a no-op to avoid data loss.
SELECT 1;
