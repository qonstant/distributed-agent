-- Intentionally non-destructive.
-- Rolling this migration back by dropping columns/tables would lose data.
-- If you must revert application behavior, deploy a forward-fix migration instead.
SELECT 1;
