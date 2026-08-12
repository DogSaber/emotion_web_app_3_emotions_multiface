-- Read-only checks to run before applying any Emotion Recognition database
-- migration. Every query should return zero rows (or a zero count) unless
-- noted otherwise.

USE emosense;

-- Inputs whose user no longer exists.
SELECT i.input_id, i.user_id
FROM input AS i
LEFT JOIN user AS u ON u.user_id = i.user_id
WHERE u.user_id IS NULL;

-- Outputs whose input no longer exists.
SELECT o.output_id, o.input_id
FROM output AS o
LEFT JOIN input AS i ON i.input_id = o.input_id
WHERE i.input_id IS NULL;

-- Support messages whose user no longer exists.
SELECT sc.chat_id, sc.user_id
FROM support_chat AS sc
LEFT JOIN user AS u ON u.user_id = sc.user_id
WHERE u.user_id IS NULL;

-- Invalid model outputs.
SELECT output_id, number, name, confidence
FROM output
WHERE number NOT BETWEEN 0 AND 4
   OR name NOT IN ('Happy', 'Angry', 'Sad', 'Neutral', 'Surprise')
   OR confidence IS NULL
   OR confidence < 0
   OR confidence > 1;

-- Diagnostic only: this is expected to show the historical hard-coded
-- input_id=1 concentration and cannot be repaired without trustworthy
-- source attribution.
SELECT input_id, COUNT(*) AS output_count
FROM output
GROUP BY input_id
ORDER BY output_count DESC, input_id ASC;
