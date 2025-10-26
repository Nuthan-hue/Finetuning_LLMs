-- Database initialization script for Kaggle Multi-Agent System
-- This script creates tables for storing agent results, workflow history, and performance metrics

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for agent state and results
-- ========================================

-- Competitions table
CREATE TABLE IF NOT EXISTS competitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    target_percentile DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflow runs table
CREATE TABLE IF NOT EXISTS workflow_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    competition_id UUID REFERENCES competitions(id) ON DELETE CASCADE,
    run_number INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL, -- RUNNING, COMPLETED, FAILED
    target_percentile DECIMAL(5, 4),
    max_iterations INTEGER,
    current_iteration INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    final_rank INTEGER,
    final_percentile DECIMAL(5, 4),
    target_met BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    UNIQUE(competition_id, run_number)
);

-- Agent executions table
CREATE TABLE IF NOT EXISTS agent_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_run_id UUID REFERENCES workflow_runs(id) ON DELETE CASCADE,
    agent_type VARCHAR(100) NOT NULL, -- DataCollector, ModelTrainer, etc.
    iteration INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL, -- IDLE, RUNNING, COMPLETED, ERROR
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds DECIMAL(10, 2),
    context JSONB, -- Input context
    results JSONB, -- Output results
    error_message TEXT,
    metadata JSONB
);

-- Data collection records
CREATE TABLE IF NOT EXISTS data_collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_execution_id UUID REFERENCES agent_executions(id) ON DELETE CASCADE,
    competition_name VARCHAR(255) NOT NULL,
    download_path VARCHAR(500),
    files_downloaded INTEGER,
    total_size_bytes BIGINT,
    analysis_report JSONB,
    external_sources JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model training records
CREATE TABLE IF NOT EXISTS model_trainings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_execution_id UUID REFERENCES agent_executions(id) ON DELETE CASCADE,
    model_type VARCHAR(100) NOT NULL,
    model_path VARCHAR(500),
    training_config JSONB,
    best_score DECIMAL(10, 6),
    metric VARCHAR(50),
    num_features INTEGER,
    training_time_seconds DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Submissions table
CREATE TABLE IF NOT EXISTS submissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_execution_id UUID REFERENCES agent_executions(id) ON DELETE CASCADE,
    workflow_run_id UUID REFERENCES workflow_runs(id) ON DELETE CASCADE,
    competition_name VARCHAR(255) NOT NULL,
    submission_file VARCHAR(500),
    submission_message TEXT,
    kaggle_submission_id VARCHAR(100),
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Leaderboard snapshots table
CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_execution_id UUID REFERENCES agent_executions(id) ON DELETE CASCADE,
    workflow_run_id UUID REFERENCES workflow_runs(id) ON DELETE CASCADE,
    competition_name VARCHAR(255) NOT NULL,
    current_rank INTEGER,
    total_teams INTEGER,
    current_percentile DECIMAL(5, 4),
    target_percentile DECIMAL(5, 4),
    gap_to_target DECIMAL(5, 4),
    recommendation VARCHAR(100),
    top_score DECIMAL(10, 6),
    our_score DECIMAL(10, 6),
    score_difference DECIMAL(10, 6),
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    leaderboard_data JSONB
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_run_id UUID REFERENCES workflow_runs(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6),
    metric_type VARCHAR(50), -- accuracy, f1_score, rmse, etc.
    iteration INTEGER,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
-- ==========================================

CREATE INDEX idx_workflow_runs_competition ON workflow_runs(competition_id);
CREATE INDEX idx_workflow_runs_status ON workflow_runs(status);
CREATE INDEX idx_workflow_runs_started ON workflow_runs(started_at DESC);

CREATE INDEX idx_agent_executions_workflow ON agent_executions(workflow_run_id);
CREATE INDEX idx_agent_executions_type ON agent_executions(agent_type);
CREATE INDEX idx_agent_executions_status ON agent_executions(status);

CREATE INDEX idx_submissions_workflow ON submissions(workflow_run_id);
CREATE INDEX idx_submissions_competition ON submissions(competition_name);
CREATE INDEX idx_submissions_date ON submissions(submitted_at DESC);

CREATE INDEX idx_leaderboard_workflow ON leaderboard_snapshots(workflow_run_id);
CREATE INDEX idx_leaderboard_competition ON leaderboard_snapshots(competition_name);
CREATE INDEX idx_leaderboard_captured ON leaderboard_snapshots(captured_at DESC);

CREATE INDEX idx_metrics_workflow ON performance_metrics(workflow_run_id);
CREATE INDEX idx_metrics_name ON performance_metrics(metric_name);

-- Create views for easier querying
-- =================================

-- View: Latest workflow runs per competition
CREATE OR REPLACE VIEW latest_workflow_runs AS
SELECT DISTINCT ON (c.name)
    c.name as competition_name,
    wr.id as workflow_run_id,
    wr.status,
    wr.current_iteration,
    wr.max_iterations,
    wr.final_rank,
    wr.final_percentile,
    wr.target_percentile,
    wr.target_met,
    wr.started_at,
    wr.completed_at
FROM competitions c
JOIN workflow_runs wr ON c.id = wr.competition_id
ORDER BY c.name, wr.started_at DESC;

-- View: Agent performance summary
CREATE OR REPLACE VIEW agent_performance_summary AS
SELECT
    agent_type,
    status,
    COUNT(*) as execution_count,
    AVG(duration_seconds) as avg_duration_seconds,
    MIN(duration_seconds) as min_duration_seconds,
    MAX(duration_seconds) as max_duration_seconds
FROM agent_executions
WHERE completed_at IS NOT NULL
GROUP BY agent_type, status;

-- View: Competition leaderboard progress
CREATE OR REPLACE VIEW competition_progress AS
SELECT
    ls.competition_name,
    ls.workflow_run_id,
    ls.current_rank,
    ls.current_percentile,
    ls.target_percentile,
    ls.gap_to_target,
    ls.recommendation,
    ls.our_score,
    ls.top_score,
    ls.captured_at,
    ROW_NUMBER() OVER (PARTITION BY ls.workflow_run_id ORDER BY ls.captured_at) as snapshot_number
FROM leaderboard_snapshots ls
ORDER BY ls.captured_at DESC;

-- Functions for common operations
-- ================================

-- Function: Get workflow summary
CREATE OR REPLACE FUNCTION get_workflow_summary(p_workflow_run_id UUID)
RETURNS TABLE (
    agent_type VARCHAR,
    status VARCHAR,
    iterations INTEGER,
    total_duration_seconds DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ae.agent_type,
        ae.status,
        COUNT(*)::INTEGER as iterations,
        SUM(ae.duration_seconds) as total_duration_seconds
    FROM agent_executions ae
    WHERE ae.workflow_run_id = p_workflow_run_id
    GROUP BY ae.agent_type, ae.status
    ORDER BY ae.agent_type;
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate improvement over iterations
CREATE OR REPLACE FUNCTION calculate_improvement(p_workflow_run_id UUID)
RETURNS TABLE (
    iteration INTEGER,
    percentile DECIMAL,
    improvement_from_previous DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        snapshot_number::INTEGER as iteration,
        current_percentile,
        current_percentile - LAG(current_percentile) OVER (ORDER BY snapshot_number) as improvement
    FROM competition_progress
    WHERE workflow_run_id = p_workflow_run_id
    ORDER BY snapshot_number;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data (optional, for testing)
-- ==========================================

-- Example competition
INSERT INTO competitions (name, description, target_percentile)
VALUES ('titanic', 'Titanic: Machine Learning from Disaster', 0.20)
ON CONFLICT (name) DO NOTHING;

-- Grant permissions (optional, adjust based on your setup)
-- =========================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO kaggle;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO kaggle;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO kaggle;

-- Logging
SELECT 'Database initialization completed successfully' AS status;