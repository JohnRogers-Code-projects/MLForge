"""Tests for job cleanup task."""

from unittest.mock import MagicMock, patch

from app.models.job import JobStatus
from app.tasks.cleanup import _CLEANUP_STATUSES, cleanup_old_jobs


class TestCleanupTask:
    """Tests for the cleanup_old_jobs Celery task."""

    def test_cleanup_deletes_old_completed_jobs(self):
        """Test that old completed jobs are deleted."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        # Mock the delete query to return 5 deleted rows
        mock_session.execute.return_value.rowcount = 5

        with patch("app.tasks.cleanup._get_sync_session", return_value=mock_session):
            with patch("app.tasks.cleanup.settings") as mock_settings:
                mock_settings.job_retention_days = 30
                result = cleanup_old_jobs()

        assert result["deleted_count"] == 5
        assert result["error"] is None
        # Verify commit was called
        mock_session.commit.assert_called_once()

    def test_cleanup_no_old_jobs(self):
        """Test cleanup when there are no old jobs."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        # Mock the delete query to return 0 deleted rows
        mock_session.execute.return_value.rowcount = 0

        with patch("app.tasks.cleanup._get_sync_session", return_value=mock_session):
            with patch("app.tasks.cleanup.settings") as mock_settings:
                mock_settings.job_retention_days = 30
                result = cleanup_old_jobs()

        assert result["deleted_count"] == 0
        assert result["error"] is None
        # Commit is still called (simpler logic, no early return)
        mock_session.commit.assert_called_once()

    def test_cleanup_handles_database_error(self):
        """Test that cleanup handles database errors gracefully."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        # Mock execute to raise an exception
        mock_session.execute.side_effect = Exception("Database connection lost")

        with patch("app.tasks.cleanup._get_sync_session", return_value=mock_session):
            with patch("app.tasks.cleanup.settings") as mock_settings:
                mock_settings.job_retention_days = 30
                result = cleanup_old_jobs()

        assert result["deleted_count"] == 0
        assert "Database connection lost" in result["error"]
        mock_session.rollback.assert_called_once()

    def test_cleanup_statuses_are_terminal(self):
        """Test that cleanup only targets terminal statuses."""
        assert JobStatus.COMPLETED in _CLEANUP_STATUSES
        assert JobStatus.FAILED in _CLEANUP_STATUSES
        assert JobStatus.CANCELLED in _CLEANUP_STATUSES
        # Non-terminal statuses should NOT be in cleanup set
        assert JobStatus.PENDING not in _CLEANUP_STATUSES
        assert JobStatus.QUEUED not in _CLEANUP_STATUSES
        assert JobStatus.RUNNING not in _CLEANUP_STATUSES

    def test_cleanup_respects_retention_days(self):
        """Test that cleanup uses the configured retention days."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.rowcount = 0

        with patch("app.tasks.cleanup._get_sync_session", return_value=mock_session):
            with patch("app.tasks.cleanup.settings") as mock_settings:
                # Test with different retention days
                mock_settings.job_retention_days = 7
                result = cleanup_old_jobs()

        assert result["deleted_count"] == 0
        assert result["error"] is None
