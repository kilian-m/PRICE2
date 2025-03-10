import sqlite3 as sql


class JobScheduler:
    def __init__(self, db_path, job_list=None):
        self.db_path = db_path
        self.conn = sql.connect(self.db_path)

        if job_list is not None:
            self._create_table()
            self.fill_jobs(job_list)

    def _create_table(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  job TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending', 'in_progress', 'complete', 'failed'))
                )
                """
            )

    def fill_jobs(self, jobs):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.executemany(
                "INSERT INTO jobs VALUES (?, 'pending')", ((str(job),) for job in jobs)
            )
        self.conn.commit()

    def claim_job(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE;")
            cursor.execute("SELECT job FROM jobs WHERE status='pending'")
            job = cursor.fetchone()[0]
            if job is None:
                self.conn.commit()
                return None
            cursor.execute("UPDATE jobs SET status='in_progress' WHERE job=?;", (job,))
            self.conn.commit()
            return job
        except (sql.OperationalError, TypeError):
            self.conn.rollback()
            return None

    def mark_complete(self, job):
        with self.conn:
            self.conn.execute("UPDATE jobs SET status='complete' WHERE job=?;", (job,))

    def mark_failed(self, job):
        with self.conn:
            self.conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (job,))

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
