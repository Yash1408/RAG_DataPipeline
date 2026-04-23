"""Celery app — Redis broker/backend. Scales horizontally via replica count."""
import os
from celery import Celery

celery_app = Celery(
    "docai",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1"),
)
celery_app.conf.update(
    task_acks_late=True,               # re-queue if worker dies mid-task
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,      # fair dispatch for long PDF jobs
    task_time_limit=15 * 60,
    task_soft_time_limit=13 * 60,
    result_expires=24 * 3600,
    task_default_queue="extract",
    task_routes={"app.workers.extractor.extract_document_task": {"queue": "extract"}},
)
