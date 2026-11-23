"""
Celery application configuration.

Broker: RabbitMQ (AMQP)
Result backend: Redis
"""

from celery import Celery

from educore.config import settings

celery_app = Celery(
    "educore",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Reliability
    task_acks_late=True,
    worker_prefetch_multiplier=1,

    # Task discovery
    imports=["educore.tasks.ingestion_tasks"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Result expiry (24 hours)
    result_expires=86400,

    # Task routes
    task_routes={
        "educore.process_document": {"queue": "educore.process"},
    },
)
