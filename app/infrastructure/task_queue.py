import uuid
import logging
import threading

from app.core.config import settings

logger = logging.getLogger(__name__)

# In-memory status store for inline (no-broker) mode.
_results: dict[str, dict] = {}
_lock = threading.Lock()

# Lazily-created Celery app (only when enabled AND celery is installed).
celery_app = None


def _init_celery():
    global celery_app
    if celery_app is not None:
        return celery_app
    if not settings.CELERY_ENABLED:
        return None
    try:
        from celery import Celery
        celery_app = Celery("documind", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
        celery_app.conf.task_track_started = True
        celery_app.conf.broker_connection_retry_on_startup = True
        logger.info(f"Celery initialized with broker {settings.REDIS_URL}")
    except Exception as e:
        logger.error(f"Celery/Redis unavailable, falling back to inline execution: {e}")
        celery_app = None
    return celery_app


def is_enabled() -> bool:
    return bool(settings.CELERY_ENABLED and _init_celery() is not None)


def submit(func, *args, **kwargs) -> dict:
    """
    Run a job. When Celery+Redis is active the job is dispatched to a worker;
    otherwise it executes inline (current default behaviour). Returns a record
    with at least {task_id, status}.
    """
    task_id = uuid.uuid4().hex

    if is_enabled():
        # Celery dispatch requires registered tasks; we expose a generic shim.
        try:
            async_result = _celery_run.delay(func.__module__, func.__name__, args, kwargs)
            return {"task_id": async_result.id, "status": "PENDING", "mode": "celery"}
        except Exception as e:
            logger.error(f"Celery dispatch failed, running inline: {e}")

    # Inline fallback
    with _lock:
        _results[task_id] = {"task_id": task_id, "status": "STARTED", "mode": "inline"}
    try:
        result = func(*args, **kwargs)
        with _lock:
            _results[task_id] = {"task_id": task_id, "status": "SUCCESS", "result": result, "mode": "inline"}
    except Exception as e:
        logger.error(f"Inline task {task_id} failed: {e}", exc_info=True)
        with _lock:
            _results[task_id] = {"task_id": task_id, "status": "FAILURE", "error": str(e), "mode": "inline"}
    return _results[task_id]


def get_status(task_id: str) -> dict:
    if is_enabled():
        try:
            from celery.result import AsyncResult
            res = AsyncResult(task_id, app=celery_app)
            payload = {"task_id": task_id, "status": res.status, "mode": "celery"}
            if res.successful():
                payload["result"] = res.result
            elif res.failed():
                payload["error"] = str(res.result)
            return payload
        except Exception as e:
            logger.error(f"Celery status lookup failed: {e}")
    with _lock:
        return _results.get(task_id, {"task_id": task_id, "status": "UNKNOWN"})


# Registered Celery task shim (only meaningful when a worker is running).
if settings.CELERY_ENABLED:
    _app = _init_celery()
    if _app is not None:
        import importlib

        @_app.task(name="documind.run")
        def _celery_run(module_name, func_name, args, kwargs):
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name)
            return fn(*args, **(kwargs or {}))
