"""
observability.py

Observability & Metrics module featuring Prometheus instrumentation and tracing hooks.
Exposes /metrics endpoint for Prometheus scrapers and Grafana dashboards.
"""

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from logger import get_logger

logger = get_logger(__name__)

# Prometheus Metrics Counters & Histograms
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "status"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in seconds",
    ["method", "endpoint"]
)

TOOL_INVOCATIONS_TOTAL = Counter(
    "tool_invocations_total",
    "Total Tool Call Invocations",
    ["tool_name", "status"]
)

LLM_CALL_LATENCY_SECONDS = Histogram(
    "llm_call_latency_seconds",
    "LLM Generation Latency in seconds",
    ["model_name"]
)


def track_http_metrics(method: str, endpoint: str, status_code: int, duration_sec: float):
    """Record HTTP request metrics for Prometheus."""
    try:
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint).observe(duration_sec)
    except Exception as e:
        logger.warning(f"Error recording HTTP metric: {e}")


def track_tool_metric(tool_name: str, status: str = "success"):
    """Record tool invocation metrics."""
    try:
        TOOL_INVOCATIONS_TOTAL.labels(tool_name=tool_name, status=status).inc()
    except Exception as e:
        logger.warning(f"Error recording tool metric: {e}")


async def metrics_endpoint():
    """FastAPI route handler exporting Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
