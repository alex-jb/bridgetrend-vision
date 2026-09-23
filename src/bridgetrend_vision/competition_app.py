"""FastAPI entry point for the BridgeTrend OpenCV judge console."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .competition_runtime import CompetitionRuntime

WEB_ROOT = Path(__file__).resolve().parent / "web"


class HumanReviewRequest(BaseModel):
    decision: str
    reviewer: str = Field(default="judge", max_length=80)
    note: str = Field(default="", max_length=1000)


def create_app(runtime: CompetitionRuntime | None = None) -> FastAPI:
    runtime = runtime or CompetitionRuntime.from_environment()
    application = FastAPI(
        title="BridgeTrend Visual Evidence Agent",
        version="1.0.0",
        description=(
            "OpenCV 5 perception-decision-action demo with auditable traces, "
            "bounded evidence acquisition, and human review."
        ),
    )
    application.state.runtime = runtime
    application.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @application.get("/healthz")
    def health() -> dict[str, object]:
        import cv2

        major = int(cv2.__version__.split(".", 1)[0])
        return {
            "status": "ok" if major >= 5 else "degraded",
            "opencv_version": cv2.__version__,
            "opencv_major_ok": major >= 5,
            "trace_backend": runtime.trace_store.backend_name,
            "metrics_backend": runtime.metrics_sink.backend_name,
            "fixture_pack": runtime.pack["pack_version"],
        }

    @application.get("/api/v1/cases")
    def cases() -> dict[str, object]:
        return {
            "cases": [_decorate_case(item) for item in runtime.list_cases()],
            "fixture_boundary": {
                "license": runtime.pack["license"],
                "provenance": runtime.pack["provenance"],
                "claim_eligible": False,
            },
        }

    @application.post("/api/v1/cases/{case_id}/run")
    def run_case(
        case_id: Annotated[str, ApiPath(min_length=1, max_length=80)],
    ) -> dict[str, object]:
        try:
            return runtime.run_case(case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/v1/run-all")
    def run_all() -> dict[str, object]:
        return runtime.run_all()

    @application.get("/api/v1/metrics")
    def metrics() -> dict[str, object]:
        return runtime.metrics()

    @application.get("/api/v1/failure-gallery")
    def failure_gallery() -> dict[str, object]:
        return {
            "cases": [
                _decorate_case(item) for item in runtime.failure_gallery()
            ]
        }

    @application.get("/api/v1/sessions/{session_id}")
    def session(
        session_id: Annotated[str, ApiPath(min_length=1, max_length=160)],
    ) -> dict[str, object]:
        record = runtime.get_session(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="session not found")
        return record

    @application.post("/api/v1/sessions/{session_id}/review")
    def review(
        session_id: Annotated[str, ApiPath(min_length=1, max_length=160)],
        request: HumanReviewRequest,
    ) -> dict[str, object]:
        try:
            return runtime.review_session(
                session_id=session_id,
                decision=request.decision,
                reviewer=request.reviewer,
                note=request.note,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/fixtures/{case_id}/{role}", include_in_schema=False)
    def fixture(
        case_id: Annotated[str, ApiPath(min_length=1, max_length=80)],
        role: Annotated[str, ApiPath(min_length=1, max_length=80)],
    ) -> FileResponse:
        try:
            return FileResponse(runtime.resolve_asset(case_id, role))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return application


def _decorate_case(case: dict[str, object]) -> dict[str, object]:
    case_id = str(case["case_id"])
    return {
        **case,
        "query_image_url": f"/fixtures/{case_id}/query",
        "candidate_image_urls": [
            f"/fixtures/{case_id}/candidate-{index}"
            for index, _ in enumerate(case["candidate_images"])
        ],
    }


app = create_app()
