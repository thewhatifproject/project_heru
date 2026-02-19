from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.schemas import HealthResponse, WsMessage
from app.services.session import SessionRegistry

load_dotenv()


registry = SessionRegistry()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="cam2inference backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(active_sessions=await registry.active_count())


@app.get("/api/session/{session_id}/config")
async def get_session_config(session_id: str) -> dict:
    session = await registry.get_or_create(session_id)
    return {
        "revision": session.revision,
        "config": session.config.model_dump(),
        "runtime_status": session.runtime_status,
    }


@app.put("/api/session/{session_id}/config")
async def put_session_config(session_id: str, patch: dict) -> dict:
    session = await registry.get_or_create(session_id)
    try:
        config = await session.update_config(patch)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error
    return {
        "revision": session.revision,
        "config": config.model_dump(),
        "runtime_status": session.runtime_status,
    }


@app.websocket("/ws/session/{session_id}")
async def realtime_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = await registry.get_or_create(session_id)

    await websocket.send_json(
        {
            "type": "config.current",
            "payload": {
                "revision": session.revision,
                "config": session.config.model_dump(),
                "metrics": session.metrics.__dict__,
                "runtime_status": session.runtime_status,
            },
        }
    )

    try:
        while True:
            incoming = WsMessage.model_validate_json(await websocket.receive_text())

            if incoming.type == "ping":
                await websocket.send_json({"type": "pong", "payload": incoming.payload})
                continue

            if incoming.type == "config.update":
                try:
                    config = await session.update_config(incoming.payload)
                except ValidationError as error:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "payload": {
                                "message": "Invalid config update",
                                "details": error.errors(),
                            },
                        }
                    )
                    continue
                await websocket.send_json(
                    {
                        "type": "config.current",
                        "payload": {
                            "revision": session.revision,
                            "config": config.model_dump(),
                            "metrics": session.metrics.__dict__,
                            "runtime_status": session.runtime_status,
                        },
                    }
                )
                continue

            if incoming.type == "preset.apply":
                preset_name = str(incoming.payload.get("name", "")).strip()
                if not preset_name:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "payload": {"message": "Missing preset name"},
                        }
                    )
                    continue

                try:
                    config = await session.apply_preset(preset_name)
                except ValueError as error:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "payload": {"message": str(error)},
                        }
                    )
                    continue

                await websocket.send_json(
                    {
                        "type": "config.current",
                        "payload": {
                            "revision": session.revision,
                            "config": config.model_dump(),
                            "metrics": session.metrics.__dict__,
                            "runtime_status": session.runtime_status,
                        },
                    }
                )
                continue

            if incoming.type == "frame":
                image_b64 = incoming.payload.get("image_b64")
                timestamp_ms = int(incoming.payload.get("timestamp_ms", 0))
                if not image_b64:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "payload": {"message": "frame payload missing image_b64"},
                        }
                    )
                    continue

                processed = await session.process_frame(image_b64=image_b64, timestamp_ms=timestamp_ms)
                await websocket.send_json({"type": "frame.processed", "payload": processed})
                await websocket.send_json(
                    {
                        "type": "stats",
                        "payload": {
                            "metrics": session.metrics.__dict__,
                            "revision": session.revision,
                        },
                    }
                )
                continue

            await websocket.send_json(
                {
                    "type": "error",
                    "payload": {"message": f"Unknown message type: {incoming.type}"},
                }
            )

    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
