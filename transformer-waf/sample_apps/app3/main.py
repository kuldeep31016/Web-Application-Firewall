from __future__ import annotations

from fastapi import FastAPI

app = FastAPI()


@app.get("/api/users/{user_id}/profile")
async def profile(user_id: int):
    return {"user_id": user_id, "profile": "ok"}


