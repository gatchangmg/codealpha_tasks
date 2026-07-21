from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

app = FastAPI(title="AuraTranslate Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    sourceLang: str = Field(..., min_length=2, max_length=10)
    targetLang: str = Field(..., min_length=2, max_length=10)


class TranslateResponse(BaseModel):
    translatedText: str
    sourceLang: str
    targetLang: str


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return INDEX_FILE.read_text(encoding="utf-8")


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    if request.sourceLang == request.targetLang:
        raise HTTPException(status_code=400, detail="Source and target languages must be different.")

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            "https://api.mymemory.translated.net/get",
            params={
                "q": request.text,
                "langpair": f"{request.sourceLang}|{request.targetLang}",
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Translation service returned an error.")

    payload = response.json()
    if payload.get("responseStatus") != 200:
        detail = payload.get("responseDetails") or "Translation failed."
        raise HTTPException(status_code=502, detail=detail)

    translated_text = payload.get("responseData", {}).get("translatedText")
    if translated_text is None:
        raise HTTPException(status_code=502, detail="Invalid translation response.")

    return TranslateResponse(
        translatedText=translated_text,
        sourceLang=request.sourceLang,
        targetLang=request.targetLang,
    )


@app.get("/health")
async def health_check():
    return {"status": "ok"}
