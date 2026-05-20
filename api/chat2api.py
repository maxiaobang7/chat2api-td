import asyncio
import hashlib
import json
import random
import time
import types
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import jwt
from fastapi import Request, HTTPException, Form, Security
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette.background import BackgroundTask

import utils.globals as globals
from app import app, templates, security_scheme
from chatgpt.ChatService import ChatService
from chatgpt.authorization import refresh_all_tokens, is_refresh_token
from chatgpt.fp import get_fp
from chatgpt.refreshToken import rt2ac
from utils.Client import Client
from utils.Logger import logger
from utils.configs import api_prefix, scheduled_refresh, authorization_list, chatgpt_base_url_list, proxy_url_list, oai_language
from utils.retry import async_retry

scheduler = AsyncIOScheduler()
optional_security_scheme = HTTPBearer(auto_error=False)


class TokenTextPayload(BaseModel):
    text: str


class TokenPayload(BaseModel):
    token: str


class TokenCheckPayload(BaseModel):
    token: str
    live: bool = True
    force_refresh: bool = True


def tokens_path(path):
    return f"/{api_prefix}/tokens{path}" if api_prefix else f"/tokens{path}"


def require_token_admin(credentials: HTTPAuthorizationCredentials | None):
    if not authorization_list:
        return
    if not credentials or credentials.credentials not in authorization_list:
        raise HTTPException(status_code=401, detail="Invalid token management authorization")


def dedupe_tokens(tokens):
    seen = set()
    result = []
    for token in tokens:
        token = token.strip()
        if token and not token.startswith("#") and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def save_token_file():
    globals.token_list[:] = dedupe_tokens(globals.token_list)
    with open(globals.TOKENS_FILE, "w", encoding="utf-8") as f:
        for token in globals.token_list:
            f.write(token + "\n")


def save_error_token_file():
    globals.error_token_list[:] = dedupe_tokens(globals.error_token_list)
    with open(globals.ERROR_TOKENS_FILE, "w", encoding="utf-8") as f:
        for token in globals.error_token_list:
            f.write(token + "\n")


def token_kind(token):
    if token.startswith("eyJhbGciOi") or token.startswith("fk-"):
        return "access"
    if is_refresh_token(token):
        return "refresh"
    return "unknown"


def mask_token(token):
    if len(token) <= 18:
        return token[:4] + "..." if token else ""
    return f"{token[:10]}...{token[-8:]}"


def jwt_payload(token):
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return {}


def format_timestamp(timestamp):
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def describe_token(token):
    kind = token_kind(token)
    payload = jwt_payload(token) if kind == "access" else {}
    exp = payload.get("exp")
    cached = globals.refresh_map.get(token, {}) if kind == "refresh" else {}
    cached_access = cached.get("token", "")
    cached_payload = jwt_payload(cached_access) if cached_access else {}
    cached_at = cached.get("timestamp")
    is_error = token in globals.error_token_list
    status = "error" if is_error else "unknown"
    if kind == "access" and exp:
        status = "expired" if int(exp) <= int(time.time()) else "available"
    elif kind == "refresh" and cached_access:
        status = "cached"

    return {
        "id": hashlib.sha256(token.encode()).hexdigest(),
        "token": token,
        "masked": mask_token(token),
        "type": kind,
        "status": status,
        "in_error_list": is_error,
        "expires_at": format_timestamp(exp),
        "cached_access_masked": mask_token(cached_access) if cached_access else None,
        "cached_access_expires_at": format_timestamp(cached_payload.get("exp")),
        "last_refreshed_at": format_timestamp(cached_at),
    }


async def live_check_access_token(access_token):
    fp = get_fp(access_token).copy()
    fp_proxy_url = fp.pop("proxy_url", None)
    impersonate = fp.pop("impersonate", "safari15_3")
    session_id = hashlib.md5(access_token.encode()).hexdigest()
    if fp_proxy_url:
        proxy_url = fp_proxy_url.replace("{}", session_id)
    else:
        proxy_url = random.choice(proxy_url_list).replace("{}", session_id) if proxy_url_list else None
    host_url = random.choice(chatgpt_base_url_list) if chatgpt_base_url_list else "https://chatgpt.com"
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "oai-language": oai_language,
        "origin": host_url,
        "referer": f"{host_url}/",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "authorization": f"Bearer {access_token}",
    }
    headers.update(fp)
    client = Client(proxy=proxy_url, impersonate=impersonate)
    try:
        response = await client.get(
            f"{host_url}/backend-api/models?history_and_training_disabled=false",
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "ok": True,
                "status": "available",
                "message": f"AccessToken is usable. Models: {len(models)}",
            }
        return {
            "ok": False,
            "status": "error",
            "message": response.text[:300],
            "http_status": response.status_code,
        }
    finally:
        await client.close()


async def check_token_status(token, live=True, force_refresh=True):
    token = token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")
    kind = token_kind(token)
    if kind == "unknown":
        return {"ok": False, "status": "unknown", "message": "Unsupported token format", "token": describe_token(token)}

    access_token = token
    if kind == "refresh":
        try:
            access_token = await rt2ac(token, force_refresh=force_refresh)
            if token in globals.error_token_list:
                globals.error_token_list.remove(token)
                save_error_token_file()
        except HTTPException as e:
            if token not in globals.error_token_list:
                globals.error_token_list.append(token)
                save_error_token_file()
            return {"ok": False, "status": "error", "message": str(e.detail), "token": describe_token(token)}

    payload = jwt_payload(access_token)
    exp = payload.get("exp")
    if exp and int(exp) <= int(time.time()):
        if token not in globals.error_token_list:
            globals.error_token_list.append(token)
            save_error_token_file()
        return {"ok": False, "status": "expired", "message": "AccessToken is expired", "token": describe_token(token)}

    if not live:
        return {"ok": True, "status": "available", "message": "Token format and expiration look valid", "token": describe_token(token)}

    try:
        result = await live_check_access_token(access_token)
    except Exception as e:
        result = {"ok": False, "status": "error", "message": str(e)}

    if result["ok"]:
        if token in globals.error_token_list:
            globals.error_token_list.remove(token)
            save_error_token_file()
    else:
        if token not in globals.error_token_list:
            globals.error_token_list.append(token)
            save_error_token_file()
    result["token"] = describe_token(token)
    return result


@app.on_event("startup")
async def app_start():
    if scheduled_refresh:
        scheduler.add_job(id='refresh', func=refresh_all_tokens, trigger='cron', hour=3, minute=0, day='*/2',
                          kwargs={'force_refresh': True})
        scheduler.start()
        asyncio.get_event_loop().call_later(0, lambda: asyncio.create_task(refresh_all_tokens(force_refresh=False)))


async def to_send_conversation(request_data, req_token):
    chat_service = ChatService(req_token)
    try:
        await chat_service.set_dynamic_data(request_data)
        await chat_service.get_chat_requirements()
        return chat_service
    except HTTPException as e:
        await chat_service.close_client()
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        await chat_service.close_client()
        logger.error(f"Server error, {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")


async def process(request_data, req_token):
    chat_service = await to_send_conversation(request_data, req_token)
    await chat_service.prepare_send_conversation()
    res = await chat_service.send_conversation()
    return chat_service, res


@app.post(f"/{api_prefix}/v1/chat/completions" if api_prefix else "/v1/chat/completions")
async def send_conversation(request: Request, credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    req_token = credentials.credentials
    try:
        request_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "Invalid JSON body"})
    chat_service, res = await async_retry(process, request_data, req_token)
    try:
        if isinstance(res, types.AsyncGeneratorType):
            background = BackgroundTask(chat_service.close_client)
            return StreamingResponse(res, media_type="text/event-stream", background=background)
        else:
            background = BackgroundTask(chat_service.close_client)
            return JSONResponse(res, media_type="application/json", background=background)
    except HTTPException as e:
        await chat_service.close_client()
        if e.status_code == 500:
            logger.error(f"Server error, {str(e)}")
            raise HTTPException(status_code=500, detail="Server error")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        await chat_service.close_client()
        logger.error(f"Server error, {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")


@app.get(f"/{api_prefix}/tokens" if api_prefix else "/tokens", response_class=HTMLResponse)
async def upload_html(request: Request):
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return templates.TemplateResponse("tokens.html",
                                      {
                                          "request": request,
                                          "api_prefix": api_prefix,
                                          "tokens_count": tokens_count,
                                          "authorization_required": bool(authorization_list),
                                      })


@app.post(f"/{api_prefix}/tokens/upload" if api_prefix else "/tokens/upload")
async def upload_post(text: str = Form(...), credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    lines = text.split("\n")
    for line in lines:
        if line.strip() and not line.startswith("#"):
            globals.token_list.append(line.strip())
    save_token_file()
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return {"status": "success", "tokens_count": tokens_count}


@app.post(f"/{api_prefix}/tokens/clear" if api_prefix else "/tokens/clear")
async def clear_tokens(credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    globals.token_list.clear()
    globals.error_token_list.clear()
    globals.refresh_map.clear()
    save_token_file()
    save_error_token_file()
    with open(globals.REFRESH_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(globals.refresh_map, f, indent=4)
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return {"status": "success", "tokens_count": tokens_count}


@app.post(f"/{api_prefix}/tokens/error" if api_prefix else "/tokens/error")
async def error_tokens(credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    error_tokens_list = list(set(globals.error_token_list))
    return {"status": "success", "error_tokens": error_tokens_list}


@app.get(f"/{api_prefix}/tokens/add/{{token}}" if api_prefix else "/tokens/add/{token}")
async def add_token(token: str, credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    if token.strip() and not token.startswith("#"):
        globals.token_list.append(token.strip())
        save_token_file()
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return {"status": "success", "tokens_count": tokens_count}


@app.get(tokens_path("/manage/list"))
async def managed_tokens(credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    tokens = [describe_token(token) for token in dedupe_tokens(globals.token_list)]
    return {
        "status": "success",
        "tokens_count": len([token for token in tokens if not token["in_error_list"]]),
        "error_tokens_count": len(set(globals.error_token_list)),
        "tokens": tokens,
    }


@app.post(tokens_path("/manage/add"))
async def managed_add_tokens(payload: TokenTextPayload, credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    incoming = dedupe_tokens(payload.text.splitlines())
    existing = set(globals.token_list)
    added = []
    for token in incoming:
        if token not in existing:
            globals.token_list.append(token)
            existing.add(token)
            added.append(token)
        if token in globals.error_token_list:
            globals.error_token_list.remove(token)
    save_token_file()
    save_error_token_file()
    return {"status": "success", "added": len(added), "tokens_count": len(set(globals.token_list) - set(globals.error_token_list))}


@app.post(tokens_path("/manage/delete"))
async def managed_delete_token(payload: TokenPayload, credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    token = payload.token.strip()
    globals.token_list[:] = [item for item in globals.token_list if item != token]
    globals.error_token_list[:] = [item for item in globals.error_token_list if item != token]
    globals.refresh_map.pop(token, None)
    save_token_file()
    save_error_token_file()
    with open(globals.REFRESH_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(globals.refresh_map, f, indent=4)
    return {"status": "success", "tokens_count": len(set(globals.token_list) - set(globals.error_token_list))}


@app.post(tokens_path("/manage/check"))
async def managed_check_token(payload: TokenCheckPayload, credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    return await check_token_status(payload.token, live=payload.live, force_refresh=payload.force_refresh)


@app.post(tokens_path("/manage/refresh"))
async def managed_refresh_token(payload: TokenPayload, credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    token = payload.token.strip()
    if not is_refresh_token(token):
        raise HTTPException(status_code=400, detail="Only RefreshToken can refresh AccessToken")
    try:
        access_token = await rt2ac(token, force_refresh=True)
        if token in globals.error_token_list:
            globals.error_token_list.remove(token)
            save_error_token_file()
        payload = jwt_payload(access_token)
        return {
            "status": "success",
            "message": "AccessToken refreshed",
            "access_token_masked": mask_token(access_token),
            "access_token_expires_at": format_timestamp(payload.get("exp")),
            "token": describe_token(token),
        }
    except HTTPException as e:
        if token not in globals.error_token_list:
            globals.error_token_list.append(token)
            save_error_token_file()
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post(tokens_path("/manage/check-all"))
async def managed_check_all_tokens(credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    results = []
    for token in dedupe_tokens(globals.token_list):
        results.append(await check_token_status(token, live=True, force_refresh=False))
        await asyncio.sleep(0.2)
    return {"status": "success", "results": results}


@app.post(tokens_path("/manage/refresh-all"))
async def managed_refresh_all_tokens(credentials: HTTPAuthorizationCredentials | None = Security(optional_security_scheme)):
    require_token_admin(credentials)
    results = []
    for token in dedupe_tokens(globals.token_list):
        if not is_refresh_token(token):
            continue
        try:
            access_token = await rt2ac(token, force_refresh=True)
            if token in globals.error_token_list:
                globals.error_token_list.remove(token)
            payload = jwt_payload(access_token)
            results.append({
                "ok": True,
                "message": "AccessToken refreshed",
                "access_token_masked": mask_token(access_token),
                "access_token_expires_at": format_timestamp(payload.get("exp")),
                "token": describe_token(token),
            })
        except HTTPException as e:
            if token not in globals.error_token_list:
                globals.error_token_list.append(token)
            results.append({
                "ok": False,
                "message": str(e.detail),
                "token": describe_token(token),
            })
        await asyncio.sleep(0.2)
    save_error_token_file()
    return {"status": "success", "results": results}


@app.post(f"/{api_prefix}/seed_tokens/clear" if api_prefix else "/seed_tokens/clear")
async def clear_seed_tokens():
    globals.seed_map.clear()
    globals.conversation_map.clear()
    with open(globals.SEED_MAP_FILE, "w", encoding="utf-8") as f:
        f.write("{}")
    with open(globals.CONVERSATION_MAP_FILE, "w", encoding="utf-8") as f:
        f.write("{}")
    logger.info(f"Seed token count: {len(globals.seed_map)}")
    return {"status": "success", "seed_tokens_count": len(globals.seed_map)}
