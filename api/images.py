import asyncio
import json
import os
import re
import time
import uuid
from urllib.parse import urlparse

import pybase64
from fastapi import File, Form, HTTPException, Request, Security, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from app import app, security_scheme
from chatgpt.ChatService import ChatService
from utils.Client import Client
from utils.Logger import logger
from utils.configs import api_prefix, file_host
from utils.tokenStats import record_token_usage


IMAGE_MODEL_ALIASES = {"gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini", "chatgpt-image-latest"}
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_CHATGPT_IMAGE_MODEL = os.getenv("IMAGE_GENERATION_MODEL", "gpt-4o")
MAX_IMAGE_COUNT = int(os.getenv("IMAGE_GENERATION_MAX_N", "4"))
IMAGE_GENERATION_POLL_TIMEOUT = int(os.getenv("IMAGE_GENERATION_POLL_TIMEOUT", "240"))
IMAGE_GENERATION_POLL_INTERVAL = int(os.getenv("IMAGE_GENERATION_POLL_INTERVAL", "5"))
GENERATED_IMAGES_DIR = os.path.join("data", "generated_images")
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = DEFAULT_IMAGE_MODEL
    n: int = Field(default=1, ge=1)
    size: str = "1024x1024"
    quality: str = "auto"
    background: str | None = None
    response_format: str = "url"
    output_format: str | None = None
    user: str | None = None
    chatgpt_model: str | None = None


def images_path(path):
    return f"/{api_prefix}/v1/images{path}" if api_prefix else f"/v1/images{path}"


def public_images_path(path):
    return f"/{api_prefix}/v1/images{path}" if api_prefix else f"/v1/images{path}"


def resolve_chatgpt_image_model(requested_model, chatgpt_model=None):
    if chatgpt_model:
        return chatgpt_model
    if requested_model in IMAGE_MODEL_ALIASES:
        return DEFAULT_CHATGPT_IMAGE_MODEL
    return requested_model or DEFAULT_CHATGPT_IMAGE_MODEL


def build_generation_prompt(prompt, size="1024x1024", quality="auto", background=None, output_format=None, edit=False):
    requirements = [
        "Create the requested image with ChatGPT image generation.",
        f"User prompt: {prompt}",
        f"Target size or aspect ratio: {size}",
        f"Quality preference: {quality}",
    ]
    if background:
        requirements.append(f"Background preference: {background}")
    if output_format:
        requirements.append(f"Preferred output format: {output_format}")
    if edit:
        requirements.extend([
            "Use the attached image or images as visual references and apply the user's requested edit.",
            "Make the result visibly different from the source image. Do not return a near-identical photo.",
            "Preserve the main subject and core pose when helpful, but clearly add creative transformation.",
            "For reference-image edits, always make obvious visual changes such as smart retouching, doodles, text, stickers, brush marks, graphic overlays, color accents, or background stylization when requested or appropriate.",
            "If the prompt is broad or ambiguous, favor a stronger creative editorial transformation over minimal beautification.",
        ])
    requirements.append("Return the final generated image. Keep any text response brief.")
    return "\n".join(requirements)


def extract_image_urls(content):
    if not content:
        return []
    urls = []
    for match in re.finditer(r"!\[[^\]]*]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", content):
        urls.append(match.group(1))
    for match in re.finditer(r"https?://[^\s)]+", content):
        url = match.group(0)
        if url not in urls and any(ext in url.lower() for ext in [".png", ".jpg", ".jpeg", ".webp", "download", "oaiusercontent"]):
            urls.append(url)
    return urls


async def fetch_image_as_b64(url):
    client = Client()
    try:
        response = await client.get(url, timeout=60)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch generated image")
        return pybase64.b64encode(response.content).decode("utf-8")
    finally:
        await client.close()


def image_extension_from_content_type(content_type, fallback_url=""):
    content_type = (content_type or "").split(";")[0].strip().lower()
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    if content_type in mapping:
        return mapping[content_type]
    path = urlparse(fallback_url).path.lower()
    for extension in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        if path.endswith(extension):
            return extension
    return ".png"


async def download_image_with_service(service, url):
    headers = service.base_headers.copy()
    response = await service.s.get(url, headers=headers, timeout=60)
    if response.status_code != 200:
        response = await service.s.get(url, timeout=60)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to download generated image: {response.status_code}")
    return {
        "url": url,
        "bytes": response.content,
        "content_type": response.headers.get("content-type", "image/png"),
    }


def save_generated_image(image):
    extension = image_extension_from_content_type(image.get("content_type"), image.get("url", ""))
    image_id = f"img-{uuid.uuid4().hex}{extension}"
    image_path = os.path.join(GENERATED_IMAGES_DIR, image_id)
    with open(image_path, "wb") as f:
        f.write(image["bytes"])
    return image_id


def public_url_for_image(request, image_id):
    host = file_host or str(request.base_url).rstrip("/")
    return host.rstrip("/") + public_images_path(f"/files/{image_id}")


def normalize_image_size(size):
    if not size or str(size).lower() == "auto":
        return "1024x1024"
    return size


async def extract_image_urls_from_conversation(service, conversation_id, conversation):
    urls = []

    async def visit(value):
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, dict):
                parts = content.get("parts", [])
                if content.get("content_type") == "multimodal_text":
                    for part in parts:
                        if isinstance(part, str):
                            urls.extend(extract_image_urls(part))
                            continue
                        if not isinstance(part, dict):
                            continue
                        if part.get("content_type") != "image_asset_pointer":
                            continue
                        asset_pointer = part.get("asset_pointer", "")
                        if asset_pointer.startswith("file-service://"):
                            file_id = asset_pointer.replace("file-service://", "")
                            url = await service.get_download_url(file_id)
                        else:
                            file_id = asset_pointer.replace("sediment://", "")
                            url = await service.get_attachment_url(file_id, conversation_id)
                        if url:
                            urls.append(url)
                elif parts:
                    for part in parts:
                        if isinstance(part, str):
                            urls.extend(extract_image_urls(part))
            for child in value.values():
                await visit(child)
        elif isinstance(value, list):
            for child in value:
                await visit(child)
        elif isinstance(value, str):
            urls.extend(extract_image_urls(value))

    await visit(conversation)
    return list(dict.fromkeys(urls))


async def poll_conversation_for_images(service, conversation_id, timeout=IMAGE_GENERATION_POLL_TIMEOUT):
    deadline = time.time() + timeout
    last_detail = None
    while time.time() < deadline:
        response = await service.s.get(
            f"{service.base_url}/conversation/{conversation_id}",
            headers=service.base_headers,
            timeout=15,
        )
        if response.status_code == 200:
            conversation = response.json()
            urls = await extract_image_urls_from_conversation(service, conversation_id, conversation)
            if urls:
                return urls, conversation
            last_detail = {
                "async_status": conversation.get("async_status"),
                "title": conversation.get("title"),
            }
        else:
            last_detail = response.text[:300]
        await asyncio.sleep(IMAGE_GENERATION_POLL_INTERVAL)
    return [], last_detail


async def run_image_conversation(req_token, request_data):
    chat_service = ChatService(req_token)
    try:
        await chat_service.set_dynamic_data(request_data)
        await chat_service.get_chat_requirements()
        await chat_service.prepare_send_conversation()
        response = await chat_service.send_conversation()
        content = ""
        conversation_id = None
        message_id = None
        image_urls = []
        async for event in response:
            if not isinstance(event, str) or not event.startswith("data: "):
                continue
            payload = event[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            conversation_id = chunk.get("conversation_id", conversation_id)
            message_id = chunk.get("message_id", message_id)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content += delta.get("content", "")
            image_urls.extend(extract_image_urls(delta.get("content", "")))
        image_urls = list(dict.fromkeys(image_urls))
        conversation_detail = None
        if not image_urls and conversation_id:
            image_urls, conversation_detail = await poll_conversation_for_images(chat_service, conversation_id)
        images = []
        for image_url in image_urls:
            try:
                images.append(await download_image_with_service(chat_service, image_url))
            except HTTPException as e:
                logger.error(f"Failed to persist generated image: {e.detail}")
        result = {
            "content": content,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "image_urls": image_urls,
            "images": images,
            "conversation_detail": conversation_detail,
        }
        record_token_usage(
            chat_service.req_token,
            request_data.get("_usage_type", "image"),
            getattr(chat_service, "origin_model", request_data.get("model")),
            success=True,
            status_code=200,
        )
        return result
    except HTTPException as e:
        record_token_usage(
            chat_service.req_token,
            request_data.get("_usage_type", "image"),
            getattr(chat_service, "origin_model", request_data.get("model")),
            success=False,
            status_code=e.status_code,
            error=e.detail,
        )
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        record_token_usage(
            chat_service.req_token,
            request_data.get("_usage_type", "image"),
            getattr(chat_service, "origin_model", request_data.get("model")),
            success=False,
            status_code=500,
            error=str(e),
        )
        logger.error(f"Image generation server error: {e}")
        raise HTTPException(status_code=500, detail="Image generation server error")
    finally:
        await chat_service.close_client()


async def image_response_from_chat(req_token, payload, messages, request):
    n = min(max(payload.n, 1), MAX_IMAGE_COUNT)
    data = []
    raw_messages = []
    for index in range(n):
        request_messages = messages
        if n > 1:
            source_content = messages[0]["content"]
            if isinstance(source_content, list):
                source_content = list(source_content)
                source_content[0] = {
                    **source_content[0],
                    "text": source_content[0].get("text", "") + f"\nVariant index: {index + 1}. Make this variant visually distinct.",
                }
            else:
                source_content = source_content + f"\nVariant index: {index + 1}. Make this variant visually distinct."
            request_messages = [
                {
                    "role": "user",
                    "content": source_content,
                }
            ]
        request_data = {
            "model": resolve_chatgpt_image_model(payload.model, payload.chatgpt_model),
            "messages": request_messages,
            "stream": True,
            "history_disabled": False,
            "_usage_type": "image",
        }
        response = await run_image_conversation(req_token, request_data)
        content = response.get("content", "")
        raw_messages.append(content)
        image_urls = response.get("image_urls") or extract_image_urls(content)
        images = response.get("images") or []
        if not image_urls and not images:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "No image URL found in ChatGPT response",
                    "content": content,
                    "conversation_id": response.get("conversation_id"),
                    "message_id": response.get("message_id"),
                    "conversation_detail": response.get("conversation_detail"),
                },
            )
        if payload.response_format == "b64_json":
            if images:
                image_b64 = pybase64.b64encode(images[-1]["bytes"]).decode("utf-8")
            else:
                image_b64 = await fetch_image_as_b64(image_urls[-1])
            data.append({"b64_json": image_b64, "revised_prompt": payload.prompt})
        else:
            if images:
                image_id = save_generated_image(images[-1])
                image_url = public_url_for_image(request, image_id)
            else:
                image_url = image_urls[-1]
            data.append({"url": image_url, "revised_prompt": payload.prompt})
        if index + 1 < n:
            await asyncio.sleep(0.2)
    return {
        "created": int(time.time()),
        "data": data,
        "model": payload.model,
        "object": "list",
        "chatgpt_model": resolve_chatgpt_image_model(payload.model, payload.chatgpt_model),
        "raw_messages": raw_messages,
    }


@app.post(images_path("/generations"))
async def create_image(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
):
    try:
        payload = ImageGenerationRequest.model_validate(await request.json())
    except UnicodeDecodeError:
        raw_body = await request.body()
        payload = ImageGenerationRequest.model_validate_json(raw_body.decode("gbk"))
    prompt = build_generation_prompt(
        payload.prompt,
        size=payload.size,
        quality=payload.quality,
        background=payload.background,
        output_format=payload.output_format,
    )
    messages = [{"role": "user", "content": prompt}]
    response = await image_response_from_chat(credentials.credentials, payload, messages, request)
    return JSONResponse(response, media_type="application/json; charset=utf-8")


@app.post(images_path("/edits"))
async def create_image_edit(
    request: Request,
    prompt: str = Form(...),
    image: list[UploadFile] = File(...),
    model: str = Form(DEFAULT_IMAGE_MODEL),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    quality: str = Form("auto"),
    background: str | None = Form(None),
    response_format: str = Form("url"),
    output_format: str | None = Form(None),
    chatgpt_model: str | None = Form(None),
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
):
    size = normalize_image_size(size)
    content = [
        {
            "type": "text",
            "text": build_generation_prompt(
                prompt,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
                edit=True,
            ),
        }
    ]
    for upload in image[:16]:
        body = await upload.read()
        encoded = pybase64.b64encode(body).decode("utf-8")
        mime_type = upload.content_type or "image/png"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}})

    payload = ImageGenerationRequest(
        prompt=prompt,
        model=model,
        n=n,
        size=size,
        quality=quality,
        background=background,
        response_format=response_format,
        output_format=output_format,
        chatgpt_model=chatgpt_model,
    )
    response = await image_response_from_chat(credentials.credentials, payload, [{"role": "user", "content": content}], request)
    return JSONResponse(response, media_type="application/json; charset=utf-8")


@app.get(images_path("/models"))
async def image_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-image-2", "object": "model", "owned_by": "openai"},
            {"id": "gpt-image-1.5", "object": "model", "owned_by": "openai"},
            {"id": "gpt-image-1", "object": "model", "owned_by": "openai"},
            {"id": "gpt-image-1-mini", "object": "model", "owned_by": "openai"},
            {"id": "chatgpt-image-latest", "object": "model", "owned_by": "openai"},
        ],
    }


@app.get(images_path("/files/{image_id}"))
async def get_generated_image(image_id: str):
    image_id = os.path.basename(image_id)
    image_path = os.path.join(GENERATED_IMAGES_DIR, image_id)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Generated image not found")
    return FileResponse(image_path)
