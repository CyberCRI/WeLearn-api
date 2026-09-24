# src/app/services/security.py

import hashlib
from typing import List

from fastapi import Depends, HTTPException, Security, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
import httpx
from sqlalchemy.sql import select
from welearn_database.data.models import APIKeyManagement

from src.app.services.sql_db.queries import session_maker
from src.app.utils.logger import logger as logger_utils

from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import jwt, jwk
from jose.exceptions import JWTError
from pydantic import BaseModel

# Configuration
KEYCLOAK_URL = "https://keycloak.k8s.lp-i.dev"
REALM_NAME = "lp"
KEYCLOAK_CLIENT_ID = "welearn-api-dev"


# JWKs URL
JWKS_URL = f"{KEYCLOAK_URL}/realms/{REALM_NAME}/protocol/openid-connect/certs"

# OAuth2 scheme
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{KEYCLOAK_URL}/realms/{REALM_NAME}/protocol/openid-connect/auth",
    tokenUrl=f"{KEYCLOAK_URL}/realms/{REALM_NAME}/protocol/openid-connect/token",
    auto_error=False,
)


api_key_header = APIKeyHeader(name="X-API-Key")
logger = logger_utils(__name__)


# Models
class TokenData(BaseModel):
    username: str
    roles: List[str]
    id: str


# Item model
class Item(BaseModel):
    name: str
    description: str
    price: float


# Token validation function
async def validate_token(token: str) -> TokenData:
    print(token)
    try:
        # Fetch JWKS
        async with httpx.AsyncClient() as client:
            response = await client.get(JWKS_URL)
            response.raise_for_status()
            jwks = response.json()

            print(">>>>>>>>>>>")

        # Decode the token headers to get the key ID (kid)
        headers = jwt.get_unverified_headers(token)
        kid = headers.get("kid")
        if not kid:
            raise HTTPException(status_code=401, detail="Token missing 'kid' header")

        # Find the correct key in the JWKS
        key_data = next((key for key in jwks["keys"] if key["kid"] == kid), None)
        if not key_data:
            raise HTTPException(
                status_code=401, detail="Matching key not found in JWKS"
            )

        # Convert JWK to RSA public key
        public_key = jwk.construct(key_data).public_key()

        # Verify the token
        payload = jwt.decode(
            token, key=public_key, algorithms=["RS256"], audience=KEYCLOAK_CLIENT_ID
        )

        # Extract username and roles
        username = payload.get("preferred_username")
        roles = payload.get("realm_access", {}).get("roles", [])
        id = payload.get("sub")
        if not username or not roles or not id:
            raise HTTPException(status_code=401, detail="Token missing required claims")

        return TokenData(username=username, roles=roles, id=id)

    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


def check_api_key_sync(api_key: str) -> bool:
    digest = hashlib.sha256(api_key.encode()).digest()
    statement = select(APIKeyManagement.digest, APIKeyManagement.is_active).where(
        APIKeyManagement.digest == digest
    )
    with session_maker() as s:
        keys = s.execute(statement).first()

    if not keys:
        return False

    return True


# Dependency to get the current user
async def get_current_user(
    api_key_header, token: str = Depends(oauth2_scheme)
) -> tuple[bool, TokenData]:
    is_valid = await run_in_threadpool(check_api_key_sync, api_key_header)

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    tokenData = await validate_token(token)

    return is_valid, tokenData


async def get_user(
    api_key_header: str = Security(api_key_header), token: str = Depends(oauth2_scheme)
):
    is_valid, tokenData = await get_current_user(api_key_header, token)
    if is_valid and tokenData.id:
        return "ok"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
    )
