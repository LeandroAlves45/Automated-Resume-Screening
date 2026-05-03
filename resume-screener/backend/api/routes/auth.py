"""Rotas FastAPI para autenticação e autorização."""

import logging

from fastapi import APIRouter, Depends, status, Header

from backend.api.db.models import User
from backend.api.models.schemas import (
    UserRegister,
    LoginRequest,
    RefreshTokenRequest,
    LogoutRequest,
    UserResponse,
    TokenResponse,
    RefreshTokenResponse,
)
from backend.api.services.auth_service import (
    AuthService,
    get_auth_service,
    get_current_user,
)
from backend.api.utils.errors import UnauthorizedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# ============ Endpoints ============


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=UserResponse,
    summary="Register a new user",
)
async def register(
    register_data: UserRegister,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Registar um novo utilizador no sistema.

    Fluxo:
    1. Validar que email não existe (Pydantic + DB check)
    2. Hash password com bcrypt (COST 12)
    3. Criar User no DB
    4. Retornar UserResponse com dados do novo utilizador

    Args:
        register_data: UserRegister {email, password}
        auth_service: AuthService injetado

    Returns:
        201 Created: UserResponse {user_id, email, role}

    Raises:
        400 Bad Request: Email já registado
        422 Unprocessable Entity: Validação Pydantic falhou
    """

    logger.info("Trying to register new user: %s", register_data.email)

    user_response = auth_service.register(register_data)

    logger.info("User registered successfully: %s", user_response.email)

    return user_response


@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    response_model=TokenResponse,
    summary="Login and get tokens",
)
async def login(
    credentials: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """
    Autenticar utilizador com email e password.

    Fluxo:
    1. Validar credenciais (email existe + password correto)
    2. Criar access_token (exp 1h)
    3. Criar refresh_token (exp 7d) e guardar em BD
    4. Retornar ambos tokens

    Args:
        credentials: LoginRequest {email, password}
        auth_service: AuthService injetado

    Returns:
        200 OK: TokenResponse {access_token, refresh_token, token_type}

    Raises:
        401 Unauthorized: Email/password inválidos
        422 Unprocessable Entity: Validação Pydantic falhou
    """

    logger.info("Trying to login user: %s", credentials.email)

    tokens = auth_service.login(credentials.email, credentials.password)

    logger.info("Login successful for user: %s", credentials.email)

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
    )


@router.post(
    "/refresh",
    status_code=status.HTTP_200_OK,
    response_model=RefreshTokenResponse,
    summary="Refresh access token",
)
async def refresh(
    request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> RefreshTokenResponse:
    """
    Renovar access token usando refresh token.

    Fluxo:
    1. Validar refresh_token (não em blacklist, JWT válido)
    2. Extrair user_id do payload
    3. Criar novo access_token (exp 1h)
    4. Retornar novo access_token

    Args:
        request: RefreshTokenRequest {refresh_token}
        auth_service: AuthService injetado

    Returns:
        200 OK: RefreshTokenResponse {access_token, token_type}

    Raises:
        401 Unauthorized: Refresh token inválido, expirado ou revogado
        422 Unprocessable Entity: Validação Pydantic falhou
    """

    logger.info("Trying to refresh the access token")

    new_tokens = auth_service.refresh_access_token(request.refresh_token)

    logger.info("Access token refreshed successfully")

    return RefreshTokenResponse(
        access_token=new_tokens["access_token"],
        token_type=new_tokens["token_type"],
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout and revoke tokens",
)
async def logout(
    logout_request: LogoutRequest,
    current_user: User = Depends(get_current_user),
    authorization: str = Header(...),
    auth_service: AuthService = Depends(get_auth_service),
) -> None:
    """
    Fazer logout revogando access token e refresh token.

    Fluxo:
    1. Validar que utilizador está autenticado (via get_current_user dependency)
    2. Extrair access_token do header Authorization
    3. Extrair refresh_token do request body (opcional)
    4. Revogar ambos tokens (adicionar à blacklist, marcar em DB)
    5. Retornar 204 No Content

    Args:
        logout_request: LogoutRequest {refresh_token?}
        current_user: User injetado via get_current_user (valida access_token)
        authorization: Header Authorization com access_token
        auth_service: AuthService injetado

    Returns:
        204 No Content: sem body

    Raises:
        401 Unauthorized: Access token inválido (capturado em get_current_user)
        422 Unprocessable Entity: Validação Pydantic falhou
    """

    try:
        _, access_token = authorization.split()
    except ValueError as exc:
        logger.warning("Invalid authorization header: %s", exc)
        raise UnauthorizedError(
            detail="Invalid authorization header",
        ) from exc

    logger.info("Logout for the user: %s", current_user.email)

    auth_service.logout(
        user_id=current_user.id,
        access_token=access_token,
        refresh_token=logout_request.refresh_token,
    )

    logger.info("Tokens revoked successfully for user: %s", current_user.email)

    return None
