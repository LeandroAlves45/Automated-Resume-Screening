"""Serviços de autenticação para a API."""

import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import Depends, Header

from backend.api.config import get_settings, Settings
from backend.api.db.database import get_db
from backend.api.db.models import User, RefreshToken, TokenBlacklist
from backend.api.models.schemas import UserResponse, UserRegister
from backend.api.utils.errors import ValidationError, UnauthorizedError

logger = logging.getLogger(__name__)

# ============ Password Hashing ============
# Instância única reutilizada em todas as operações de hash
# Usar argon2 como algoritmo principal (mais seguro e compatível)

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class AuthService:
    """
    Serviço centralizado de autenticação e autorização.

    Responsabilidades:
    - Registo de utilizadores com hash de password
    - Login com geração de access + refresh tokens
    - Refresh de access tokens
    - Logout com revogação de tokens
    - Validação de tokens para injeção em rotas (get_current_user)

    Design:
    - Access token: JWT com expiração 1h, stateless
    - Refresh token: JWT + guardado em DB com hash, permite logout imediato
    - Token blacklist: registos revogados até expiração
    """

    def __init__(self, db: Session, settings: Settings):
        """
        Inicializa AuthService com dependências.

        Args:
            db: Sessão SQLAlchemy para operações de BD
            settings: Configurações da aplicação (JWT_SECRET, expirations, etc)
        """

        self.db = db
        self.settings = settings
        logger.debug("AuthService initialized with db and settings")

    # ============ Métodos públicos ============

    def register(self, register_data: UserRegister) -> UserResponse:
        """
        Registar um novo utilizador.

        Fluxo:
        1. Verificar se email já existe
        2. Hash password com argon2
        3. Criar User no DB
        4. Retornar UserResponse (sem password)

        Args:
            register_data: UserRegister com email e password

        Returns:
            UserResponse com dados do novo utilizador

        Raises:
            ValidationError: Se email já está registado
        """

        # Varificar se email já está registado
        existing_user = (
            self.db.query(User).filter(User.email == register_data.email).first()
        )

        if existing_user:
            logger.warning("Email already registered: %s", register_data.email)
            raise ValidationError(
                detail=f"Email '{register_data.email}' is already registered",
            )

        # Hash password com argon2
        hashed_password = self._hash_password(register_data.password)

        # Criar novo User
        new_user = User(
            email=register_data.email,
            hashed_password=hashed_password,
        )

        self.db.add(new_user)
        self.db.commit()
        self.db.refresh(new_user)

        logger.info("User registered successfully: %s", new_user.email)

        return UserResponse(
            user_id=new_user.id,
            email=new_user.email,
            role=new_user.role,
        )

    def login(self, email: str, password: str) -> dict:
        """
        Autenticar utilizador e retornar tokens.

        Fluxo:
        1. Encontrar User por email
        2. Verificar password com argon2
        3. Criar access_token (exp 1h)
        4. Criar refresh_token (exp 7d) e guardar em DB
        5. Retornar ambos tokens

        Args:
            email: Email do utilizador
            password: Password em plaintext

        Returns:
            {
                "access_token": str,
                "refresh_token": str,
                "token_type": "bearer"
            }

        Raises:
            UnauthorizedError: Se email/password inválidos
        """

        # Encontrar User por email
        user = self.db.query(User).filter(User.email == email).first()

        if not user or not self._verify_password(password, user.hashed_password):
            logger.warning("Login failed for user %s: invalid credentials", email)
            raise UnauthorizedError(
                detail="Invalid email or password",
            )

        # Criar tokens
        access_token = self._create_token(
            data={"sub": str(user.id), "role": user.role},
            expire_delta=timedelta(
                seconds=self.settings.get_jwt_access_token_expire_seconds()
            ),
        )

        refresh_token = self._create_token(
            data={"sub": str(user.id), "type": "refresh"},
            expire_delta=timedelta(
                seconds=self.settings.get_jwt_refresh_token_expire_seconds()
            ),
        )

        # Guardar refresh_token em DB
        self._save_refresh_token(
            user_id=user.id,
            token=refresh_token,
            expire_seconds=self.settings.get_jwt_refresh_token_expire_seconds(),
        )

        logger.info("Login successful for user %s", email)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }

    def refresh_access_token(self, refresh_token: str) -> dict:
        """
        Gerar novo access_token a partir de refresh_token.

        Fluxo:
        1. Verificar se refresh_token está em blacklist
        2. Descodificar JWT e validar exp
        3. Extrair user_id
        4. Criar novo access_token
        5. Retornar novo token

        Args:
            refresh_token: Refresh token JWT válido

        Returns:
            {
                "access_token": str,
                "token_type": "bearer"
            }

        Raises:
            UnauthorizedError: Se token inválido, expirado ou revogado
        """

        # Verificar blacklist
        token_hash = self._hash_token(refresh_token)
        if self._is_token_blacklisted(token_hash):
            logger.warning("Try to refresh revoked refresh token")
            raise UnauthorizedError(
                detail="Refresh token has been revoked",
            )

        # Descodificar JWT e validar exp
        payload = self._verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            logger.warning("Invalid refresh token or type incorrect")
            raise UnauthorizedError(
                detail="Invalid refresh token or type incorrect",
            )

        user_id = payload.get("sub")
        if not user_id:
            logger.warning("Refresh token does not contain user_id")
            raise UnauthorizedError(
                detail="Invalid refresh token",
            )

        # Buscar user para obter role atualizado
        user = self.db.query(User).filter(User.id == UUID(user_id)).first()
        if not user:
            logger.warning("User not found for refresh token: %s", user_id)
            raise UnauthorizedError(
                detail="Invalid refresh token",
            )

        # Criar novo access_token
        new_access_token = self._create_token(
            data={"sub": user.id, "role": user.role},
            expire_delta=timedelta(
                seconds=self.settings.get_jwt_access_token_expire_seconds()
            ),
        )

        logger.info("Access token refreshed for user %s", user.email)

        return {
            "access_token": new_access_token,
            "token_type": "bearer",
        }

    def logout(
        self,
        user_id: UUID,
        access_token: str,
        refresh_token: Optional[str] = None,
    ) -> None:
        """
        Revogar tokens de um utilizador.

        Fluxo:
        1. Hash access_token e refresh_token
        2. Adicionar à TokenBlacklist
        3. Marcar refresh_token em DB como revogado
        4. Retornar None (204 No Content)

        Args:
            user_id: ID do utilizador
            access_token: JWT access token a revogar
            refresh_token: JWT refresh token a revogar (opcional)

        Returns:
            None
        """

        # Revogar access_token
        access_token_hash = self._hash_token(access_token)
        self._revoke_token(access_token_hash)

        # Revogar refresh_token se fornecido
        if refresh_token:
            refresh_token_hash = self._hash_token(refresh_token)
            self._revoke_token(refresh_token_hash)

            # Marcar em DB como revogado
            refresh_token_db = (
                self.db.query(RefreshToken)
                .filter(
                    RefreshToken.token_hash == refresh_token_hash,
                )
                .first()
            )

            if refresh_token_db:
                refresh_token_db.is_revoked = True
                self.db.commit()

        logger.info("Refresh token revoked for user %s", user_id)

    async def get_current_user(
        self,
        authorization: str = Header(...),
    ) -> User:
        """
        Validar JWT e retornar User atual (FastAPI Depends).

        Fluxo:
        1. Extrair token do header Authorization: Bearer {token}
        2. Verificar se está em blacklist
        3. Descodificar JWT
        4. Validar exp
        5. Buscar User no DB
        6. Retornar User ou lançar UnauthorizedError

        Args:
            authorization: Header Authorization com "Bearer {token}"

        Returns:
            User object do utilizador autenticado

        Raises:
            UnauthorizedError: Se token inválido, expirado, revogado ou user não existe
        """

        # Extrair token do header
        try:
            scheme, token = authorization.split()

            if scheme.lower() != "bearer":
                raise UnauthorizedError(
                    detail="Invalid authorization scheme",
                )
        except ValueError as exc:
            logger.warning("Invalid authorization header")
            raise UnauthorizedError(
                detail="Invalid authorization header",
            ) from exc

        # Verificar se está em blacklist
        token_hash = self._hash_token(token)
        if self._is_token_blacklisted(token_hash):
            logger.warning("Try to access with revoked token")
            raise UnauthorizedError(
                detail="Token has been revoked",
            )

        # Descodificar JWT
        payload = self._verify_token(token)
        if not payload or "sub" not in payload:
            logger.warning("Invalid token or missing sub")
            raise UnauthorizedError(
                detail="Invalid token",
            )

        user_id = payload.get("sub")

        # Buscar User no DB
        user = self.db.query(User).filter(User.id == UUID(user_id)).first()
        if not user or not user.is_active:
            logger.warning("User not found or inactive: %s", user_id)
            raise UnauthorizedError(
                detail="User not found or inactive",
            )

        return user

    # ============ Métodos privados ============

    def _hash_password(self, password: str) -> str:
        """
        Hash password com argon2.

        Args:
            password: Password em plaintext

        Returns:
            Password hashed com argon2
        """
        return pwd_context.hash(password)

    def _verify_password(self, password: str, hashed: str) -> bool:
        """
        Verificar password contra hash com argon2.

        Args:
            password: Password em plaintext
            hashed_password: Password hashed com argon2

        Returns:
            True se password é válida, False caso contrário
        """
        return pwd_context.verify(password, hashed)

    def _hash_token(self, token: str) -> str:
        """
        Hash token com SHA-256.

        Args:
            token: Token a hash

        Returns:
            Token hashed com SHA-256
        """
        return hashlib.sha256(token.encode()).hexdigest()

    def _create_token(self, data: dict, expire_delta: timedelta) -> str:
        """
        Criar token JWT.

        Args:
            data: Payload do token (ex: {"sub": user_id, "role": "recruiter"})
            expire_delta: Tempo de expiração desde agora

        Returns:
            JWT token assinado
        """

        to_encode = data.copy()

        # Adicionar expiração
        expire = datetime.now(timezone.utc) + expire_delta
        to_encode.update({"exp": expire})

        # Assinar com secret key
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
        )

        return encoded_jwt

    def _verify_token(self, token: str) -> Optional[dict]:
        """
        Descodificar e validar JWT.

        Valida assinatura e expiração. Retorna None se inválido.

        Args:
            token: JWT token

        Returns:
            Payload decodificado ou None se inválido/expirado
        """

        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[self.settings.jwt_algorithm],
            )

            return payload
        except JWTError as e:
            logger.warning("JWT error: %s", e)
            return None

    def _save_refresh_token(
        self,
        user_id: UUID,
        token: str,
        expire_seconds: int,
    ) -> None:
        """
        Guardar refresh token no DB.

        Args:
            user_id: ID do utilizador
            token: Refresh token JWT
            expire_seconds: Tempo de expiração em segundos
        """
        token_hash = self._hash_token(token)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expire_seconds)

        refresh_token_db = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )

        self.db.add(refresh_token_db)
        self.db.commit()

        logger.debug("Refresh token saved for user %s", user_id)

    def _revoke_token(self, token_hash: str) -> None:
        """
        Adicionar token à TokenBlacklist.

        Args:
            token_hash: Hash do token a revogar
        """
        # Calcular quando este token expira
        # Para access tokens = 1 hora, refresh tokens = 7 dias
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.settings.jwt_refresh_token_expire_days,
        )

        blacklist_entry = TokenBlacklist(
            token_hash=token_hash,
            expires_at=expires_at,
        )

        self.db.add(blacklist_entry)
        self.db.commit()

        logger.debug("Token added to blacklist, expires at %s", expires_at)

    def _is_token_blacklisted(self, token_hash: str) -> bool:
        """
        Verificar se token está na TokenBlacklist.

        Args:
            token_hash: Hash do token a verificar
        """
        blacklist_entry = (
            self.db.query(TokenBlacklist)
            .filter(
                TokenBlacklist.token_hash == token_hash,
                TokenBlacklist.expires_at > datetime.now(timezone.utc),
            )
            .first()
        )

        return blacklist_entry is not None


# ============ Dependency Injection Helper ============


async def get_auth_service(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuthService:
    """
    Criar instância de AuthService para injeção em rotas.

    Uso em rotas:
        @router.post("/login")
        async def login(
            credentials: LoginRequest,
            auth_service: AuthService = Depends(get_auth_service)
        ):
            return auth_service.login(credentials.email, credentials.password)

    Args:
        db: Sessão SQLAlchemy
        settings: Configurações da aplicação

    Returns:
        AuthService instance
    """
    return AuthService(db, settings)


async def get_current_user(
    authorization: str = Header(...),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """
    Dependência FastAPI: valida o JWT do header Authorization e devolve o User.

    Encapsula AuthService.get_current_user para uso com Depends(...) nas rotas.
    """
    return await auth_service.get_current_user(authorization=authorization)
