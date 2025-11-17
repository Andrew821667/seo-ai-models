"""
UserManagement - Модуль для управления пользователями через панель управления.
Обеспечивает функциональность аутентификации, авторизации и управления пользователями.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import hashlib
import secrets

logger = logging.getLogger(__name__)


class User:
    """Класс, представляющий пользователя в системе."""

    def __init__(
        self,
        user_id: str,
        username: str,
        email: str,
        password_hash: str,
        first_name: str = "",
        last_name: str = "",
        role: str = "user",
        status: str = "active",
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        last_login: Optional[datetime] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Инициализирует пользователя.

        Args:
            user_id: Уникальный идентификатор пользователя
            username: Имя пользователя
            email: Email пользователя
            password_hash: Хеш пароля
            first_name: Имя
            last_name: Фамилия
            role: Роль пользователя (admin, manager, user)
            status: Статус пользователя (active, inactive, blocked)
            created_at: Время создания
            updated_at: Время последнего обновления
            last_login: Время последнего входа
            settings: Настройки пользователя
        """
        self.user_id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.first_name = first_name
        self.last_name = last_name
        self.role = role
        self.status = status
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.last_login = last_login
        self.settings = settings or {}

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует пользователя в словарь."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Создает пользователя из словаря."""
        # Обрабатываем даты, которые приходят в виде строк
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at")
        )
        updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at")
        )
        last_login = (
            datetime.fromisoformat(data["last_login"])
            if isinstance(data.get("last_login"), str) and data.get("last_login")
            else None
        )

        return cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            role=data.get("role", "user"),
            status=data.get("status", "active"),
            created_at=created_at,
            updated_at=updated_at,
            last_login=last_login,
            settings=data.get("settings", {}),
        )

    def get_full_name(self) -> str:
        """Возвращает полное имя пользователя."""
        if self.first_name or self.last_name:
            return f"{self.first_name} {self.last_name}".strip()
        return self.username


class Session:
    """Класс, представляющий сессию пользователя."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        token: str,
        expires_at: datetime,
        created_at: Optional[datetime] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        is_active: bool = True,
    ):
        """
        Инициализирует сессию.

        Args:
            session_id: Уникальный идентификатор сессии
            user_id: ID пользователя
            token: Токен сессии
            expires_at: Время истечения срока действия
            created_at: Время создания
            ip_address: IP-адрес
            user_agent: User-Agent браузера
            is_active: Активна ли сессия
        """
        self.session_id = session_id
        self.user_id = user_id
        self.token = token
        self.expires_at = expires_at
        self.created_at = created_at or datetime.now()
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.is_active = is_active

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует сессию в словарь."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "token": self.token,
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Создает сессию из словаря."""
        # Обрабатываем даты, которые приходят в виде строк
        expires_at = (
            datetime.fromisoformat(data["expires_at"])
            if isinstance(data.get("expires_at"), str)
            else data.get("expires_at")
        )
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at")
        )

        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            token=data["token"],
            expires_at=expires_at,
            created_at=created_at,
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            is_active=data.get("is_active", True),
        )

    def is_expired(self) -> bool:
        """Проверяет, истек ли срок действия сессии."""
        return datetime.now() > self.expires_at


class UserManagement:
    """
    Класс для управления пользователями.
    """

    def __init__(self, data_dir: Optional[str] = None, api_client=None):
        """
        Инициализирует управление пользователями.

        Args:
            data_dir: Директория для хранения данных пользователей (для локального режима)
            api_client: Клиент API для взаимодействия с бэкендом
        """
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".seo_ai_models", "users")
        self.api_client = api_client
        self.users = {}
        self.sessions = {}

        # Создаем директорию для данных, если она не существует
        os.makedirs(self.data_dir, exist_ok=True)

        # Загружаем существующих пользователей и сессии
        self._load_users()
        self._load_sessions()

        # Создаем администратора по умолчанию, если нет пользователей
        if not self.users:
            self._create_default_admin()

    def _load_users(self):
        """Загружает существующих пользователей из хранилища."""
        users_dir = os.path.join(self.data_dir, "users")

        # Создаем директорию, если она не существует
        os.makedirs(users_dir, exist_ok=True)

        # Загружаем пользователей
        for user_file in Path(users_dir).glob("*.json"):
            try:
                with open(user_file, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                    user = User.from_dict(user_data)
                    self.users[user.user_id] = user
            except Exception as e:
                logger.error(f"Failed to load user from {user_file}: {str(e)}")

    def _load_sessions(self):
        """Загружает существующие сессии из хранилища."""
        sessions_dir = os.path.join(self.data_dir, "sessions")

        # Создаем директорию, если она не существует
        os.makedirs(sessions_dir, exist_ok=True)

        # Загружаем сессии
        for session_file in Path(sessions_dir).glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    session = Session.from_dict(session_data)

                    # Пропускаем истекшие сессии
                    if session.is_expired():
                        continue

                    self.sessions[session.session_id] = session
            except Exception as e:
                logger.error(f"Failed to load session from {session_file}: {str(e)}")

    def _create_default_admin(self):
        """Создает администратора по умолчанию."""
        # Генерируем уникальный ID для пользователя
        user_id = str(uuid.uuid4())

        # Хешируем пароль по умолчанию
        password_hash = self._hash_password("admin123")

        # Создаем пользователя
        user = User(
            user_id=user_id,
            username="admin",
            email="admin@example.com",
            password_hash=password_hash,
            first_name="Admin",
            last_name="User",
            role="admin",
        )

        # Сохраняем пользователя
        self.users[user_id] = user
        self._save_user(user)

        logger.info("Created default admin user")

    def _hash_password(self, password: str) -> str:
        """
        Хеширует пароль.

        Args:
            password: Пароль

        Returns:
            str: Хеш пароля
        """
        # Для простоты используем SHA-256
        # В реальном приложении лучше использовать bcrypt или Argon2
        return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        Проверяет пароль.

        Args:
            password: Пароль
            password_hash: Хеш пароля

        Returns:
            bool: True, если пароль верный, иначе False
        """
        return self._hash_password(password) == password_hash

    def _save_user(self, user: User):
        """
        Сохраняет пользователя в хранилище.

        Args:
            user: Пользователь для сохранения
        """
        users_dir = os.path.join(self.data_dir, "users")
        os.makedirs(users_dir, exist_ok=True)

        user_file = os.path.join(users_dir, f"{user.user_id}.json")

        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(user.to_dict(), f, indent=2, ensure_ascii=False)

    def _save_session(self, session: Session):
        """
        Сохраняет сессию в хранилище.

        Args:
            session: Сессия для сохранения
        """
        sessions_dir = os.path.join(self.data_dir, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        session_file = os.path.join(sessions_dir, f"{session.session_id}.json")

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: str = "",
        last_name: str = "",
        role: str = "user",
    ) -> User:
        """
        Создает нового пользователя.

        Args:
            username: Имя пользователя
            email: Email пользователя
            password: Пароль
            first_name: Имя
            last_name: Фамилия
            role: Роль пользователя

        Returns:
            User: Созданный пользователь
        """
        # Проверяем уникальность имени пользователя и email
        for user in self.users.values():
            if user.username == username:
                raise ValueError(f"User with username '{username}' already exists")
            if user.email == email:
                raise ValueError(f"User with email '{email}' already exists")

        # Генерируем уникальный ID для пользователя
        user_id = str(uuid.uuid4())

        # Хешируем пароль
        password_hash = self._hash_password(password)

        # Создаем пользователя
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            role=role,
        )

        # Сохраняем пользователя
        self.users[user_id] = user
        self._save_user(user)

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Получает пользователя по ID.

        Args:
            user_id: ID пользователя

        Returns:
            Optional[User]: Пользователь, если найден, иначе None
        """
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Получает пользователя по имени пользователя.

        Args:
            username: Имя пользователя

        Returns:
            Optional[User]: Пользователь, если найден, иначе None
        """
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Получает пользователя по email.

        Args:
            email: Email пользователя

        Returns:
            Optional[User]: Пользователь, если найден, иначе None
        """
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    def get_users(self, role: Optional[str] = None, status: Optional[str] = None) -> List[User]:
        """
        Получает список пользователей с возможностью фильтрации.

        Args:
            role: Фильтр по роли
            status: Фильтр по статусу

        Returns:
            List[User]: Список пользователей
        """
        result = []

        for user in self.users.values():
            if role and user.role != role:
                continue
            if status and user.status != status:
                continue
            result.append(user)

        return result

    def update_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        email: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        role: Optional[str] = None,
        status: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[User]:
        """
        Обновляет пользователя.

        Args:
            user_id: ID пользователя
            username: Новое имя пользователя
            email: Новый email
            first_name: Новое имя
            last_name: Новая фамилия
            role: Новая роль
            status: Новый статус
            settings: Новые настройки

        Returns:
            Optional[User]: Обновленный пользователь, если найден, иначе None
        """
        user = self.get_user(user_id)
        if not user:
            return None

        # Проверяем уникальность имени пользователя и email
        if username and username != user.username:
            for other_user in self.users.values():
                if other_user.user_id != user_id and other_user.username == username:
                    raise ValueError(f"User with username '{username}' already exists")
            user.username = username

        if email and email != user.email:
            for other_user in self.users.values():
                if other_user.user_id != user_id and other_user.email == email:
                    raise ValueError(f"User with email '{email}' already exists")
            user.email = email

        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        if role:
            user.role = role
        if status:
            user.status = status
        if settings:
            user.settings.update(settings)

        user.updated_at = datetime.now()

        # Сохраняем пользователя
        self._save_user(user)

        return user

    def delete_user(self, user_id: str) -> bool:
        """
        Удаляет пользователя.

        Args:
            user_id: ID пользователя

        Returns:
            bool: True, если пользователь успешно удален, иначе False
        """
        user = self.get_user(user_id)
        if not user:
            return False

        # Помечаем пользователя как неактивного
        user.status = "inactive"
        user.updated_at = datetime.now()

        # Сохраняем пользователя
        self._save_user(user)

        # Удаляем сессии пользователя
        for session_id, session in list(self.sessions.items()):
            if session.user_id == user_id:
                session.is_active = False
                self._save_session(session)
                del self.sessions[session_id]

        return True

    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """
        Изменяет пароль пользователя.

        Args:
            user_id: ID пользователя
            current_password: Текущий пароль
            new_password: Новый пароль

        Returns:
            bool: True, если пароль успешно изменен, иначе False
        """
        user = self.get_user(user_id)
        if not user:
            return False

        # Проверяем текущий пароль
        if not self._verify_password(current_password, user.password_hash):
            return False

        # Обновляем пароль
        user.password_hash = self._hash_password(new_password)
        user.updated_at = datetime.now()

        # Сохраняем пользователя
        self._save_user(user)

        return True

    def reset_password(self, user_id: str, new_password: str) -> bool:
        """
        Сбрасывает пароль пользователя (для администратора).

        Args:
            user_id: ID пользователя
            new_password: Новый пароль

        Returns:
            bool: True, если пароль успешно сброшен, иначе False
        """
        user = self.get_user(user_id)
        if not user:
            return False

        # Обновляем пароль
        user.password_hash = self._hash_password(new_password)
        user.updated_at = datetime.now()

        # Сохраняем пользователя
        self._save_user(user)

        return True

    def login(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_duration: int = 86400,
    ) -> Optional[Session]:
        """
        Выполняет вход пользователя.

        Args:
            username: Имя пользователя или email
            password: Пароль
            ip_address: IP-адрес
            user_agent: User-Agent браузера
            session_duration: Продолжительность сессии в секундах (по умолчанию 24 часа)

        Returns:
            Optional[Session]: Сессия, если вход выполнен успешно, иначе None
        """
        # Ищем пользователя по имени пользователя или email
        user = self.get_user_by_username(username)
        if not user:
            user = self.get_user_by_email(username)

        if not user:
            return None

        # Проверяем пароль
        if not self._verify_password(password, user.password_hash):
            return None

        # Проверяем статус пользователя
        if user.status != "active":
            return None

        # Генерируем уникальный ID для сессии
        session_id = str(uuid.uuid4())

        # Генерируем токен
        token = secrets.token_hex(32)

        # Устанавливаем время истечения срока действия
        expires_at = datetime.now() + timedelta(seconds=session_duration)

        # Создаем сессию
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            token=token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Обновляем время последнего входа пользователя
        user.last_login = datetime.now()
        user.updated_at = datetime.now()

        # Сохраняем сессию и пользователя
        self.sessions[session_id] = session
        self._save_session(session)
        self._save_user(user)

        return session

    def logout(self, session_id: str) -> bool:
        """
        Выполняет выход пользователя.

        Args:
            session_id: ID сессии

        Returns:
            bool: True, если выход выполнен успешно, иначе False
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Деактивируем сессию
        session.is_active = False

        # Сохраняем сессию
        self._save_session(session)

        # Удаляем сессию из списка активных
        del self.sessions[session_id]

        return True

    def validate_token(self, token: str) -> Optional[Session]:
        """
        Проверяет токен сессии.

        Args:
            token: Токен сессии

        Returns:
            Optional[Session]: Сессия, если токен действителен, иначе None
        """
        for session in self.sessions.values():
            if session.token == token and session.is_active:
                # Проверяем срок действия
                if session.is_expired():
                    # Деактивируем сессию
                    session.is_active = False
                    self._save_session(session)
                    return None

                return session

        return None

    def get_user_by_token(self, token: str) -> Optional[User]:
        """
        Получает пользователя по токену сессии.

        Args:
            token: Токен сессии

        Returns:
            Optional[User]: Пользователь, если токен действителен, иначе None
        """
        session = self.validate_token(token)
        if not session:
            return None

        return self.get_user(session.user_id)

    def get_user_roles(self) -> List[str]:
        """
        Получает список доступных ролей пользователей.

        Returns:
            List[str]: Список ролей
        """
        return ["admin", "manager", "user"]

    def get_user_statuses(self) -> List[str]:
        """
        Получает список доступных статусов пользователей.

        Returns:
            List[str]: Список статусов
        """
        return ["active", "inactive", "blocked"]

    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Session]:
        """
        Получает список активных сессий.

        Args:
            user_id: ID пользователя (если указан, то только для этого пользователя)

        Returns:
            List[Session]: Список активных сессий
        """
        result = []

        for session in self.sessions.values():
            if user_id and session.user_id != user_id:
                continue
            if session.is_active and not session.is_expired():
                result.append(session)

        return result

    def terminate_sessions(self, user_id: str) -> int:
        """
        Завершает все сессии пользователя.

        Args:
            user_id: ID пользователя

        Returns:
            int: Количество завершенных сессий
        """
        count = 0

        for session_id, session in list(self.sessions.items()):
            if session.user_id == user_id and session.is_active:
                session.is_active = False
                self._save_session(session)
                del self.sessions[session_id]
                count += 1

        return count

    def get_user_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику по пользователям.

        Returns:
            Dict[str, Any]: Статистика по пользователям
        """
        active_users = len([u for u in self.users.values() if u.status == "active"])
        inactive_users = len([u for u in self.users.values() if u.status == "inactive"])
        blocked_users = len([u for u in self.users.values() if u.status == "blocked"])

        admins = len([u for u in self.users.values() if u.role == "admin"])
        managers = len([u for u in self.users.values() if u.role == "manager"])
        regular_users = len([u for u in self.users.values() if u.role == "user"])

        active_sessions = len(self.sessions)

        return {
            "total_users": len(self.users),
            "active_users": active_users,
            "inactive_users": inactive_users,
            "blocked_users": blocked_users,
            "admins": admins,
            "managers": managers,
            "regular_users": regular_users,
            "active_sessions": active_sessions,
        }
