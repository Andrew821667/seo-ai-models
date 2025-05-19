
"""
UserManagement - Модуль для управления пользователями и правами доступа.
Предоставляет функциональность для регистрации, аутентификации и
управления правами доступа пользователей в системе.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from enum import Enum
from pathlib import Path


class UserRole(Enum):
    """Роли пользователей в системе."""
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class User:
    """Класс пользователя системы."""
    
    def __init__(self,
                 username: str,
                 email: str,
                 password_hash: str,
                 role: UserRole = UserRole.VIEWER,
                 first_name: str = "",
                 last_name: str = "",
                 created_at: Optional[datetime] = None):
        self.id = str(uuid4())
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self.first_name = first_name
        self.last_name = last_name
        self.created_at = created_at or datetime.now()
        self.last_login = None
        self.is_active = True
        self.preferences = {}
        self.access_projects = []
        
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Преобразует пользователя в словарь."""
        result = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "preferences": self.preferences,
            "access_projects": self.access_projects
        }
        
        if include_sensitive:
            result["password_hash"] = self.password_hash
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Создает пользователя из словаря."""
        user = cls(
            username=data.get("username", ""),
            email=data.get("email", ""),
            password_hash=data.get("password_hash", ""),
            role=UserRole(data.get("role", "viewer")),
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", "")
        )
        
        user.id = data.get("id", str(uuid4()))
        user.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        user.last_login = datetime.fromisoformat(data.get("last_login")) if data.get("last_login") else None
        user.is_active = data.get("is_active", True)
        user.preferences = data.get("preferences", {})
        user.access_projects = data.get("access_projects", [])
        
        return user
    
    def get_full_name(self) -> str:
        """Возвращает полное имя пользователя."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        else:
            return self.username


class Permission:
    """Класс разрешения для доступа к ресурсам."""
    
    def __init__(self,
                 resource_type: str,
                 resource_id: str,
                 user_id: str,
                 can_view: bool = True,
                 can_edit: bool = False,
                 can_delete: bool = False,
                 can_share: bool = False,
                 created_at: Optional[datetime] = None):
        self.id = str(uuid4())
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.user_id = user_id
        self.can_view = can_view
        self.can_edit = can_edit
        self.can_delete = can_delete
        self.can_share = can_share
        self.created_at = created_at or datetime.now()
        self.updated_at = self.created_at
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует разрешение в словарь."""
        return {
            "id": self.id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "user_id": self.user_id,
            "can_view": self.can_view,
            "can_edit": self.can_edit,
            "can_delete": self.can_delete,
            "can_share": self.can_share,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Permission':
        """Создает разрешение из словаря."""
        permission = cls(
            resource_type=data.get("resource_type", ""),
            resource_id=data.get("resource_id", ""),
            user_id=data.get("user_id", ""),
            can_view=data.get("can_view", True),
            can_edit=data.get("can_edit", False),
            can_delete=data.get("can_delete", False),
            can_share=data.get("can_share", False)
        )
        
        permission.id = data.get("id", str(uuid4()))
        permission.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        permission.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        
        return permission


class Session:
    """Класс пользовательской сессии."""
    
    def __init__(self,
                 user_id: str,
                 token: str,
                 expires_at: datetime,
                 created_at: Optional[datetime] = None,
                 ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None):
        self.id = str(uuid4())
        self.user_id = user_id
        self.token = token
        self.expires_at = expires_at
        self.created_at = created_at or datetime.now()
        self.last_activity = self.created_at
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.is_active = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует сессию в словарь."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "token": self.token,
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Создает сессию из словаря."""
        session = cls(
            user_id=data.get("user_id", ""),
            token=data.get("token", ""),
            expires_at=datetime.fromisoformat(data.get("expires_at", datetime.now().isoformat())),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent")
        )
        
        session.id = data.get("id", str(uuid4()))
        session.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        session.last_activity = datetime.fromisoformat(data.get("last_activity", datetime.now().isoformat()))
        session.is_active = data.get("is_active", True)
        
        return session
    
    def is_expired(self) -> bool:
        """Проверяет, истекла ли сессия."""
        return datetime.now() > self.expires_at


class UserManager:
    """Менеджер пользователей для управления пользователями и сессиями."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("./data/users")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.users: Dict[str, User] = {}
        self.users_by_username: Dict[str, str] = {}  # username -> user_id
        self.users_by_email: Dict[str, str] = {}     # email -> user_id
        self.permissions: Dict[str, Permission] = {}
        self.sessions: Dict[str, Session] = {}       # token -> session
        self.sessions_by_user: Dict[str, List[str]] = {}  # user_id -> [token]
        
    def load_users(self):
        """Загружает пользователей из файлов."""
        users_dir = self.data_dir / "users"
        if not users_dir.exists():
            users_dir.mkdir(parents=True, exist_ok=True)
            return
            
        for user_file in users_dir.glob("*.json"):
            try:
                with open(user_file, 'r') as f:
                    user_data = json.load(f)
                user = User.from_dict(user_data)
                self.users[user.id] = user
                self.users_by_username[user.username.lower()] = user.id
                self.users_by_email[user.email.lower()] = user.id
            except Exception as e:
                logging.error(f"Failed to load user from {user_file}: {str(e)}")
                
    def load_permissions(self):
        """Загружает разрешения из файлов."""
        permissions_dir = self.data_dir / "permissions"
        if not permissions_dir.exists():
            permissions_dir.mkdir(parents=True, exist_ok=True)
            return
            
        for perm_file in permissions_dir.glob("*.json"):
            try:
                with open(perm_file, 'r') as f:
                    perm_data = json.load(f)
                permission = Permission.from_dict(perm_data)
                self.permissions[permission.id] = permission
            except Exception as e:
                logging.error(f"Failed to load permission from {perm_file}: {str(e)}")
                
    def load_sessions(self):
        """Загружает сессии из файлов."""
        sessions_dir = self.data_dir / "sessions"
        if not sessions_dir.exists():
            sessions_dir.mkdir(parents=True, exist_ok=True)
            return
            
        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                session = Session.from_dict(session_data)
                
                # Пропускаем истекшие сессии
                if session.is_expired():
                    continue
                    
                self.sessions[session.token] = session
                if session.user_id not in self.sessions_by_user:
                    self.sessions_by_user[session.user_id] = []
                self.sessions_by_user[session.user_id].append(session.token)
            except Exception as e:
                logging.error(f"Failed to load session from {session_file}: {str(e)}")
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """Хеширует пароль с использованием соли."""
        if not salt:
            salt = secrets.token_hex(16)
        
        hasher = hashlib.sha256()
        hasher.update((password + salt).encode('utf-8'))
        password_hash = hasher.hexdigest()
        
        return f"{salt}:{password_hash}", salt
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Проверяет соответствие пароля хешу."""
        if ":" not in password_hash:
            return False
            
        salt, hash_value = password_hash.split(":", 1)
        calculated_hash, _ = self._hash_password(password, salt)
        _, calculated_hash_value = calculated_hash.split(":", 1)
        
        return hash_value == calculated_hash_value
    
    def create_user(self, username: str, email: str, password: str, **kwargs) -> Optional[User]:
        """Создает нового пользователя."""
        # Проверяем уникальность username и email
        if username.lower() in self.users_by_username:
            logging.error(f"Username {username} already exists")
            return None
            
        if email.lower() in self.users_by_email:
            logging.error(f"Email {email} already exists")
            return None
            
        # Хешируем пароль
        password_hash, _ = self._hash_password(password)
        
        # Создаем пользователя
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            **kwargs
        )
        
        # Сохраняем пользователя
        self.users[user.id] = user
        self.users_by_username[user.username.lower()] = user.id
        self.users_by_email[user.email.lower()] = user.id
        self._save_user(user)
        
        return user
    
    def authenticate(self, username_or_email: str, password: str) -> Optional[User]:
        """Аутентифицирует пользователя по логину/email и паролю."""
        user_id = None
        
        # Ищем пользователя по username или email
        if "@" in username_or_email:
            user_id = self.users_by_email.get(username_or_email.lower())
        else:
            user_id = self.users_by_username.get(username_or_email.lower())
            
        if not user_id or user_id not in self.users:
            return None
            
        user = self.users[user_id]
        
        # Проверяем активность пользователя
        if not user.is_active:
            return None
            
        # Проверяем пароль
        if not self._verify_password(password, user.password_hash):
            return None
            
        # Обновляем время последнего входа
        user.last_login = datetime.now()
        self._save_user(user)
        
        return user
    
    def create_session(self, user_id: str, expires_in: int = 86400, **kwargs) -> Optional[Session]:
        """Создает новую сессию для пользователя."""
        if user_id not in self.users:
            return None
            
        # Генерируем токен
        token = secrets.token_urlsafe(32)
        
        # Создаем сессию
        session = Session(
            user_id=user_id,
            token=token,
            expires_at=datetime.now() + timedelta(seconds=expires_in),
            **kwargs
        )
        
        # Сохраняем сессию
        self.sessions[token] = session
        if user_id not in self.sessions_by_user:
            self.sessions_by_user[user_id] = []
        self.sessions_by_user[user_id].append(token)
        self._save_session(session)
        
        return session
    
    def validate_session(self, token: str) -> Optional[User]:
        """Проверяет валидность сессии и возвращает пользователя."""
        if token not in self.sessions:
            return None
            
        session = self.sessions[token]
        
        # Проверяем срок действия сессии
        if session.is_expired() or not session.is_active:
            self._invalidate_session(token)
            return None
            
        # Проверяем существование пользователя
        if session.user_id not in self.users:
            self._invalidate_session(token)
            return None
            
        user = self.users[session.user_id]
        
        # Проверяем активность пользователя
        if not user.is_active:
            self._invalidate_session(token)
            return None
            
        # Обновляем время последней активности сессии
        session.last_activity = datetime.now()
        self._save_session(session)
        
        return user
    
    def _invalidate_session(self, token: str):
        """Инвалидирует сессию."""
        if token not in self.sessions:
            return
            
        session = self.sessions[token]
        session.is_active = False
        
        user_id = session.user_id
        if user_id in self.sessions_by_user and token in self.sessions_by_user[user_id]:
            self.sessions_by_user[user_id].remove(token)
            
        self._save_session(session)
    
    def create_permission(self, resource_type: str, resource_id: str, user_id: str, **kwargs) -> Optional[Permission]:
        """Создает новое разрешение для пользователя."""
        if user_id not in self.users:
            return None
            
        # Создаем разрешение
        permission = Permission(
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            **kwargs
        )
        
        # Сохраняем разрешение
        self.permissions[permission.id] = permission
        self._save_permission(permission)
        
        return permission
    
    def get_user_permissions(self, user_id: str, resource_type: Optional[str] = None) -> List[Permission]:
        """Возвращает разрешения пользователя."""
        if user_id not in self.users:
            return []
            
        return [
            perm for perm in self.permissions.values()
            if perm.user_id == user_id and (resource_type is None or perm.resource_type == resource_type)
        ]
    
    def has_permission(self, user_id: str, resource_type: str, resource_id: str, 
                       view: bool = False, edit: bool = False, delete: bool = False, share: bool = False) -> bool:
        """Проверяет наличие разрешений у пользователя."""
        if user_id not in self.users:
            return False
            
        # Администраторы имеют полный доступ
        user = self.users[user_id]
        if user.role == UserRole.ADMIN:
            return True
            
        for perm in self.permissions.values():
            if perm.user_id == user_id and perm.resource_type == resource_type and perm.resource_id == resource_id:
                if view and not perm.can_view:
                    return False
                if edit and not perm.can_edit:
                    return False
                if delete and not perm.can_delete:
                    return False
                if share and not perm.can_share:
                    return False
                return True
                
        return False
    
    def _save_user(self, user: User):
        """Сохраняет пользователя в файл."""
        users_dir = self.data_dir / "users"
        users_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = users_dir / f"{user.id}.json"
        with open(file_path, 'w') as f:
            json.dump(user.to_dict(include_sensitive=True), f, indent=2)
    
    def _save_permission(self, permission: Permission):
        """Сохраняет разрешение в файл."""
        permissions_dir = self.data_dir / "permissions"
        permissions_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = permissions_dir / f"{permission.id}.json"
        with open(file_path, 'w') as f:
            json.dump(permission.to_dict(), f, indent=2)
    
    def _save_session(self, session: Session):
        """Сохраняет сессию в файл."""
        sessions_dir = self.data_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = sessions_dir / f"{session.id}.json"
        with open(file_path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Возвращает пользователя по ID."""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Возвращает пользователя по имени пользователя."""
        user_id = self.users_by_username.get(username.lower())
        if user_id:
            return self.users.get(user_id)
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Возвращает пользователя по email."""
        user_id = self.users_by_email.get(email.lower())
        if user_id:
            return self.users.get(user_id)
        return None
    
    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Обновляет данные пользователя."""
        if user_id not in self.users:
            return None
            
        user = self.users[user_id]
        
        # Проверяем изменение username
        if "username" in kwargs and kwargs["username"].lower() != user.username.lower():
            new_username = kwargs["username"].lower()
            if new_username in self.users_by_username:
                logging.error(f"Username {kwargs['username']} already exists")
                return None
            del self.users_by_username[user.username.lower()]
            self.users_by_username[new_username] = user_id
            
        # Проверяем изменение email
        if "email" in kwargs and kwargs["email"].lower() != user.email.lower():
            new_email = kwargs["email"].lower()
            if new_email in self.users_by_email:
                logging.error(f"Email {kwargs['email']} already exists")
                return None
            del self.users_by_email[user.email.lower()]
            self.users_by_email[new_email] = user_id
            
        # Проверяем изменение пароля
        if "password" in kwargs:
            password_hash, _ = self._hash_password(kwargs["password"])
            kwargs["password_hash"] = password_hash
            del kwargs["password"]
            
        # Обновляем данные пользователя
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
                
        self._save_user(user)
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """Удаляет пользователя."""
        if user_id not in self.users:
            return False
            
        user = self.users[user_id]
        
        # Удаляем пользователя из индексов
        del self.users_by_username[user.username.lower()]
        del self.users_by_email[user.email.lower()]
        
        # Удаляем сессии пользователя
        if user_id in self.sessions_by_user:
            for token in self.sessions_by_user[user_id]:
                self._invalidate_session(token)
            del self.sessions_by_user[user_id]
            
        # Удаляем разрешения пользователя
        permissions_to_delete = [perm.id for perm in self.permissions.values() if perm.user_id == user_id]
        for perm_id in permissions_to_delete:
            del self.permissions[perm_id]
            perm_file = self.data_dir / "permissions" / f"{perm_id}.json"
            if perm_file.exists():
                perm_file.unlink()
                
        # Удаляем пользователя
        del self.users[user_id]
        user_file = self.data_dir / "users" / f"{user_id}.json"
        if user_file.exists():
            user_file.unlink()
            
        return True


# Функция для создания экземпляра UserManager
def create_user_manager(data_dir: Optional[str] = None) -> UserManager:
    """Создает экземпляр менеджера пользователей."""
    manager = UserManager(data_dir)
    manager.load_users()
    manager.load_permissions()
    manager.load_sessions()
    return manager
