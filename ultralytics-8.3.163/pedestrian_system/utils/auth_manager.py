from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable

from .paths import writable_path


DEFAULT_PERMISSIONS = [
    ("can_switch_model", "允许用户切换检测模型"),
    ("can_edit_roi", "允许用户绘制/编辑ROI区域"),
    ("can_edit_line", "允许用户绘制/编辑计数线"),
    ("can_export_data", "允许用户导出统计结果"),
    ("can_manage_users", "允许用户管理员工账号"),
    ("can_reset_password", "允许重置他人密码"),
    ("can_clear_logs", "允许清除操作日志"),
]

DEFAULT_EMPLOYEE_PERMISSIONS = {
    "can_switch_model": False,
    "can_edit_roi": False,
    "can_edit_line": False,
    "can_export_data": False,
    "can_manage_users": False,
    "can_reset_password": False,
    "can_clear_logs": False,
}

DEFAULT_ADMIN_PERMISSIONS = {
    "can_switch_model": True,
    "can_edit_roi": True,
    "can_edit_line": True,
    "can_export_data": True,
    "can_manage_users": True,
    "can_reset_password": True,
    "can_clear_logs": True,
}


def check_permission(permission_name: str):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "auth_manager") and self.auth_manager is not None:
                current_user = getattr(self, "current_user", None)
                if current_user and current_user.get("id"):
                    if not self.auth_manager.has_permission(current_user["id"], permission_name):
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.warning(self, "权限不足", f"您没有执行此操作的权限（{permission_name}）。")
                        return None
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class AuthManager:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else writable_path("outputs/users.db")
        if not self.db_path.is_absolute():
            self.db_path = writable_path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._init_db()
        self._seed_permissions()
        self._seed_role_permissions()

    def _init_db(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                fullname TEXT DEFAULT '',
                role TEXT NOT NULL DEFAULT '员工',
                status TEXT NOT NULL DEFAULT '已启用',
                phone TEXT DEFAULT '',
                email TEXT DEFAULT '',
                permissions_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                last_login TEXT
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                permission_name TEXT NOT NULL UNIQUE,
                description TEXT DEFAULT ''
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS role_permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                permission_id INTEGER NOT NULL,
                FOREIGN KEY(permission_id) REFERENCES permissions(id),
                UNIQUE(role, permission_id)
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS operation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id INTEGER,
                operator_name TEXT DEFAULT '',
                target_id INTEGER,
                target_name TEXT DEFAULT '',
                action_type TEXT NOT NULL,
                description TEXT DEFAULT '',
                result TEXT DEFAULT '成功',
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        self._ensure_column("users", "fullname", "TEXT DEFAULT ''")
        self._ensure_column("users", "status", "TEXT NOT NULL DEFAULT '已启用'")
        self._ensure_column("users", "phone", "TEXT DEFAULT ''")
        self._ensure_column("users", "email", "TEXT DEFAULT ''")
        self._ensure_column("users", "permissions_json", "TEXT DEFAULT '{}'")
        self._ensure_column("users", "last_login", "TEXT")
        self.conn.commit()

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in self.cursor.fetchall()}
        if column_name not in existing_columns:
            self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def _seed_permissions(self) -> None:
        for name, desc in DEFAULT_PERMISSIONS:
            self.cursor.execute(
                "INSERT OR IGNORE INTO permissions (permission_name, description) VALUES (?, ?)",
                (name, desc),
            )
        self.conn.commit()

    def _seed_role_permissions(self) -> None:
        self.cursor.execute("SELECT id, permission_name FROM permissions")
        perm_map = {row["permission_name"]: row["id"] for row in self.cursor.fetchall()}

        for perm_name, perm_id in perm_map.items():
            admin_enabled = DEFAULT_ADMIN_PERMISSIONS.get(perm_name, True)
            employee_enabled = DEFAULT_EMPLOYEE_PERMISSIONS.get(perm_name, False)
            if admin_enabled:
                self.cursor.execute(
                    "INSERT OR IGNORE INTO role_permissions (role, permission_id) VALUES (?, ?)",
                    ("管理员", perm_id),
                )
            if employee_enabled:
                self.cursor.execute(
                    "INSERT OR IGNORE INTO role_permissions (role, permission_id) VALUES (?, ?)",
                    ("员工", perm_id),
                )
        self.conn.commit()

    def _normalize_role(self, role: str | None) -> str:
        normalized = (role or "").strip()
        if normalized not in {"员工", "管理员"}:
            raise ValueError("角色必须是员工或管理员")
        return normalized

    def _hash_password(self, password: str, salt: str | None = None) -> tuple[str, str]:
        salt_value = salt or secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_value.encode("utf-8"),
            120000,
        ).hex()
        return salt_value, password_hash

    def _get_default_permissions_json(self, role: str) -> str:
        if role == "管理员":
            return json.dumps(DEFAULT_ADMIN_PERMISSIONS, ensure_ascii=False)
        return json.dumps(DEFAULT_EMPLOYEE_PERMISSIONS, ensure_ascii=False)

    def register_user(self, username: str, password: str, role: str, fullname: str = "", phone: str = "", email: str = "") -> tuple[bool, str]:
        username = (username or "").strip()
        password = password or ""
        try:
            role = self._normalize_role(role)
        except ValueError as exc:
            return False, str(exc)

        if not username or not password:
            return False, "用户名和密码不能为空"

        self.cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if self.cursor.fetchone() is not None:
            return False, "该账号已存在"

        salt, password_hash = self._hash_password(password)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        permissions_json = self._get_default_permissions_json(role)
        self.cursor.execute(
            """
            INSERT INTO users (username, password_hash, password_salt, fullname, role, status, phone, email, permissions_json, created_at)
            VALUES (?, ?, ?, ?, ?, '已启用', ?, ?, ?, ?)
            """,
            (username, password_hash, salt, fullname, role, phone, email, permissions_json, created_at),
        )
        self.conn.commit()
        return True, "注册成功"

    def login(self, username: str, password: str, role: str) -> tuple[bool, str, dict | None]:
        username = (username or "").strip()
        password = password or ""
        try:
            role = self._normalize_role(role)
        except ValueError as exc:
            return False, str(exc), None

        if not username or not password:
            return False, "用户名和密码不能为空", None

        self.cursor.execute(
            "SELECT id, username, password_hash, password_salt, role, status, permissions_json FROM users WHERE username = ?",
            (username,),
        )
        user_row = self.cursor.fetchone()
        if user_row is None:
            return False, "账号不存在", None

        if user_row["status"] == "已禁用":
            return False, "该账号已被禁用，请联系管理员", None

        if user_row["role"] != role:
            return False, "账号身份与所选角色不匹配", None

        _, password_hash = self._hash_password(password, user_row["password_salt"])
        if not hmac.compare_digest(password_hash, user_row["password_hash"]):
            return False, "密码错误", None

        last_login = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (last_login, user_row["id"]))
        self.conn.commit()

        permissions = self.get_user_permissions(user_row["id"])

        return True, "登录成功", {
            "id": user_row["id"],
            "username": user_row["username"],
            "role": user_row["role"],
            "status": user_row["status"],
            "permissions": permissions,
            "last_login": last_login,
        }

    def get_user_profile(self, username: str) -> dict | None:
        username = (username or "").strip()
        if not username:
            return None

        self.cursor.execute(
            "SELECT id, username, fullname, role, status, phone, email, created_at, last_login FROM users WHERE username = ?",
            (username,),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "fullname": row["fullname"] or row["username"],
            "role": row["role"],
            "status": row["status"],
            "phone": row["phone"] or "-",
            "email": row["email"] or "-",
            "created_at": row["created_at"],
            "last_login": row["last_login"] or "-",
        }

    def change_password(self, username: str, old_password: str, new_password: str) -> tuple[bool, str]:
        username = (username or "").strip()
        old_password = old_password or ""
        new_password = new_password or ""

        if not username or not old_password or not new_password:
            return False, "用户名、旧密码和新密码不能为空"
        if len(new_password) < 6:
            return False, "新密码长度至少为 6 位"

        self.cursor.execute(
            "SELECT id, password_hash, password_salt FROM users WHERE username = ?",
            (username,),
        )
        row = self.cursor.fetchone()
        if row is None:
            return False, "账号不存在"

        _, old_hash = self._hash_password(old_password, row["password_salt"])
        if not hmac.compare_digest(old_hash, row["password_hash"]):
            return False, "旧密码不正确"

        new_salt, new_hash = self._hash_password(new_password)
        self.cursor.execute(
            "UPDATE users SET password_hash = ?, password_salt = ? WHERE id = ?",
            (new_hash, new_salt, row["id"]),
        )
        self.conn.commit()
        return True, "密码修改成功"

    def get_all_permissions(self) -> list[dict]:
        self.cursor.execute("SELECT id, permission_name, description FROM permissions ORDER BY id")
        return [
            {"id": row["id"], "permission_name": row["permission_name"], "description": row["description"]}
            for row in self.cursor.fetchall()
        ]

    def get_user_permissions(self, user_id: int) -> dict:
        self.cursor.execute("SELECT role, permissions_json FROM users WHERE id = ?", (user_id,))
        user_row = self.cursor.fetchone()
        if not user_row:
            return {}

        role = user_row["role"]
        try:
            custom_perms = json.loads(user_row["permissions_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            custom_perms = {}

        self.cursor.execute(
            """
            SELECT p.permission_name FROM permissions p
            INNER JOIN role_permissions rp ON p.id = rp.permission_id
            WHERE rp.role = ?
            """,
            (role,),
        )
        role_perms = {row["permission_name"]: True for row in self.cursor.fetchall()}

        if role == "管理员":
            defaults = dict(DEFAULT_ADMIN_PERMISSIONS)
        else:
            defaults = dict(DEFAULT_EMPLOYEE_PERMISSIONS)

        merged = {}
        for perm in DEFAULT_PERMISSIONS:
            name = perm[0]
            if name in custom_perms:
                merged[name] = bool(custom_perms[name])
            elif name in role_perms:
                merged[name] = True
            else:
                merged[name] = defaults.get(name, False)

        return merged

    def has_permission(self, user_id: int, permission_name: str) -> bool:
        perms = self.get_user_permissions(user_id)
        return perms.get(permission_name, False)

    def update_user_permissions(self, user_id: int, permissions: dict) -> bool:
        try:
            perms_json = json.dumps(permissions, ensure_ascii=False)
            self.cursor.execute(
                "UPDATE users SET permissions_json = ? WHERE id = ?",
                (perms_json, user_id),
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def get_all_users(self) -> list[dict]:
        self.cursor.execute(
            "SELECT id, username, fullname, role, status, phone, email, created_at, last_login, permissions_json FROM users ORDER BY id"
        )
        users = []
        for row in self.cursor.fetchall():
            try:
                custom_perms = json.loads(row["permissions_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                custom_perms = {}
            permission_summary = self._build_permission_summary(row["id"])
            users.append({
                "id": row["id"],
                "username": row["username"],
                "fullname": row["fullname"] or row["username"],
                "role": row["role"],
                "status": row["status"],
                "phone": row["phone"] or "-",
                "email": row["email"] or "-",
                "created_at": row["created_at"],
                "last_login": row["last_login"] or "-",
                "permissions_json": custom_perms,
                "permission_summary": permission_summary,
            })
        return users

    def _build_permission_summary(self, user_id: int) -> str:
        perms = self.get_user_permissions(user_id)
        label_map = {
            "can_switch_model": "切换模型",
            "can_edit_roi": "编辑ROI",
            "can_edit_line": "编辑计数线",
            "can_export_data": "导出数据",
            "can_manage_users": "管理员工",
            "can_reset_password": "重置密码",
            "can_clear_logs": "清除日志",
        }
        enabled = [label_map[k] for k, v in perms.items() if v and k in label_map]
        if not enabled:
            return "基础权限"
        return " | ".join(enabled)

    def create_user(self, emp_data: dict) -> dict:
        username = (emp_data.get("username") or "").strip()
        password = emp_data.get("password") or "123456"
        fullname = emp_data.get("fullname") or username
        role = self._normalize_role(emp_data.get("role", "员工"))
        phone = emp_data.get("phone") or ""
        email = emp_data.get("email") or ""

        if not username:
            raise ValueError("用户名不能为空")

        self.cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if self.cursor.fetchone() is not None:
            raise ValueError("该账号已存在")

        salt, password_hash = self._hash_password(password)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        permissions_json = self._get_default_permissions_json(role)
        self.cursor.execute(
            """
            INSERT INTO users (username, password_hash, password_salt, fullname, role, status, phone, email, permissions_json, created_at)
            VALUES (?, ?, ?, ?, ?, '已启用', ?, ?, ?, ?)
            """,
            (username, password_hash, salt, fullname, role, phone, email, permissions_json, created_at),
        )
        self.conn.commit()
        new_id = self.cursor.lastrowid
        return {
            "id": new_id,
            "username": username,
            "fullname": fullname,
            "role": role,
            "status": "已启用",
            "phone": phone,
            "email": email,
            "created_at": created_at,
            "last_login": "-",
        }

    def update_user(self, user_id: int, emp_data: dict) -> dict:
        updates = {}
        params = []
        if "fullname" in emp_data:
            updates.append("fullname = ?")
            params.append(emp_data["fullname"])
        if "role" in emp_data:
            updates.append("role = ?")
            params.append(self._normalize_role(emp_data["role"]))
        if "status" in emp_data:
            status = emp_data["status"]
            if status not in ("已启用", "已禁用"):
                status = "已启用"
            updates.append("status = ?")
            params.append(status)
        if "phone" in emp_data:
            updates.append("phone = ?")
            params.append(emp_data["phone"])
        if "email" in emp_data:
            updates.append("email = ?")
            params.append(emp_data["email"])
        if "password" in emp_data:
            salt, password_hash = self._hash_password(emp_data["password"])
            updates.append("password_hash = ?")
            params.append(password_hash)
            updates.append("password_salt = ?")
            params.append(salt)

        if not updates:
            raise ValueError("没有需要更新的字段")

        params.append(user_id)
        self.cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
        self.conn.commit()

        self.cursor.execute(
            "SELECT id, username, fullname, role, status, phone, email, created_at, last_login FROM users WHERE id = ?",
            (user_id,),
        )
        row = self.cursor.fetchone()
        if row is None:
            raise ValueError("用户不存在")
        return {
            "id": row["id"],
            "username": row["username"],
            "fullname": row["fullname"] or row["username"],
            "role": row["role"],
            "status": row["status"],
            "phone": row["phone"] or "-",
            "email": row["email"] or "-",
            "created_at": row["created_at"],
            "last_login": row["last_login"] or "-",
        }

    def delete_user(self, user_id: int) -> None:
        self.cursor.execute("SELECT role, username FROM users WHERE id = ?", (user_id,))
        row = self.cursor.fetchone()
        if row is None:
            raise ValueError("用户不存在")
        if row["role"] == "管理员" and row["username"] == "admin":
            raise ValueError("超级管理员无法删除")
        self.cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()

    def reset_user_password(self, user_id: int, new_password: str = "123456") -> None:
        self.cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if self.cursor.fetchone() is None:
            raise ValueError("用户不存在")
        salt, password_hash = self._hash_password(new_password)
        self.cursor.execute(
            "UPDATE users SET password_hash = ?, password_salt = ? WHERE id = ?",
            (password_hash, salt, user_id),
        )
        self.conn.commit()

    def toggle_user_status(self, user_id: int, new_status: str) -> None:
        if new_status not in ("已启用", "已禁用"):
            raise ValueError("无效的状态值")
        self.cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if self.cursor.fetchone() is None:
            raise ValueError("用户不存在")
        self.cursor.execute("UPDATE users SET status = ? WHERE id = ?", (new_status, user_id))
        self.conn.commit()

    def get_permission_labels(self) -> dict:
        return {
            "can_switch_model": "切换模型",
            "can_edit_roi": "编辑ROI",
            "can_edit_line": "编辑计数线",
            "can_export_data": "导出数据",
            "can_manage_users": "管理员工",
            "can_reset_password": "重置密码",
            "can_clear_logs": "清除日志",
        }

    def add_operation_log(self, operator_id, operator_name, action_type, target_id=None, target_name="",
                          description="", result="成功"):
        from datetime import datetime
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO operation_logs (operator_id, operator_name, target_id, target_name, action_type, description, result, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (operator_id, operator_name, target_id, target_name, action_type, description, result, created_at),
        )
        self.conn.commit()

    def get_operation_logs(self, limit=50, offset=0):
        self.cursor.execute(
            "SELECT * FROM operation_logs ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [
            {
                "id": row["id"],
                "operator_id": row["operator_id"],
                "operator_name": row["operator_name"],
                "target_id": row["target_id"],
                "target_name": row["target_name"],
                "action_type": row["action_type"],
                "description": row["description"],
                "result": row["result"],
                "created_at": row["created_at"],
            }
            for row in self.cursor.fetchall()
        ]

    def clear_operation_logs(self):
        self.cursor.execute("DELETE FROM operation_logs")
        self.conn.commit()

    def close(self) -> None:
        try:
            self.cursor.close()
            self.conn.close()
        except Exception:
            pass
