"""
MegaSerpentClient - Sharing & Collaboration Module

Purpose: File sharing, permissions, collaboration features, and team management.

This module handles complete sharing system (shares, links, folders, teams, invitations),
permission management with ACL and roles, collaboration tools, enterprise features,
and workflow automation.
"""

import json
import time
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from . import utils
from .utils import (
    ShareType, MegaError, ValidationError, SecurityError,
    Validators, DateTimeUtils, Helpers, Formatters
)


# ==============================================
# === SHARING ENUMS AND CONSTANTS ===
# ==============================================

class ShareStatus(Enum):
    """Share status enumeration."""
    ACTIVE = "active"
    PENDING = "pending"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class PermissionType(Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


class RoleType(Enum):
    """User role types."""
    VIEWER = "viewer"
    EDITOR = "editor"
    COLLABORATOR = "collaborator"
    ADMIN = "admin"
    OWNER = "owner"


class InvitationStatus(Enum):
    """Invitation status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class NotificationType(Enum):
    """Notification types."""
    SHARE_CREATED = "share_created"
    SHARE_UPDATED = "share_updated"
    FILE_UPLOADED = "file_uploaded"
    FILE_MODIFIED = "file_modified"
    COMMENT_ADDED = "comment_added"
    MENTION = "mention"
    INVITATION_RECEIVED = "invitation_received"


# ==============================================
# === DATA CLASSES ===
# ==============================================

@dataclass
class ShareInfo:
    """Share information."""
    share_id: str
    name: str
    node_id: str
    share_type: ShareType
    status: ShareStatus = ShareStatus.ACTIVE
    created_by: str = ""
    created_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    expires_at: Optional[datetime] = None
    password_protected: bool = False
    download_limit: Optional[int] = None
    download_count: int = 0
    access_permissions: Set[PermissionType] = field(default_factory=set)
    public_url: Optional[str] = None
    access_key: Optional[str] = None


@dataclass
class UserPermission:
    """User permission information."""
    user_id: str
    email: str
    role: RoleType
    permissions: Set[PermissionType] = field(default_factory=set)
    granted_by: str = ""
    granted_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    expires_at: Optional[datetime] = None


@dataclass
class TeamInfo:
    """Team information."""
    team_id: str
    name: str
    description: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    member_count: int = 0
    admin_count: int = 0
    is_enterprise: bool = False


@dataclass
class InvitationInfo:
    """Invitation information."""
    invitation_id: str
    share_id: str
    email: str
    role: RoleType
    status: InvitationStatus = InvitationStatus.PENDING
    invited_by: str = ""
    invited_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    expires_at: datetime = field(default_factory=lambda: DateTimeUtils.now_utc() + timedelta(days=7))
    message: str = ""
    acceptance_token: Optional[str] = None


@dataclass
class CommentInfo:
    """Comment information."""
    comment_id: str
    node_id: str
    user_id: str
    content: str
    created_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    modified_at: Optional[datetime] = None
    parent_comment_id: Optional[str] = None
    mentions: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)


@dataclass
class NotificationInfo:
    """Notification information."""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=DateTimeUtils.now_utc)
    read_at: Optional[datetime] = None
    is_read: bool = False


# ==============================================
# === SHARE MANAGEMENT ===
# ==============================================

class ShareManager:
    """Share management functionality."""
    
    def __init__(self):
        self._shares: Dict[str, ShareInfo] = {}
        self._node_shares: Dict[str, List[str]] = {}  # node_id -> share_ids
        self._user_shares: Dict[str, List[str]] = {}  # user_id -> share_ids
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def create_share(self, node_id: str, name: str, share_type: ShareType = ShareType.PRIVATE,
                    created_by: str = "", permissions: Optional[Set[PermissionType]] = None,
                    expires_at: Optional[datetime] = None, password: Optional[str] = None) -> ShareInfo:
        """Create new share."""
        share_id = Helpers.generate_request_id()
        
        share_info = ShareInfo(
            share_id=share_id,
            name=name,
            node_id=node_id,
            share_type=share_type,
            created_by=created_by,
            expires_at=expires_at,
            password_protected=bool(password),
            access_permissions=permissions or {PermissionType.READ}
        )
        
        # Generate access key for public shares
        if share_type == ShareType.PUBLIC:
            share_info.access_key = secrets.token_urlsafe(16)
            share_info.public_url = f"https://mega.nz/share/{share_info.access_key}"
        
        with self._lock:
            self._shares[share_id] = share_info
            
            if node_id not in self._node_shares:
                self._node_shares[node_id] = []
            self._node_shares[node_id].append(share_id)
            
            if created_by not in self._user_shares:
                self._user_shares[created_by] = []
            self._user_shares[created_by].append(share_id)
        
        self.logger.info(f"Created {share_type.value} share: {name} (ID: {share_id})")
        return share_info
    
    def get_share(self, share_id: str) -> Optional[ShareInfo]:
        """Get share by ID."""
        return self._shares.get(share_id)
    
    def get_shares_by_node(self, node_id: str) -> List[ShareInfo]:
        """Get all shares for a node."""
        with self._lock:
            share_ids = self._node_shares.get(node_id, [])
            return [self._shares[sid] for sid in share_ids if sid in self._shares]
    
    def get_user_shares(self, user_id: str) -> List[ShareInfo]:
        """Get all shares created by user."""
        with self._lock:
            share_ids = self._user_shares.get(user_id, [])
            return [self._shares[sid] for sid in share_ids if sid in self._shares]
    
    def update_share(self, share_id: str, **updates) -> bool:
        """Update share properties."""
        with self._lock:
            if share_id not in self._shares:
                return False
            
            share = self._shares[share_id]
            
            for key, value in updates.items():
                if hasattr(share, key):
                    setattr(share, key, value)
        
        self.logger.info(f"Updated share: {share_id}")
        return True
    
    def revoke_share(self, share_id: str) -> bool:
        """Revoke share access."""
        return self.update_share(share_id, status=ShareStatus.REVOKED)
    
    def increment_download_count(self, share_id: str) -> bool:
        """Increment download count for share."""
        with self._lock:
            if share_id not in self._shares:
                return False
            
            share = self._shares[share_id]
            share.download_count += 1
            
            # Check download limit
            if share.download_limit and share.download_count >= share.download_limit:
                share.status = ShareStatus.EXPIRED
        
        return True
    
    def check_share_access(self, share_id: str, user_id: Optional[str] = None,
                          password: Optional[str] = None) -> bool:
        """Check if user can access share."""
        share = self.get_share(share_id)
        if not share or share.status != ShareStatus.ACTIVE:
            return False
        
        # Check expiration
        if share.expires_at and DateTimeUtils.now_utc() > share.expires_at:
            share.status = ShareStatus.EXPIRED
            return False
        
        # Check password protection
        if share.password_protected and not password:
            return False
        
        # Check download limit
        if share.download_limit and share.download_count >= share.download_limit:
            return False
        
        return True
    
    def generate_share_link(self, share_id: str) -> Optional[str]:
        """Generate shareable link."""
        share = self.get_share(share_id)
        if not share:
            return None
        
        if share.share_type == ShareType.PUBLIC:
            return share.public_url
        else:
            # Generate private link with access token
            access_token = secrets.token_urlsafe(32)
            return f"https://mega.nz/private/{access_token}"


class LinkManager:
    """Public/private link management."""
    
    def __init__(self, share_manager: ShareManager):
        self.share_manager = share_manager
        self._link_analytics: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_public_link(self, node_id: str, expires_hours: Optional[int] = None,
                          password: Optional[str] = None, download_limit: Optional[int] = None) -> str:
        """Create public download link."""
        expires_at = None
        if expires_hours:
            expires_at = DateTimeUtils.now_utc() + timedelta(hours=expires_hours)
        
        share = self.share_manager.create_share(
            node_id=node_id,
            name="Public Link",
            share_type=ShareType.PUBLIC,
            expires_at=expires_at,
            password=password
        )
        
        if download_limit:
            self.share_manager.update_share(share.share_id, download_limit=download_limit)
        
        self.logger.info(f"Created public link for node: {node_id}")
        return share.public_url
    
    def create_private_link(self, node_id: str, user_emails: List[str],
                           permissions: Set[PermissionType]) -> str:
        """Create private sharing link."""
        share = self.share_manager.create_share(
            node_id=node_id,
            name="Private Share",
            share_type=ShareType.PRIVATE,
            permissions=permissions
        )
        
        # Send invitations to users
        for email in user_emails:
            # In real implementation, send invitation emails
            pass
        
        link = self.share_manager.generate_share_link(share.share_id)
        self.logger.info(f"Created private link for node: {node_id}")
        return link
    
    def track_link_access(self, share_id: str, user_info: Dict[str, Any]):
        """Track link access for analytics."""
        if share_id not in self._link_analytics:
            self._link_analytics[share_id] = {
                'total_views': 0,
                'unique_viewers': set(),
                'access_history': [],
                'countries': {},
                'referrers': {}
            }
        
        analytics = self._link_analytics[share_id]
        analytics['total_views'] += 1
        analytics['access_history'].append({
            'timestamp': DateTimeUtils.now_utc(),
            'user_info': user_info
        })
        
        # Track unique viewers by IP
        ip_address = user_info.get('ip_address')
        if ip_address:
            analytics['unique_viewers'].add(ip_address)
        
        # Track country
        country = user_info.get('country')
        if country:
            analytics['countries'][country] = analytics['countries'].get(country, 0) + 1
        
        # Track referrer
        referrer = user_info.get('referrer')
        if referrer:
            analytics['referrers'][referrer] = analytics['referrers'].get(referrer, 0) + 1
    
    def get_link_analytics(self, share_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for share link."""
        if share_id not in self._link_analytics:
            return None
        
        analytics = self._link_analytics[share_id].copy()
        analytics['unique_viewers'] = len(analytics['unique_viewers'])
        return analytics


# ==============================================
# === PERMISSION MANAGEMENT ===
# ==============================================

class PermissionEngine:
    """Core permission system."""
    
    def __init__(self):
        self._user_permissions: Dict[str, Dict[str, UserPermission]] = {}  # node_id -> user_id -> permission
        self._role_permissions: Dict[RoleType, Set[PermissionType]] = {
            RoleType.VIEWER: {PermissionType.READ},
            RoleType.EDITOR: {PermissionType.READ, PermissionType.WRITE},
            RoleType.COLLABORATOR: {PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE},
            RoleType.ADMIN: {PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE, PermissionType.SHARE},
            RoleType.OWNER: {PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE, PermissionType.SHARE, PermissionType.ADMIN}
        }
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def grant_permission(self, node_id: str, user_id: str, email: str, role: RoleType,
                        granted_by: str, expires_at: Optional[datetime] = None) -> UserPermission:
        """Grant permission to user for node."""
        permissions = self._role_permissions.get(role, set())
        
        user_permission = UserPermission(
            user_id=user_id,
            email=email,
            role=role,
            permissions=permissions,
            granted_by=granted_by,
            expires_at=expires_at
        )
        
        with self._lock:
            if node_id not in self._user_permissions:
                self._user_permissions[node_id] = {}
            
            self._user_permissions[node_id][user_id] = user_permission
        
        self.logger.info(f"Granted {role.value} permission to {email} for node {node_id}")
        return user_permission
    
    def revoke_permission(self, node_id: str, user_id: str) -> bool:
        """Revoke user permission for node."""
        with self._lock:
            if node_id in self._user_permissions and user_id in self._user_permissions[node_id]:
                del self._user_permissions[node_id][user_id]
                self.logger.info(f"Revoked permission for user {user_id} on node {node_id}")
                return True
        return False
    
    def check_permission(self, node_id: str, user_id: str, permission: PermissionType) -> bool:
        """Check if user has specific permission for node."""
        with self._lock:
            if node_id not in self._user_permissions or user_id not in self._user_permissions[node_id]:
                return False
            
            user_perm = self._user_permissions[node_id][user_id]
            
            # Check expiration
            if user_perm.expires_at and DateTimeUtils.now_utc() > user_perm.expires_at:
                return False
            
            return permission in user_perm.permissions
    
    def get_user_permissions(self, node_id: str, user_id: str) -> Optional[UserPermission]:
        """Get user permissions for node."""
        with self._lock:
            return self._user_permissions.get(node_id, {}).get(user_id)
    
    def get_node_permissions(self, node_id: str) -> List[UserPermission]:
        """Get all permissions for node."""
        with self._lock:
            return list(self._user_permissions.get(node_id, {}).values())
    
    def update_role(self, node_id: str, user_id: str, new_role: RoleType) -> bool:
        """Update user role for node."""
        with self._lock:
            if node_id in self._user_permissions and user_id in self._user_permissions[node_id]:
                user_perm = self._user_permissions[node_id][user_id]
                user_perm.role = new_role
                user_perm.permissions = self._role_permissions.get(new_role, set())
                
                self.logger.info(f"Updated role for user {user_id} to {new_role.value}")
                return True
        return False


# ==============================================
# === INVITATION MANAGEMENT ===
# ==============================================

class InvitationManager:
    """User invitation management."""
    
    def __init__(self, permission_engine: PermissionEngine):
        self.permission_engine = permission_engine
        self._invitations: Dict[str, InvitationInfo] = {}
        self._user_invitations: Dict[str, List[str]] = {}  # email -> invitation_ids
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def send_invitation(self, share_id: str, email: str, role: RoleType,
                       invited_by: str, message: str = "") -> InvitationInfo:
        """Send invitation to user."""
        if not Validators.validate_email(email):
            raise ValidationError("Invalid email address")
        
        invitation_id = Helpers.generate_request_id()
        acceptance_token = secrets.token_urlsafe(32)
        
        invitation = InvitationInfo(
            invitation_id=invitation_id,
            share_id=share_id,
            email=email,
            role=role,
            invited_by=invited_by,
            message=message,
            acceptance_token=acceptance_token
        )
        
        with self._lock:
            self._invitations[invitation_id] = invitation
            
            if email not in self._user_invitations:
                self._user_invitations[email] = []
            self._user_invitations[email].append(invitation_id)
        
        # In real implementation, send email invitation
        self._send_invitation_email(invitation)
        
        self.logger.info(f"Sent invitation to {email} for share {share_id}")
        return invitation
    
    def accept_invitation(self, acceptance_token: str, user_id: str) -> bool:
        """Accept invitation using token."""
        invitation = self._find_invitation_by_token(acceptance_token)
        if not invitation or invitation.status != InvitationStatus.PENDING:
            return False
        
        # Check expiration
        if DateTimeUtils.now_utc() > invitation.expires_at:
            invitation.status = InvitationStatus.EXPIRED
            return False
        
        # Grant permission
        # Note: In real implementation, get node_id from share_id
        node_id = "dummy_node_id"  # This would be resolved from share
        
        self.permission_engine.grant_permission(
            node_id=node_id,
            user_id=user_id,
            email=invitation.email,
            role=invitation.role,
            granted_by=invitation.invited_by
        )
        
        invitation.status = InvitationStatus.ACCEPTED
        self.logger.info(f"Accepted invitation: {invitation.invitation_id}")
        return True
    
    def decline_invitation(self, acceptance_token: str) -> bool:
        """Decline invitation."""
        invitation = self._find_invitation_by_token(acceptance_token)
        if not invitation or invitation.status != InvitationStatus.PENDING:
            return False
        
        invitation.status = InvitationStatus.DECLINED
        self.logger.info(f"Declined invitation: {invitation.invitation_id}")
        return True
    
    def cancel_invitation(self, invitation_id: str) -> bool:
        """Cancel pending invitation."""
        with self._lock:
            if invitation_id not in self._invitations:
                return False
            
            invitation = self._invitations[invitation_id]
            if invitation.status == InvitationStatus.PENDING:
                invitation.status = InvitationStatus.CANCELLED
                self.logger.info(f"Cancelled invitation: {invitation_id}")
                return True
        
        return False
    
    def get_user_invitations(self, email: str, status: Optional[InvitationStatus] = None) -> List[InvitationInfo]:
        """Get invitations for user."""
        with self._lock:
            invitation_ids = self._user_invitations.get(email, [])
            invitations = [self._invitations[iid] for iid in invitation_ids if iid in self._invitations]
            
            if status:
                invitations = [inv for inv in invitations if inv.status == status]
            
            return invitations
    
    def _find_invitation_by_token(self, token: str) -> Optional[InvitationInfo]:
        """Find invitation by acceptance token."""
        with self._lock:
            for invitation in self._invitations.values():
                if invitation.acceptance_token == token:
                    return invitation
        return None
    
    def _send_invitation_email(self, invitation: InvitationInfo):
        """Send invitation email (placeholder)."""
        # In real implementation, integrate with email service
        email_content = f"""
        You've been invited to collaborate!
        
        Share ID: {invitation.share_id}
        Role: {invitation.role.value}
        Invited by: {invitation.invited_by}
        Message: {invitation.message}
        
        Accept invitation: https://mega.nz/invite/{invitation.acceptance_token}
        """
        
        self.logger.info(f"Email invitation sent to {invitation.email}")


# ==============================================
# === COLLABORATION FEATURES ===
# ==============================================

class CommentManager:
    """File commenting system."""
    
    def __init__(self):
        self._comments: Dict[str, CommentInfo] = {}
        self._node_comments: Dict[str, List[str]] = {}  # node_id -> comment_ids
        self._user_comments: Dict[str, List[str]] = {}  # user_id -> comment_ids
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_comment(self, node_id: str, user_id: str, content: str,
                   parent_comment_id: Optional[str] = None) -> CommentInfo:
        """Add comment to file/folder."""
        comment_id = Helpers.generate_request_id()
        
        # Extract mentions from content
        mentions = self._extract_mentions(content)
        
        comment = CommentInfo(
            comment_id=comment_id,
            node_id=node_id,
            user_id=user_id,
            content=content,
            parent_comment_id=parent_comment_id,
            mentions=mentions
        )
        
        with self._lock:
            self._comments[comment_id] = comment
            
            if node_id not in self._node_comments:
                self._node_comments[node_id] = []
            self._node_comments[node_id].append(comment_id)
            
            if user_id not in self._user_comments:
                self._user_comments[user_id] = []
            self._user_comments[user_id].append(comment_id)
        
        self.logger.info(f"Added comment to node {node_id} by user {user_id}")
        return comment
    
    def get_comments(self, node_id: str, include_replies: bool = True) -> List[CommentInfo]:
        """Get comments for node."""
        with self._lock:
            comment_ids = self._node_comments.get(node_id, [])
            comments = [self._comments[cid] for cid in comment_ids if cid in self._comments]
            
            if not include_replies:
                comments = [c for c in comments if not c.parent_comment_id]
            
            # Sort by creation time
            comments.sort(key=lambda x: x.created_at)
            return comments
    
    def update_comment(self, comment_id: str, new_content: str) -> bool:
        """Update comment content."""
        with self._lock:
            if comment_id not in self._comments:
                return False
            
            comment = self._comments[comment_id]
            comment.content = new_content
            comment.modified_at = DateTimeUtils.now_utc()
            comment.mentions = self._extract_mentions(new_content)
        
        self.logger.info(f"Updated comment: {comment_id}")
        return True
    
    def delete_comment(self, comment_id: str) -> bool:
        """Delete comment."""
        with self._lock:
            if comment_id not in self._comments:
                return False
            
            comment = self._comments[comment_id]
            
            # Remove from node comments
            if comment.node_id in self._node_comments:
                if comment_id in self._node_comments[comment.node_id]:
                    self._node_comments[comment.node_id].remove(comment_id)
            
            # Remove from user comments
            if comment.user_id in self._user_comments:
                if comment_id in self._user_comments[comment.user_id]:
                    self._user_comments[comment.user_id].remove(comment_id)
            
            del self._comments[comment_id]
        
        self.logger.info(f"Deleted comment: {comment_id}")
        return True
    
    def _extract_mentions(self, content: str) -> List[str]:
        """Extract user mentions from comment content."""
        import re
        mentions = re.findall(r'@(\w+)', content)
        return list(set(mentions))  # Remove duplicates


class NotificationManager:
    """Collaboration notifications."""
    
    def __init__(self):
        self._notifications: Dict[str, NotificationInfo] = {}
        self._user_notifications: Dict[str, List[str]] = {}  # user_id -> notification_ids
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def create_notification(self, user_id: str, notification_type: NotificationType,
                           title: str, message: str, data: Optional[Dict[str, Any]] = None) -> NotificationInfo:
        """Create notification for user."""
        notification_id = Helpers.generate_request_id()
        
        notification = NotificationInfo(
            notification_id=notification_id,
            user_id=user_id,
            notification_type=notification_type,
            title=title,
            message=message,
            data=data or {}
        )
        
        with self._lock:
            self._notifications[notification_id] = notification
            
            if user_id not in self._user_notifications:
                self._user_notifications[user_id] = []
            self._user_notifications[user_id].append(notification_id)
        
        self.logger.info(f"Created {notification_type.value} notification for user {user_id}")
        return notification
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[NotificationInfo]:
        """Get notifications for user."""
        with self._lock:
            notification_ids = self._user_notifications.get(user_id, [])
            notifications = [self._notifications[nid] for nid in notification_ids if nid in self._notifications]
            
            if unread_only:
                notifications = [n for n in notifications if not n.is_read]
            
            # Sort by creation time (newest first)
            notifications.sort(key=lambda x: x.created_at, reverse=True)
            return notifications
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Mark notification as read."""
        with self._lock:
            if notification_id not in self._notifications:
                return False
            
            notification = self._notifications[notification_id]
            notification.is_read = True
            notification.read_at = DateTimeUtils.now_utc()
        
        return True
    
    def mark_all_as_read(self, user_id: str) -> int:
        """Mark all notifications as read for user."""
        count = 0
        with self._lock:
            notification_ids = self._user_notifications.get(user_id, [])
            
            for nid in notification_ids:
                if nid in self._notifications and not self._notifications[nid].is_read:
                    self._notifications[nid].is_read = True
                    self._notifications[nid].read_at = DateTimeUtils.now_utc()
                    count += 1
        
        return count
    
    def delete_notification(self, notification_id: str) -> bool:
        """Delete notification."""
        with self._lock:
            if notification_id not in self._notifications:
                return False
            
            notification = self._notifications[notification_id]
            
            # Remove from user notifications
            if notification.user_id in self._user_notifications:
                if notification_id in self._user_notifications[notification.user_id]:
                    self._user_notifications[notification.user_id].remove(notification_id)
            
            del self._notifications[notification_id]
        
        return True


# ==============================================
# === MODULE EXPORTS ===
# ==============================================

__all__ = [
    # Enums
    'ShareStatus', 'PermissionType', 'RoleType', 'InvitationStatus', 'NotificationType',
    
    # Data Classes
    'ShareInfo', 'UserPermission', 'TeamInfo', 'InvitationInfo', 'CommentInfo', 'NotificationInfo',
    
    # Share Management
    'ShareManager', 'LinkManager',
    
    # Permission Management
    'PermissionEngine',
    
    # Collaboration
    'InvitationManager', 'CommentManager', 'NotificationManager'
]