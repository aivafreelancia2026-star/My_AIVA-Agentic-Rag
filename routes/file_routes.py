"""
routes/file_routes.py

REST API for Phase 2 — In-App File Viewing and Editing.

Endpoints:
  GET  /api/files                    — List all documents in the embeddings directory
  GET  /api/files/read               — Read / view a file's content  (?path=...)
  POST /api/files/save-text          — Save .txt / .md
  POST /api/files/save-csv           — Save .csv
  POST /api/files/save-docx          — Save .docx
  POST /api/files/save-xlsx          — Save .xlsx
  POST /api/files/reindex            — Re-index a file into FAISS
  POST /api/files/ai-edit            — Propose + apply an AI edit
  GET  /api/files/backups            — List backups for a file  (?path=...)
  POST /api/files/restore-backup     — Restore a backup
  GET  /api/files/edit-log           — View edit history
"""

import os
import time
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request
from modules.auth import require_login

file_bp = Blueprint('files', __name__, url_prefix='/api/files')
logger = logging.getLogger(__name__)


def _agent():
    """Lazy import to avoid circular imports at module load time."""
    import agents.file_editing_agent as fa
    return fa


def _config():
    from config import config as c
    return c


# ── List documents ────────────────────────────────────────────────────────────

@file_bp.route('', methods=['GET'])
@require_login
def list_files():
    """
    Return all files found inside the embeddings directory (and upload folders).
    Also checks whether each file is indexed in FAISS.
    """
    cfg = _config()
    root = Path(cfg.PROJECT_ROOT)

    # Collect candidate directories
    dirs_to_scan = []
    embed_dir = root / "embedding"
    if embed_dir.is_dir():
        dirs_to_scan.append(embed_dir)
    for upload_dir_name in ("temp_uploads", "agent_uploads"):
        ud = root / upload_dir_name
        if ud.is_dir():
            dirs_to_scan.append(ud)

    fa = _agent()
    files = []
    seen = set()

    for scan_dir in dirs_to_scan:
        for p in scan_dir.rglob("*"):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix not in fa.ALL_SUPPORTED:
                continue
            rel = str(p.relative_to(root))
            if rel in seen:
                continue
            seen.add(rel)

            stat = p.stat()
            files.append({
                "name":        p.name,
                "path":        rel,
                "suffix":      suffix,
                "size_bytes":  stat.st_size,
                "modified":    stat.st_mtime,
                "is_editable": suffix in fa.EDITABLE_TYPES,
                "is_viewable": suffix in fa.ALL_SUPPORTED,
            })

    files.sort(key=lambda f: f["modified"], reverse=True)
    return jsonify({"files": files, "total": len(files)})


# ── Read / view ───────────────────────────────────────────────────────────────

@file_bp.route('/read', methods=['GET'])
@require_login
def read_file():
    """Read (view) a file. ?path=relative/path/to/file"""
    path = request.args.get('path', '').strip()
    if not path:
        return jsonify({"error": "path parameter required"}), 400
    result = _agent().read_file_content(path)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# ── Save endpoints ────────────────────────────────────────────────────────────

@file_bp.route('/save-text', methods=['POST'])
@require_login
def save_text():
    """
    Save a .txt or .md file.
    Body: { path, content, confirmed (bool), save_as_new_version (bool) }
    """
    data = request.get_json() or {}
    path      = data.get('path', '').strip()
    content   = data.get('content', '')
    confirmed = bool(data.get('confirmed', False))
    new_ver   = bool(data.get('save_as_new_version', False))

    if not path:
        return jsonify({"error": "path required"}), 400

    result = _agent().save_text_file(path, content, confirmed=confirmed, save_as_new_version=new_ver)
    _log_edit("save_text", path, result)
    return jsonify(result)


@file_bp.route('/save-csv', methods=['POST'])
@require_login
def save_csv():
    """
    Save a .csv file.
    Body: { path, rows (list of lists), confirmed, save_as_new_version }
    """
    data = request.get_json() or {}
    path      = data.get('path', '').strip()
    rows      = data.get('rows', [])
    confirmed = bool(data.get('confirmed', False))
    new_ver   = bool(data.get('save_as_new_version', False))

    if not path:
        return jsonify({"error": "path required"}), 400

    result = _agent().save_csv_file(path, rows, confirmed=confirmed, save_as_new_version=new_ver)
    _log_edit("save_csv", path, result)
    return jsonify(result)


@file_bp.route('/save-docx', methods=['POST'])
@require_login
def save_docx():
    """
    Save a .docx file (paragraphs only).
    Body: { path, paragraphs (list of {text, style}), confirmed, save_as_new_version }
    """
    data = request.get_json() or {}
    path        = data.get('path', '').strip()
    paragraphs  = data.get('paragraphs', [])
    confirmed   = bool(data.get('confirmed', False))
    new_ver     = bool(data.get('save_as_new_version', True))

    if not path:
        return jsonify({"error": "path required"}), 400

    result = _agent().save_docx_file(path, paragraphs, confirmed=confirmed, save_as_new_version=new_ver)
    _log_edit("save_docx", path, result)
    return jsonify(result)


@file_bp.route('/save-xlsx', methods=['POST'])
@require_login
def save_xlsx():
    """
    Save one sheet of an .xlsx file.
    Body: { path, sheet_name, rows, confirmed, save_as_new_version }
    """
    data = request.get_json() or {}
    path       = data.get('path', '').strip()
    sheet_name = data.get('sheet_name', '')
    rows       = data.get('rows', [])
    confirmed  = bool(data.get('confirmed', False))
    new_ver    = bool(data.get('save_as_new_version', True))

    if not path or not sheet_name:
        return jsonify({"error": "path and sheet_name required"}), 400

    result = _agent().save_xlsx_file(
        path, sheet_name, rows, confirmed=confirmed, save_as_new_version=new_ver
    )
    _log_edit("save_xlsx", path, result)
    return jsonify(result)


# ── Re-index ──────────────────────────────────────────────────────────────────

@file_bp.route('/reindex', methods=['POST'])
@require_login
def reindex():
    """
    Remove old FAISS chunks for a file and re-process it.
    Body: { path }
    """
    data = request.get_json() or {}
    path = data.get('path', '').strip()
    if not path:
        return jsonify({"error": "path required"}), 400

    result = _agent().reindex_file(path)
    _log_edit("reindex", path, result)
    return jsonify(result)


# ── AI edits ──────────────────────────────────────────────────────────────────

@file_bp.route('/ai-edit', methods=['POST'])
@require_login
def ai_edit():
    """
    Propose or apply an AI-generated edit to a .txt / .md file.
    Body: { path, instruction, confirmed (bool), proposed_content (optional) }

    First call (confirmed=False): returns diff preview.
    Second call (confirmed=True + proposed_content): saves the file.
    """
    data        = request.get_json() or {}
    path        = data.get('path', '').strip()
    instruction = data.get('instruction', '').strip()
    confirmed   = bool(data.get('confirmed', False))
    proposed    = data.get('proposed_content')

    if not path or not instruction:
        return jsonify({"error": "path and instruction required"}), 400

    # If user confirmed with a proposed_content, save directly
    if confirmed and proposed is not None:
        result = _agent().save_text_file(path, proposed, confirmed=True)
        _log_edit("ai_edit_save", path, result)
        return jsonify(result)

    # Otherwise ask the LLM to produce the edit
    from app_state import get_system_state
    _, _, _, _, _, _, llm = get_system_state()
    result = _agent().apply_ai_edits(path, instruction, llm=llm, confirmed=confirmed)
    _log_edit("ai_edit_propose", path, result)
    return jsonify(result)


# ── Backups ───────────────────────────────────────────────────────────────────

@file_bp.route('/backups', methods=['GET'])
@require_login
def list_backups():
    """List backup files for a document. ?path=..."""
    path = request.args.get('path', '').strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    backups = _agent().list_backups(path)
    return jsonify({"backups": backups})


@file_bp.route('/restore-backup', methods=['POST'])
@require_login
def restore_backup():
    """
    Restore a file from a backup.
    Body: { backup_path, target_path }
    """
    data   = request.get_json() or {}
    backup = data.get('backup_path', '').strip()
    target = data.get('target_path', '').strip()
    if not backup or not target:
        return jsonify({"error": "backup_path and target_path required"}), 400
    result = _agent().restore_backup(backup, target)
    _log_edit("restore_backup", target, result)
    return jsonify(result)


# ── Edit log ──────────────────────────────────────────────────────────────────

@file_bp.route('/edit-log', methods=['GET'])
@require_login
def edit_log():
    """Return recent edit history."""
    n = int(request.args.get('n', 50))
    entries = _agent().get_edit_log(last_n=n)
    return jsonify({"entries": entries, "total": len(entries)})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_edit(action: str, path: str, result: dict) -> None:
    """Write an entry to the edit log."""
    try:
        _agent().append_edit_log({
            "action":    action,
            "path":      path,
            "success":   result.get("success", False),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception as exc:
        logger.warning("[FileRoutes] Failed to log edit: %s", exc)
