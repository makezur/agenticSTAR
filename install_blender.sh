#!/usr/bin/env bash
#
# install_blender.sh — Idempotently install Blender 4.2.5 LTS to $BLENDER_DIR (default ~/blender)
#
# Replicates the exact stock official Linux x64 build currently in use by the
# harness (Blender 4.2.5 LTS, build hash 27be13ad2d5a). Safe to re-run: if the
# target version is already present it does nothing. Pass --force to reinstall.
#
# Usage:
#   ./install_blender.sh            # install if missing / wrong version
#   ./install_blender.sh --force    # wipe and reinstall regardless
#
set -euo pipefail

# --- Pinned to reproduce the existing install exactly ------------------------
BLENDER_VERSION="4.2.5"
BLENDER_SERIES="4.2"                                    # download.blender.org dir
EXPECTED_MD5="9b0af377b2f37f733a83c87d8486fadb"
TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
REL_PATH="release/Blender${BLENDER_SERIES}/${TARBALL}"
# Mirror candidates, tried in order. download.blender.org refuses some datacenter
# egress ranges, so the official
# project mirrors — which serve the identical file and don't blanket-block
# datacenters — are the fallback. The pinned MD5 below verifies whichever wins,
# so any mirror is as trustworthy as the canonical host.
MIRRORS=(
    "https://download.blender.org/${REL_PATH}"
    "https://mirrors.ocf.berkeley.edu/blender/${REL_PATH}"
    "https://ftp.nluug.nl/pub/graphics/blender/${REL_PATH}"
)

# Install location. The sandbox launchers mount this dir at /home/ubuntu/blender
# (the path harness/render.sh defaults to inside the sandbox); on the host, point
# BLENDER=$BLENDER_DIR/blender at it.
DEST="${BLENDER_DIR:-$HOME/blender}"
BLENDER_BIN="${DEST}/blender"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

log() { printf '==> %s\n' "$*"; }
err() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- Idempotency check -------------------------------------------------------
installed_version() {
    [[ -x "$BLENDER_BIN" ]] || return 1
    "$BLENDER_BIN" --version 2>/dev/null | awk '/^Blender/ {print $2; exit}'
}

if [[ "$FORCE" -eq 0 ]]; then
    if current="$(installed_version)" && [[ "$current" == "$BLENDER_VERSION" ]]; then
        log "Blender ${BLENDER_VERSION} already installed at ${BLENDER_BIN} — nothing to do."
        exit 0
    fi
    if [[ -n "${current:-}" ]]; then
        log "Found Blender ${current} at ${BLENDER_BIN}; replacing with ${BLENDER_VERSION}."
    fi
fi

# --- Preconditions -----------------------------------------------------------
arch="$(uname -m)"
[[ "$arch" == "x86_64" ]] || err "This build is x86_64 only; host arch is ${arch}."

for tool in curl tar xz md5sum; do
    command -v "$tool" >/dev/null 2>&1 || err "Required tool '${tool}' not found in PATH."
done

# --- Download (cached) -------------------------------------------------------
WORK="$(mktemp -d /tmp/blender-install.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

DL="${WORK}/${TARBALL}"
downloaded=0
for url in "${MIRRORS[@]}"; do
    log "Downloading ${url}"
    if curl -fSL --retry 3 --retry-delay 5 -o "$DL" "$url"; then
        downloaded=1
        break
    fi
    log "mirror failed (curl exit $?), trying next"
done
[[ "$downloaded" -eq 1 ]] || \
    err "all ${#MIRRORS[@]} Blender mirrors failed for ${TARBALL}"

log "Verifying MD5 (${EXPECTED_MD5})"
actual_md5="$(md5sum "$DL" | awk '{print $1}')"
[[ "$actual_md5" == "$EXPECTED_MD5" ]] || \
    err "Checksum mismatch: got ${actual_md5}, expected ${EXPECTED_MD5}."

# --- Extract -----------------------------------------------------------------
log "Extracting archive"
tar -xJf "$DL" -C "$WORK"
SRC="${WORK}/blender-${BLENDER_VERSION}-linux-x64"
[[ -x "${SRC}/blender" ]] || err "Extracted tree missing blender binary at ${SRC}."

# --- Install atomically ------------------------------------------------------
# Replace the whole tree so no stale files linger from a previous version.
mkdir -p "$(dirname "$DEST")"
if [[ -e "$DEST" ]]; then
    BACKUP="${DEST}.old.$$"
    log "Moving existing ${DEST} aside to ${BACKUP}"
    mv "$DEST" "$BACKUP"
    trap 'rm -rf "$WORK" "$BACKUP"' EXIT
fi

log "Installing to ${DEST}"
mv "$SRC" "$DEST"

# --- Verify ------------------------------------------------------------------
final="$(installed_version)" || err "Post-install: blender binary not runnable."
[[ "$final" == "$BLENDER_VERSION" ]] || \
    err "Post-install version mismatch: got ${final}, expected ${BLENDER_VERSION}."

log "Success: Blender ${final} installed at ${BLENDER_BIN}"
"$BLENDER_BIN" --version | head -1
