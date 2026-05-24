#!/usr/bin/env bash
set -u

SRC="${SRC:-/work/runyi_yang/Worldcept/streetdata/}"
DST="${DST:-/group/streetsplat/}"
LOG_DIR="${LOG_DIR:-/work/runyi_yang/FloWAM/logs}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/rsync_streetdata_alignment_status.txt}"
MONITOR_LOG="${MONITOR_LOG:-${LOG_DIR}/rsync_streetdata_alignment_monitor-$(date -u +%Y%m%d_%H%M%S).log}"
RSYNC_PATTERN='[r]sync -avh --partial --append-verify --info=progress2 streetdata/ /group/streetsplat'

mkdir -p "$LOG_DIR"

write_status() {
  local state="$1"
  local detail="$2"
  {
    printf 'updated_utc=%s\n' "$(date -u --iso-8601=seconds)"
    printf 'state=%s\n' "$state"
    printf 'detail=%s\n' "$detail"
    printf 'source=%s\n' "$SRC"
    printf 'dest=%s\n' "$DST"
    printf 'monitor_log=%s\n' "$MONITOR_LOG"
  } > "$STATUS_FILE"
}

active_rsync_pids() {
  pgrep -u runyi_yang -f "$RSYNC_PATTERN" | tr '\n' ' ' || true
}

{
  printf '[%s] monitor started\n' "$(date -u --iso-8601=seconds)"
  printf 'source=%s\n' "$SRC"
  printf 'dest=%s\n' "$DST"
  printf 'extra destination entries are ignored\n'

  while true; do
    active="$(active_rsync_pids)"
    if [ -z "$active" ]; then
      break
    fi
    write_status waiting_for_rsync "active rsync pid(s): $active"
    printf '[%s] waiting for active rsync pid(s): %s\n' "$(date -u --iso-8601=seconds)" "$active"
    sleep 60
  done

  write_status size_check 'rsync exited; running source-keyed size check'
  size_log="${LOG_DIR}/rsync_streetdata_size_compare-$(date -u +%Y%m%d_%H%M%S).tsv"
  printf 'name\tsource_bytes\tdest_bytes\tdelta_bytes\tstatus\n' > "$size_log"
  total_src=0
  total_dst=0
  bad=0

  for path in "$SRC"*; do
    [ -e "$path" ] || continue
    name="$(basename "$path")"
    s="$(du -sb "$path" | awk '{print $1}')"
    if [ -e "${DST}${name}" ]; then
      d="$(du -sb "${DST}${name}" | awk '{print $1}')"
      delta=$((d - s))
      row_status=ok
      if [ "$delta" -ne 0 ]; then
        row_status=mismatch
        bad=$((bad + 1))
      fi
    else
      d=0
      delta=$((d - s))
      row_status=missing_dest
      bad=$((bad + 1))
    fi
    total_src=$((total_src + s))
    total_dst=$((total_dst + d))
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$s" "$d" "$delta" "$row_status" >> "$size_log"
  done

  printf 'extra_dest_entries_ignored:\n' >> "$size_log"
  for path in "$DST"*; do
    [ -e "$path" ] || continue
    name="$(basename "$path")"
    [ -e "${SRC}${name}" ] || printf '%s\n' "$name" >> "$size_log"
  done
  printf '[%s] size check done: bad_rows=%s delta_bytes=%s log=%s\n' \
    "$(date -u --iso-8601=seconds)" "$bad" "$((total_dst - total_src))" "$size_log"

  write_status checksum_check 'running rsync checksum dry-run; this can take hours on 7.6 TB'
  verify_log="${LOG_DIR}/rsync_streetdata_checksum_verify-$(date -u +%Y%m%d_%H%M%S).log"
  summary="${LOG_DIR}/rsync_streetdata_checksum_verify_latest.summary"
  rsync -anvci --no-times --no-perms --no-owner --no-group --omit-dir-times \
    --out-format='%i\t%n%L' "$SRC" "$DST" > "$verify_log" 2>&1
  rs=$?
  changes="$(awk 'length($1)==11 && $1 ~ /^[<>ch.*][fdLDS]/ {c++} END {print c+0}' "$verify_log")"

  if [ "$rs" -eq 0 ] && [ "$bad" -eq 0 ] && [ "$changes" -eq 0 ]; then
    aligned=YES
    state=aligned
  else
    aligned=NO
    state=mismatch_or_error
  fi

  {
    printf 'updated_utc=%s\n' "$(date -u --iso-8601=seconds)"
    printf 'aligned=%s\n' "$aligned"
    printf 'rsync_exit=%s\n' "$rs"
    printf 'size_bad_rows=%s\n' "$bad"
    printf 'source_total_bytes=%s\n' "$total_src"
    printf 'dest_total_bytes=%s\n' "$total_dst"
    printf 'delta_bytes=%s\n' "$((total_dst - total_src))"
    printf 'checksum_itemized_changes=%s\n' "$changes"
    printf 'source=%s\n' "$SRC"
    printf 'dest=%s\n' "$DST"
    printf 'size_log=%s\n' "$size_log"
    printf 'verify_log=%s\n' "$verify_log"
    printf 'extra_dest_entries_ignored=yes\n'
  } > "$summary"

  write_status "$state" "summary=$summary verify_log=$verify_log"
  printf '[%s] checksum verify done: aligned=%s rsync_exit=%s changes=%s summary=%s verify_log=%s\n' \
    "$(date -u --iso-8601=seconds)" "$aligned" "$rs" "$changes" "$summary" "$verify_log"
} >> "$MONITOR_LOG" 2>&1
