# Swap the live chat worker to v3cond (palette/style), and roll back

The copy lives in `$CHAT_COND/`. It is not enabled. The live unit
`pixie-chat.service` still starts `$CHAT_FAST/worker_v3.py`.
Checkpoints, votes, and the v3fast tree are not modified.

Cond is backwards compatible: a job with no `cond` field uses the same default
tensor the live worker already prefixes (NO_PALETTE + ANY style). Old staged/live
servers that never send `cond` keep working.

Lock port for this copy is **8795** (live fast uses 8794, older v3 uses 8793).
Never run two pollers at once.

## Swap (after review)

On the inference machine, with `CHAT_COND` and `CHAT_FAST` set to the two worker directories:

```bash
# 1. Confirm the copy is in place and the key is linked (never print the key).
ls $CHAT_COND/worker_v3.py
# worker.key should be a symlink to the live fast key
test -L $CHAT_COND/worker.key || ln -s "$CHAT_FAST/worker.key" $CHAT_COND/worker.key
chmod -h 600 $CHAT_COND/worker.key 2>/dev/null || true

# 2. Stop live chat. Public /chat will be down for about a minute.
systemctl --user stop pixie-chat
# wait until lock 8794 is free and VRAM drops (from ~4.5 GB to a few hundred MB)
ss -ltnp | grep -E '8793|8794|8795' || true
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv

# 3. Point the same unit at the cond worker (keeps the unit name).
mkdir -p ~/.config/systemd/user
cp $CHAT_COND/pixie-chat-cond.service ~/.config/systemd/user/pixie-chat.service
systemctl --user daemon-reload
# The cond unit uses ConditionPathExists=!$CHAT_COND/DISABLE
systemctl --user start pixie-chat
systemctl --user is-active pixie-chat
# Wait for READY in the log, then generate one sprite on the site (with and without a palette).
tail -20 $CHAT_COND/worker.log
```

Public chat should answer in about 650 ms, same as v3fast, because the sequence
length does not change when a palette/style is set.

## Rollback to v3fast

```bash
systemctl --user stop pixie-chat
ss -ltnp | grep -E '8793|8794|8795' || true
# restore the fast unit file
cp $CHAT_FAST/pixie-chat.service ~/.config/systemd/user/pixie-chat.service
# if that backup is missing, the original unit is also at
#   /workspace copies, or recreate from this repo's v3fast unit:
#   ExecStart=.../chat-worker-v3fast/worker_v3.py
#   ConditionPathExists=!/path/to/pixie
systemctl --user daemon-reload
systemctl --user start pixie-chat
tail -20 $CHAT_FAST/worker.log
```

The cond tree can stay on disk.

## Training window

`touch $CHAT_COND/DISABLE && systemctl --user stop pixie-chat`
(after the swap). The v3fast DISABLE path no longer applies once the unit points here.
