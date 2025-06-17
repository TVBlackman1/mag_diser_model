#!/bin/bash

LOCAL_PROJECT_DIR=$(pwd)
LOCAL_IP=$(ipconfig getifaddr en0)
LOCAL_USER=$(whoami)

# 1. Wake-on-LAN
echo "[1/4] Sending WOL to remote server..."
wakeonlan -i "$BROADCAST_MASK_REMOTE_IP" "$REMOTE_MAC"

# 2. Wait for host to be reachable
echo "[2/4] Waiting for $REMOTE_IP to respond to ping..."
while ! ping -c 1 -W 1 "$REMOTE_IP" &>/dev/null; do
  sleep 1
done
echo "Host is up!"

# 3. Connect over SSH (optional, can be skipped if just mounting)
echo "[3/4] Verifying SSH access to $SSH_REMOTE_USER@$REMOTE_IP..."
until ssh -o ConnectTimeout=2 "$SSH_REMOTE_USER@$REMOTE_IP" 'echo "SSH available"' 2>/dev/null; do
  sleep 1
done

# 4. Remote SSHFS mount and open shell in mounted dir
echo "[4/4] Mounting $LOCAL_PROJECT_DIR to $REMOTE_DIRECTORY and entering session..."

ssh -t "$SSH_REMOTE_USER@$REMOTE_IP" "
  export TERM=xterm-256color &&
  mkdir -p '$REMOTE_DIRECTORY' &&
  fusermount -u '$REMOTE_DIRECTORY' 2>/dev/null || true &&
  /usr/bin/sshfs -o allow_other '$LOCAL_USER@$LOCAL_IP:$LOCAL_PROJECT_DIR' '$REMOTE_DIRECTORY' &&
  cd '$REMOTE_DIRECTORY' &&
  exec zsh --login -i
"