#!/usr/bin/env bash
      # Use the default Wine prefix as C: drive
      export WINEPREFIX="$HOME/.wine"

      WINBOX_EXE="$WINEPREFIX/drive_c/winbox64.exe"

      # Create Wine prefix if it doesn't exist
      if [ ! -d "$WINEPREFIX" ]; then
        echo "Initializing Wine prefix..."
        wineboot
      fi

      # Download Winbox if missing
      if [ ! -f "$WINBOX_EXE" ]; then
        echo "Downloading Winbox..."
        wget -O "$WINBOX_EXE" https://download.mikrotik.com/routeros/winbox/3.43/winbox64.exe
      fi

      HOST="$(hostname)"

      echo "Hostname: $HOST"

      case "$HOST" in
        msi)
          echo "Running Winbox directly (no cage)..."
          exec wine "$WINBOX_EXE"
          ;;
        *)
          echo "Running Winbox inside cage..."
          exec cage -- wine "$WINBOX_EXE"
          ;;
      esac
