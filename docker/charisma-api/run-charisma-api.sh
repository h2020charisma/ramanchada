#!/usr/bin/env sh
set -eux
python -c 'import ramanchada' 2>/dev/null || pip install --user /opt/ramanchada
cd /opt/charisma-api
uwsgi \
  --http-socket 0.0.0.0:5000 \
  --wsgi-file flask_charisma.py \
  --callable app \
  --master \
  --processes 4 \
  --threads 2
