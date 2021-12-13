FROM python:3.9-slim

LABEL org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.description="CHARISMA RamanChada API service" \
    org.label-schema.name="charisma-api" \
    org.label-schema.schema-version="1.0.0" \
    org.label-schema.usage="https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization/-/blob/master/README.md" \
    org.label-schema.vcs-url="https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vendor="IDEAconsult" \
    org.label-schema.version="latest"

RUN set -eux; \
    apt_mark_manual="$(apt-mark showmanual)"; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        gcc \
        git \
        libc6-dev \
    ; \
    pip install uwsgi; \
    apt-mark auto '.*' > /dev/null; \
    apt-mark manual $apt_mark_manual git; \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
    rm -rf /var/lib/apt/lists/*
        
RUN pip install \
    IPython \
    flask_restful \
    h5py \
    h5pyd \
    pandas

RUN useradd \
        --comment "CHARISMA RamanChada API service" \
        --home-dir /home/charisma-api \
        --create-home \
        --system \
        --user-group \
        charisma-api

USER charisma-api
CMD ["/bin/bash"]
