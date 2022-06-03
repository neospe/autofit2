#
# training container
# mount volume under /service
#

FROM python:3.10-slim

COPY requirements.txt ./

RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends gcc python3-dev; \
  apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
	rm -rf /var/lib/apt/lists/*; \
  python -m pip install --upgrade pip; \
  pip install --no-cache-dir -r requirements.txt

WORKDIR /service
CMD ["/bin/bash"]
