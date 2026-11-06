# Local compatibility fixture only; use your existing supported S3 in production.
# Community images are no longer available from their original registries.
FROM golang:1.24.2-bookworm@sha256:79390b5e5af9ee6e7b1173ee3eac7fadf6751a545297672916b59bfa0ecf6f71 AS build
RUN git clone --depth 1 --branch RELEASE.2025-04-22T22-12-26Z https://github.com/minio/minio.git /src
WORKDIR /src
RUN CGO_ENABLED=0 go build -p 2 -trimpath -ldflags="-s -w" -o /out/minio .

FROM debian:bookworm-slim@sha256:99c0b25c46d0214d5a2091092e7908907dcaecd6a962cde0ac700acee222b6f3
COPY --from=build /out/minio /usr/local/bin/minio
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=build /src/LICENSE /licenses/minio-LICENSE
COPY docker/minio-entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh && mkdir /data && chown 10001:10001 /data
USER 10001:10001
ENV MINIO_UPDATE=off
EXPOSE 9000 9001
ENTRYPOINT ["/entrypoint.sh"]
CMD ["server", "/data", "--console-address", ":9001"]
