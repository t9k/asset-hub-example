FROM bitnami/kubectl:latest

USER root

RUN apt-get update
RUN apt-get install -y wget

RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq \
    && chmod +x /usr/bin/yq