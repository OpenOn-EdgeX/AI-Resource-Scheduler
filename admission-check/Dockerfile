# keti-scheduler (AdmissionCheck 컨트롤러) — 자원 풀 gRPC 클라이언트(grpcio) 외에는 표준 라이브러리
FROM python:3.11-slim
RUN pip install --no-cache-dir grpcio==1.82.1 protobuf==7.35.1
WORKDIR /app
COPY keti_scheduler.py .
COPY rpc/ rpc/
USER 65534
ENTRYPOINT ["python", "/app/keti_scheduler.py"]
