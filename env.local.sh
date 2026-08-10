# 로컬 실행용 환경변수 템플릿
# 사용법: 이 파일을 env.sh 로 복사해 NEED_TO_SET 을 환경에 맞는 값으로 채운 뒤
#         source env.sh 후 에이전트/매니저를 실행한다. (env.sh 는 git 미포함)

# ppo-agent 서빙
export CHECKPOINT_DIR=NEED_TO_SET        # 학습 체크포인트 디렉토리 (예: /path/to/ppo-training/checkpoints)
export SB3_MODEL_PATH=NEED_TO_SET        # 서빙에 로드할 SB3 모델 zip 경로
export API_PORT=NEED_TO_SET              # 에이전트 API 포트 (예: 8081)

# resource-pool-manager
export NODE_NAME=NEED_TO_SET             # 노드 이름 (K8s 배포 시에는 fieldRef 로 자동 주입)
export DEVICE_LSU_CAPACITY=NEED_TO_SET   # 디바이스 LSU 용량 실측값 (예: 174)

# KETRIS 피드백 루프 (metrics_logger/shm 위치)
export KETRIS_DIR=NEED_TO_SET            # KETRIS 런타임 디렉토리 경로
