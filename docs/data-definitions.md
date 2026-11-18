# 데이터 정의 (안 v0.1, 2026-10-14)

> 근거: 2026.09 월간회의 KETI v3.2 (p.9, p.28, p.32, p.41~42). 모두 **안**이며 연세대·성균관대 회신에 따라 바뀐다.
> **정본: [`structure-sets/`](../structure-sets/README.md) 의 JSON** (필드 정의 + 예시). proto 파일([`exec_env_info`](../proto/exec_env_info.proto), [`partition_set`](../proto/partition_set.proto))은 같은 필드 이름의 전송 구현.
> 확정 계약인 `proto/resource_pool.proto`(C-RP-01~03)는 고치지 않았다. 확장이 필요한 필드는 §3.3 에 제안으로만 적는다.

| § | 내용 | 받는 곳 | 순서 |
|---|---|---|---|
| 1 | 실행환경 정보 structure set | 연세대 → 이후 성균관대 | 먼저 |
| 2 | 1세부에서 받을 정보 항목 매핑 | 1세부·위즈온텍 | |
| 3 | 프로파일 데이터 전달 방식 비교 | 연세대 | |
| 4 | 파티션 정보 세트 (KETI 기준) | 성균관대 | 먼저 |

공통 키는 **`workload_id`** (`keti.re.kr/workload-id`) 다. keti-exec-router 가 최상위 오브젝트에서 한 번 발급하고
Pod 로 내려보내므로 Pod 교체·재시도·복제에도 바뀌지 않는다(2026-10-14 클러스터 검증). Kueue Workload 이름은
Pod 마다 새로 생기므로 키로 쓰지 않고, Workload → 소유자 오브젝트 경유로 번호를 찾는다(`exec-env/router/wid_resolve.py`).

---

## 1. 실행환경 정보 structure set (KETI → 연세대)

p.9 "실행 정보" 세 묶음을 그대로 최상위 필드로 둔다. 메시지: `keti.execenv.v1.ExecEnvInfo`.

| p.9 묶음 | 메시지 | 주요 필드 | 출처 | 지금 채울 수 있나 |
|---|---|---|---|---|
| ① 모델·파이프라인 정보 | `ModelPipelineInfo` | model_id·version·uri, **model_graph_uri**(p.28), image·**digest** | 위즈온텍 모델 등록 | 모델 등록 연계 전 — 이미지·digest 만 가능 |
| | | pipeline_id·version, run_id, step_name | KFP | ✅ (Pod 라벨 `pipeline/runid`, Argo 노드 이름) |
| | | workload_kind, exec_type, accel_class, queue_name | keti-exec-router 라벨 | ✅ (라우터 0.2) |
| | | requested_resources, replicas | Pod 스펙 | ✅ |
| | | slo (지연·처리량 목표, 우선순위) | 1세부 운영 요구사항 (§2) | ❌ 수신 형식 미정 |
| 큐·실행 상태 | `ExecStatus` | phase, 제출·승인·시작·종료 시각, restart_count, pod_names, node | Kueue Workload + Pod | ✅ |
| ② 실행 로그·샘플링 | `ExecLogSampling` | 시계열(`MetricSeries`), 창 요약(처리량·p50/p95·avg_gpu_util), log_uri, 짧은 발췌 | 컨테이너 로그, DCGM, 앱 지표 | 🟡 GPU 지표는 dcgm-exporter 가 있음. 처리량·지연은 워크로드가 내보내야 함 |
| ③ 가속기·메모리 상태 | `AcceleratorState` | 장치·파티션, 받은 몫(LSU·공간·시간·PE), 실측(util·메모리·전력·온도) | 자원 풀 매니저, DCGM, NPU 지표 | 🟡 GPU ✅ / NPU 지표 경로 확인 필요 |

설계 원칙:
- **로그 본문은 싣지 않는다.** 위치(`log_uri`)와 수십 줄 발췌만 둔다. 메시지 크기를 예측 가능하게 하려는 것.
- **장치는 워크로드가 쓰는 것만** 싣는다. 장치 전체 상태는 §4 파티션 세트가 맡는다.
- **값 집합을 기존 계약과 맞춘다.** `WorkloadKind` = SliceAllocation `workloadType`(TRAIN/INFER/PREPROC), `avg_gpu_util` = `ModelExecutionHistory.avg_gpu_util`.
- 조회 방식: `Get`(하나), `List`(대상 고르기), `Watch`(실행 중 샘플 스트림). 연세대가 실행 중 프로파일을 만들 때는 `Watch` 를 쓴다.

연세대에 확인할 것:
1. 모델 그래프 형식 (ONNX? framework 원본?), 크기 상한
2. 샘플링 주기와 필요한 지표 이름 목록 — `MetricSeries.name` 의 고정 목록을 같이 정한다
3. 실행 중 스트림(`Watch`)이 필요한지, 종료 후 일괄(`Get`)로 충분한지

## 2. 1세부에서 받을 정보 항목 (p.32) → 어디로 가나

p.32: "이미지와 워크로드는 1:1 이 아님" — 하나의 이미지가 학습·추론 등 여러 워크로드로 쓰일 수 있으므로,
아래 항목은 **이미지가 아니라 워크로드(제출) 단위**로 받는다.

| 구분 | 항목 | K8s 반영 위치 | 라우터 입력 | ExecEnvInfo 필드 |
|---|---|---|---|---|
| 실행 패키지 | 이미지 주소·Digest | `containers[].image` (digest 고정 권장) | – | `model.image`, `image_digest` |
| | 모델 파일·저장 위치 | 볼륨 / env | – | `model.model_uri` |
| | 실행환경·가속기 호환 조건 | **자원 요청** (`keti.re.kr/lsu`, `keti.re.kr/npu-pe`, `nvidia.com/gpu`) + nodeSelector | ✅ 큐 결정 (gpu/npu/cpu) | `requested_resources`, `accel_class` |
| 실행 내용 | 학습인지 추론인지 | **`keti.re.kr/workload-kind` 라벨** + kind (Job / Deployment) | ✅ 유형 라벨, exec-type | `workload_kind`, `exec_type` |
| | 실행 명령·인자 | `command` / `args` | – | – (필요 시 추가) |
| | 데이터 위치·볼륨 | `volumes` / PVC | – | – |
| | 추론 포트 | `containerPort` + Service | – | – |
| | 학습 종료 조건 | `backoffLimit`, `activeDeadlineSeconds`, 학습 인자 | – | – |
| 운영 요구사항 | 필요 자원·우선순위 | requests/limits, `WorkloadPriorityClass` | ✅ 자원 / – | `requested_resources`, `slo.priority` |
| | 응답시간·처리량 목표 | **새 annotation 제안** `keti.re.kr/slo-latency-ms`, `keti.re.kr/slo-throughput` | (스케줄러 입력) | `slo.*` |
| | 복제본·worker 수 | `replicas`, TrainJob `numNodes` | – | `replicas` |
| | 준비·진행 상태 확인 방법 | readiness/liveness probe, 진행률 엔드포인트 | – | – |

정리:
- **라우터가 실제로 쓰는 입력은 두 가지뿐이다:** 자원 요청(큐 결정)과 `workload-kind` 라벨(유형).
  나머지는 K8s 스펙 그대로 실리고, 실행환경 정보(§1)로 수집해 넘긴다.
- **SLO 목표는 지금 K8s 에 들어갈 자리가 없다.** 스케줄러(P4) 승인 판단에 필요하므로 annotation 두 개를 제안한다.
  값이 작고 제출 때 한 번 정해지므로 annotation 이 맞다(§3 의 프로파일과는 성격이 다름).

## 3. 프로파일 데이터 전달 방식 비교 (연세대 → KETI)

대상: 연세대가 만드는 **자원 요구량 프로파일·성능 예측·스케줄러 피드백** (p.28).

| 기준 | A. Webhook 으로 annotation 주입 | B. 별도 구조 (gRPC) |
|---|---|---|
| 크기 | 오브젝트당 annotation 합계 **256 KiB** 상한. 조건별 예측값(디바이스 × 파티션 크기 × 간섭) 표는 넘칠 수 있음 | 제한 없음 (gRPC 기본 4 MiB, 조정 가능) |
| 갱신 주기 | 제출 시점 1회. 실행 중 바뀐 프로파일은 반영 불가 (오브젝트 수정 = API 서버 쓰기 + 모든 watcher 깨움) | 스케줄러가 판단할 때마다 최신값 조회 |
| 결합도 | 연세대 결과가 **제출 경로(Webhook)에 끼어듦** → 프로파일 서버 지연·장애가 제출 지연·라벨 누락으로 번짐 | 제출 경로와 분리. 프로파일이 없으면 "프로파일 대기" 보류 또는 KETI 스코어 폴백 |
| 이력·재현 | 오브젝트가 지워지면 같이 사라짐 | 이력 저장소에 남음 → 결정 이력(p.41)의 입력 스냅샷으로 재사용 |
| 키 | 오브젝트 이름 (Pod 교체 시 바뀜) | model_id + workload_id (불변) |
| 장점 | 오브젝트만 보면 다 보임, 추가 서비스 없음 | 위 넷 |

### 3.1 결론

**당분간 B (별도 gRPC) 유지.** 기존 `GetModelExecutionHistory`(C-RP-02)를 확장해서 쓴다.
annotation 은 §2 의 SLO 처럼 **작고, 제출 때 한 번 정해지는 값**에만 쓴다.

### 3.2 A 로 갈 수 있는 조건 (재검토 기준)
프로파일이 수십 개 스칼라로 줄고, 제출 후 바뀌지 않으며, 연세대 서비스가 제출 경로 SLA(3초 이내 응답)를 보장할 때.

### 3.3 `ModelExecutionHistory` 확장 제안 (resource_pool.proto 는 아직 수정하지 않음)

현재 필드 1~9 는 그대로 두고 10 번부터 추가하는 안. 연세대 회신 후 반영한다.

| 번호 | 필드 | 의미 (p.28 연세대 → KETI) |
|---|---|---|
| 10 | `int64 est_peak_mem_mb` | 메모리 요구량 (SliceAllocation `profile.estPeakMemMb` 와 같은 값) |
| 11 | `double compute_boundness` / 12 `memory_boundness` | 연산·메모리 민감도 (SliceAllocation `profile.*Boundness`) |
| 13 | `double gpu_sensitivity` / 14 `npu_sensitivity` | GPU·NPU 민감도 (p.9) |
| 15 | `repeated PerfPrediction predictions` | 조건별 성능 예측 (device_type, product, space_ratio·pe_count, 예측 처리량·지연) |
| 16 | `double confidence` | 예측 신뢰도 (p.28 연세대 → 성균관대 항목과 같은 값) |
| 17 | `string profile_version` / 18 `Timestamp profiled_at` | 결정 이력 입력 스냅샷에 남길 버전 |

## 4. 파티션 정보 세트 (KETI 기준, 성균관대에 제시)

메시지: `keti.partition.v1.PartitionSet` = 노드 → 장치 → (가능한 분할 범위, 현재 파티션, 사용량).

### 4.1 단위 — 새로 만들지 않고 기존 계약 재사용

| 분할 방식 | 크기 단위 | 기존 계약 | 비고 |
|---|---|---|---|
| `WHOLE` | 장치 1개 | `nvidia.com/gpu` | 233 RTX 5090 (지금은 확장 리소스 미노출) |
| `SPACE_TIME` | SM 공간 비율 + 시간 비율, LSU 환산 | SliceAllocation `spaceRatio/timeRatio/memMb`, resource_pool `lsu_amount`, `keti.re.kr/lsu` | 232 PRO 6000, 현재 할당 가능 LSU 240 |
| `MIG` | MIG 프로필 | SliceAllocation `migPartitionId` | **구성은 성균관대 몫**, KETI 는 ID·크기만 참조 |
| `NPU_PE` | PE 개수 | `keti.re.kr/npu-pe` | 232 RNGD 1장, 현재 할당 가능 PE 8 |

### 4.2 세 층

- **Capability (가능한 범위):** 공간·시간 비율 최소 단위, 최소 공간 비율, 장치당 최대 워크로드 수, 가능한 MIG 프로필, PE 최소 단위, 오버커밋 허용 여부.
  → 성균관대가 파티션 구성안을 낼 때 지켜야 하는 **경계**.
- **Partition (현재 상태):** 파티션 ID(KETI 발급, resource_pool `partition_id` 와 같은 값), 방식, 상태(FREE / RESERVED / BOUND / UNAVAILABLE), 크기, 올라간 workload_id, KETRIS 적용 설정 버전(p.42 "분할 설정 v2").
  → p.42 "계획안 / KETI 승인 / 실제 적용" 비교의 **실제 적용** 열이 여기서 나온다.
- **Usage (장치 전체 여유):** free LSU·공간·시간·메모리·PE, 활성 워크로드 수, 오버커밋 비율, 활용률, 단편화 지수.
  → KETI 자체 스코어(가용 공간·단편화)의 입력.

### 4.3 단편화 지수 (KETI 자체 스코어 요소, 안)

장치 집합에서 `fragmentation = 1 − (가장 큰 단일 장치 여유 / 전체 여유 합)` (여유 합이 0 이면 0).
여유가 한 장치에 몰려 있으면 0, 여러 장치에 잘게 흩어져 있으면 1 에 가깝다. 공간 비율·PE 각각 따로 계산한다.

### 4.4 성균관대에 확인할 것

1. 1회차 입력(필터 전반 + 스코어 일부)과 2회차 입력(디바이스 매칭 최종)이 이 세트의 **어느 층을 참조하는지** — 우리 가정: 1회차는 Capability + Usage, 2회차는 특정 Partition 지정
2. 파티션 구성안의 단위 — 공간 비율·시간 비율로 받는지, MIG 프로필로 받는지
3. 재구성 트리거, 복수안 제출 여부, 체크포인트 (회신 대기 중인 3건 — 그 전까지 승인 구조는 **안**)

---

## 다음
- 이 문서를 연세대(§1, §3) / 성균관대(§4)에 보낼 판으로 정리
- P4 스케줄러: 이 세트를 참조하는 계획안(1·2회차) 메시지와 결정 이력 레코드 정의
- 수집기: ExecEnvInfo 를 실제로 채우는 서버 (KFP·Kueue·라우터 라벨 → ✅ 항목부터)
